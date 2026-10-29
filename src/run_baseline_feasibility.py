"""Fit calibration baselines and evaluate frozen deployment conformity labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["cf1", "atm", "rh", "temperature_c"]


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def fitted_models(seed: int) -> dict:
    return {
        "linear": LinearRegression(),
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=5,
            max_features=0.8,
            n_jobs=1,
            random_state=seed,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=250,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=seed,
        ),
    }


def predict_group(train: pd.DataFrame, target: pd.DataFrame, seed: int) -> pd.DataFrame:
    result = target[["dataset", "site", "device", "date"]].copy()
    result["raw_cf1"] = target["cf1"].to_numpy()
    result["epa_correction"] = (
        0.524 * target["cf1"].to_numpy() - 0.0862 * target["rh"].to_numpy() + 5.75
    )
    x_train = train[FEATURES].to_numpy()
    y_train = train["reference_pm25"].to_numpy()
    x_target = target[FEATURES].to_numpy()
    for name, model in fitted_models(seed).items():
        model.fit(x_train, y_train)
        result[name] = model.predict(x_target)
    prediction_columns = [column for column in result if column not in ["dataset", "site", "device", "date"]]
    result[prediction_columns] = result[prediction_columns].clip(lower=0.0)
    return result.melt(
        id_vars=["dataset", "site", "device", "date"],
        var_name="model",
        value_name="prediction_pm25",
    )


def build_predictions(daily: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, dict]:
    development = daily[daily["dataset"].eq("QUANT")].copy()
    external = daily[daily["dataset"].eq("CampmierIndia")].copy()
    outputs: list[pd.DataFrame] = []
    audit = {"development_target_groups": [], "external_training_rows": int(len(development))}

    for (site, device), target in development.groupby(["site", "device"], sort=True):
        train = development[~development["site"].eq(site) & ~development["device"].eq(device)]
        if train.empty:
            raise ValueError(f"No training rows remain for {site}/{device}")
        outputs.append(predict_group(train, target, seed))
        audit["development_target_groups"].append(
            {
                "site": site,
                "device": device,
                "target_rows": int(len(target)),
                "training_rows": int(len(train)),
                "target_site_in_training": bool(train["site"].eq(site).any()),
                "target_device_in_training": bool(train["device"].eq(device).any()),
            }
        )

    if not external.empty:
        outputs.append(predict_group(development, external, seed))
    predictions = pd.concat(outputs, ignore_index=True)
    if not np.isfinite(predictions["prediction_pm25"]).all():
        raise ValueError("Non-finite baseline prediction detected")
    return predictions, audit


def conformity_metrics(reference: np.ndarray, prediction: np.ndarray, targets: dict) -> dict:
    regression = linregress(reference, prediction)
    residual = prediction - reference
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    reference_mean = float(np.mean(reference))
    nrmse = float(100.0 * rmse / reference_mean) if reference_mean > 0 else np.inf
    slope = float(regression.slope)
    intercept = float(regression.intercept)
    r2 = float(regression.rvalue**2)
    error_pass = rmse < targets["rmse_max_ug_m3"] or nrmse < targets["nrmse_max_percent"]
    conforms = (
        targets["slope_min"] <= slope <= targets["slope_max"]
        and targets["intercept_min_ug_m3"] <= intercept <= targets["intercept_max_ug_m3"]
        and r2 >= targets["r2_min"]
        and error_pass
    )
    return {
        "slope": slope,
        "intercept": intercept,
        "r2": r2,
        "bias": float(np.mean(residual)),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": rmse,
        "nrmse_percent": nrmse,
        "error_target_pass": bool(error_pass),
        "conforms": bool(conforms),
    }


def evaluate_windows(
    daily: pd.DataFrame,
    predictions: pd.DataFrame,
    manifest: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    joined = daily.merge(
        predictions,
        on=["dataset", "site", "device", "date"],
        how="inner",
        validate="one_to_many",
    )
    budgets = sorted(int(value) for value in config["target_reference_budgets_days"])
    records: list[dict] = []
    for window in manifest.itertuples(index=False):
        group = joined[
            joined["dataset"].eq(window.dataset)
            & joined["site"].eq(window.site)
            & joined["device"].eq(window.device)
        ]
        start = pd.Timestamp(window.evaluation_start)
        end = pd.Timestamp(window.evaluation_end)
        evaluation = group[group["date"].between(start, end)]
        for model, model_evaluation in evaluation.groupby("model", sort=True):
            for budget in budgets:
                if not getattr(window, f"eligible_b{budget}"):
                    continue
                audit_dates_text = getattr(window, f"audit_dates_b{budget}")
                audit_dates = [] if pd.isna(audit_dates_text) else [
                    pd.Timestamp(value) for value in str(audit_dates_text).split(";") if value
                ]
                model_audit = group[group["model"].eq(model) & group["date"].isin(audit_dates)]
                adaptations = ["none"] if budget == 0 else ["none", "shrunk_offset"]
                for adaptation in adaptations:
                    offset = 0.0
                    if adaptation == "shrunk_offset":
                        if len(model_audit) != budget:
                            raise ValueError(f"Audit row mismatch for {window.block_id}/{model}/b{budget}")
                        raw_offset = float(
                            np.mean(model_audit["reference_pm25"] - model_audit["prediction_pm25"])
                        )
                        offset = raw_offset * budget / (budget + 3.0)
                    prediction = model_evaluation["prediction_pm25"].to_numpy() + offset
                    prediction = np.clip(prediction, 0.0, None)
                    reference = model_evaluation["reference_pm25"].to_numpy()
                    metrics = conformity_metrics(reference, prediction, config["conformity_targets"])
                    records.append(
                        {
                            "block_id": window.block_id,
                            "protocol": window.protocol,
                            "role": window.role,
                            "dataset": window.dataset,
                            "site": window.site,
                            "device": window.device,
                            "evaluation_start": window.evaluation_start,
                            "evaluation_end": window.evaluation_end,
                            "model": model,
                            "reference_budget_days": budget,
                            "adaptation": adaptation,
                            "audit_offset": offset,
                            "valid_days": int(len(model_evaluation)),
                            **metrics,
                        }
                    )
    return pd.DataFrame(records)


def summarize(metrics: pd.DataFrame, prediction_audit: dict) -> dict:
    primary = metrics[metrics["protocol"].eq("primary")]
    counts = (
        primary.groupby(["role", "model", "reference_budget_days", "adaptation"])["conforms"]
        .agg(total="size", conforming="sum")
        .reset_index()
    )
    counts["nonconforming"] = counts["total"] - counts["conforming"]
    development_candidates = counts[
        counts["role"].eq("development")
        & counts["conforming"].ge(20)
        & counts["nonconforming"].ge(20)
    ]
    external_candidates = counts[
        counts["role"].eq("external_test")
        & counts["conforming"].gt(0)
        & counts["nonconforming"].gt(0)
    ]
    key_columns = ["model", "reference_budget_days", "adaptation"]
    qualified = development_candidates[key_columns].merge(
        external_candidates[key_columns], on=key_columns, how="inner", validate="one_to_one"
    )
    training_leakage = any(
        item["target_site_in_training"] or item["target_device_in_training"]
        for item in prediction_audit["development_target_groups"]
    )
    return {
        "integrity": {
            "india_used_for_model_fitting": False,
            "development_site_or_device_leakage": training_leakage,
            "prediction_file_contains_reference": False,
            "status": "PASS" if not training_leakage else "FAIL",
        },
        "primary_counts": counts.to_dict(orient="records"),
        "go_rule": {
            "development_mixed_candidates": int(len(development_candidates)),
            "external_mixed_candidates": int(len(external_candidates)),
            "same_variant_candidates": qualified.to_dict(orient="records"),
            "status": "PASS" if len(qualified) > 0 else "FAIL",
        },
        "prediction_training_audit": prediction_audit,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "study_config.json")
    args = parser.parse_args()
    config = load_config(args.config)
    daily = pd.read_csv(PROJECT_ROOT / config["derived_data"]["daily"], parse_dates=["date"])
    manifest = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "deployment_windows.csv")

    predictions, prediction_audit = build_predictions(daily, int(config["random_seed"]))
    metrics = evaluate_windows(daily, predictions, manifest, config)
    summary = summarize(metrics, prediction_audit)
    if summary["integrity"]["status"] != "PASS":
        raise RuntimeError(json.dumps(summary["integrity"], indent=2))

    prediction_path = PROJECT_ROOT / "data" / "processed" / "base_daily_predictions.csv.gz"
    metrics_path = PROJECT_ROOT / "results" / "baseline_window_metrics.csv"
    summary_path = PROJECT_ROOT / "results" / "baseline_feasibility.json"
    predictions.to_csv(
        prediction_path,
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    metrics.to_csv(metrics_path, index=False)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)
    print(json.dumps({"integrity": summary["integrity"], "go_rule": summary["go_rule"]}, indent=2))


if __name__ == "__main__":
    main()
