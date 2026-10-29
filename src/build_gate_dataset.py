"""Build leakage-separated gate features and hidden conformity labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SENSOR_FEATURES = ["cf1", "atm", "rh", "temperature_c"]
KEY_COLUMNS = [
    "block_id",
    "protocol",
    "role",
    "dataset",
    "site",
    "device",
    "model",
    "reference_budget_days",
    "adaptation",
]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def summarize(series: pd.Series, prefix: str) -> dict:
    values = series.dropna().to_numpy(dtype=float)
    if not len(values):
        return {f"{prefix}_{name}": np.nan for name in ["mean", "std", "q05", "q50", "q95", "min", "max"]}
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        f"{prefix}_q05": float(np.quantile(values, 0.05)),
        f"{prefix}_q50": float(np.quantile(values, 0.50)),
        f"{prefix}_q95": float(np.quantile(values, 0.95)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
    }


def source_support(development: pd.DataFrame, dataset: str, site: str, device: str) -> dict:
    if dataset == "QUANT":
        source = development[~development["site"].eq(site) & ~development["device"].eq(device)]
    else:
        source = development
    values = source[SENSOR_FEATURES]
    return {
        "mean": values.mean(),
        "std": values.std().replace(0.0, 1.0),
        "q01": values.quantile(0.01),
        "q99": values.quantile(0.99),
    }


def audit_features(audit: pd.DataFrame, budget: int) -> tuple[dict, float]:
    empty = {
        "audit_residual_mean": np.nan,
        "audit_residual_std": np.nan,
        "audit_abs_error_mean": np.nan,
        "audit_abs_error_max": np.nan,
        "audit_last_residual": np.nan,
        "audit_residual_trend": np.nan,
    }
    if budget == 0:
        return empty, 0.0
    if len(audit) != budget:
        raise ValueError(f"Expected {budget} audit rows, found {len(audit)}")
    residual = (audit["reference_pm25"] - audit["prediction_pm25"]).to_numpy(dtype=float)
    trend = float(np.polyfit(np.arange(len(residual)), residual, 1)[0]) if len(residual) >= 3 else np.nan
    values = {
        "audit_residual_mean": float(np.mean(residual)),
        "audit_residual_std": float(np.std(residual, ddof=1)) if len(residual) > 1 else 0.0,
        "audit_abs_error_mean": float(np.mean(np.abs(residual))),
        "audit_abs_error_max": float(np.max(np.abs(residual))),
        "audit_last_residual": float(residual[-1]),
        "audit_residual_trend": trend,
    }
    offset = float(np.mean(residual) * budget / (budget + 3.0))
    return values, offset


def validate_audit_dates(manifest: pd.DataFrame, metrics: pd.DataFrame) -> int:
    """Verify audit dates for every block-budget pair used by the experiment."""
    checked = 0
    used = metrics[["block_id", "reference_budget_days"]].drop_duplicates()
    selected = used.merge(manifest, on="block_id", how="left", validate="many_to_one")
    if selected["evaluation_start"].isna().any():
        raise ValueError("Experiment metrics contain blocks absent from the manifest")
    for window in selected.itertuples(index=False):
        evaluation_start = pd.Timestamp(window.evaluation_start)
        budget = int(window.reference_budget_days)
        value = getattr(window, f"audit_dates_b{budget}")
        audit_dates = [] if pd.isna(value) else [
            pd.Timestamp(date) for date in str(value).split(";") if date
        ]
        if len(audit_dates) != budget:
            raise ValueError(
                f"{window.block_id} budget {budget} has {len(audit_dates)} audit dates"
            )
        if any(date >= evaluation_start for date in audit_dates):
            raise ValueError(f"Audit date overlaps assessment block {window.block_id}")
        checked += len(audit_dates)
    return checked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "study_config.json")
    args = parser.parse_args()
    config = load_json(args.config)

    daily = pd.read_csv(PROJECT_ROOT / config["derived_data"]["daily"], parse_dates=["date"])
    predictions = pd.read_csv(
        PROJECT_ROOT / "data" / "processed" / "base_daily_predictions.csv.gz",
        parse_dates=["date"],
    )
    manifest = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "deployment_windows.csv")
    metrics = pd.read_csv(PROJECT_ROOT / "results" / "baseline_window_metrics.csv")
    development = daily[daily["dataset"].eq("QUANT")].copy()
    checked_audit_dates = validate_audit_dates(manifest, metrics)

    daily_groups = {
        key: frame.sort_values("date").set_index("date", drop=False)
        for key, frame in daily.groupby(["dataset", "site", "device"], sort=False)
    }
    prediction_groups = {
        key: frame.sort_values("date").set_index("date", drop=False)
        for key, frame in predictions.groupby(["dataset", "site", "device"], sort=False)
    }
    disagreement_groups = {}
    for key, frame in predictions.groupby(["dataset", "site", "device"], sort=False):
        wide = frame.pivot(index="date", columns="model", values="prediction_pm25")
        disagreement_groups[key] = wide.std(axis=1, ddof=1)
    support_cache = {
        key: source_support(development, *key)
        for key in daily_groups
    }
    manifest_by_id = manifest.set_index("block_id", drop=False)

    common_cache: dict[str, dict] = {}
    records: list[dict] = []
    for row in metrics.itertuples(index=False):
        window = manifest_by_id.loc[row.block_id]
        group_key = (row.dataset, row.site, row.device)
        daily_group = daily_groups[group_key]
        prediction_group = prediction_groups[group_key]
        start = pd.Timestamp(row.evaluation_start)
        end = pd.Timestamp(row.evaluation_end)

        if row.block_id not in common_cache:
            evaluation_unlabeled = daily_group.loc[
                (daily_group.index >= start) & (daily_group.index <= end),
                SENSOR_FEATURES + ["valid_hours", "cf1_channel_gap", "atm_channel_gap"],
            ].copy()
            common = {
                "evaluation_valid_days": int(len(evaluation_unlabeled)),
                "evaluation_mean_valid_hours": float(evaluation_unlabeled["valid_hours"].mean()),
                "start_month_sin": float(np.sin(2 * np.pi * start.month / 12.0)),
                "start_month_cos": float(np.cos(2 * np.pi * start.month / 12.0)),
            }
            for feature in SENSOR_FEATURES:
                common.update(summarize(evaluation_unlabeled[feature], f"evaluation_{feature}"))

            support = support_cache[group_key]
            z = (evaluation_unlabeled[SENSOR_FEATURES] - support["mean"]) / support["std"]
            outside = (
                evaluation_unlabeled[SENSOR_FEATURES].lt(support["q01"])
                | evaluation_unlabeled[SENSOR_FEATURES].gt(support["q99"])
            )
            common["support_mean_z_l2"] = float(np.sqrt(np.square(z).sum(axis=1)).mean())
            common["support_max_abs_z"] = float(np.abs(z).max(axis=1).max())
            common["support_outside_fraction"] = float(outside.to_numpy().mean())
            disagreement = disagreement_groups[group_key].loc[start:end]
            common.update(summarize(disagreement, "evaluation_model_disagreement"))
            common.update(summarize(evaluation_unlabeled["cf1_channel_gap"], "evaluation_cf1_channel_gap"))
            common["channel_gap_available"] = bool(evaluation_unlabeled["cf1_channel_gap"].notna().any())
            common_cache[row.block_id] = common

        audit_text = window[f"audit_dates_b{row.reference_budget_days}"]
        audit_dates = [] if pd.isna(audit_text) else [
            pd.Timestamp(value) for value in str(audit_text).split(";") if value
        ]
        model_predictions = prediction_group[prediction_group["model"].eq(row.model)]
        audit_predictions = model_predictions.loc[
            model_predictions.index.isin(audit_dates),
            ["date", "prediction_pm25"],
        ].reset_index(drop=True)
        audit_reference = daily_group.loc[
            daily_group.index.isin(audit_dates),
            ["date", "reference_pm25"],
        ].reset_index(drop=True)
        audit = audit_predictions.merge(
            audit_reference,
            on="date",
            how="inner",
            validate="one_to_one",
        )
        audit_values, expected_offset = audit_features(audit, int(row.reference_budget_days))
        offset = expected_offset if row.adaptation == "shrunk_offset" else 0.0

        evaluation_prediction = model_predictions.loc[
            (model_predictions.index >= start) & (model_predictions.index <= end),
            "prediction_pm25",
        ].clip(lower=0.0) + offset
        record = {column: getattr(row, column) for column in KEY_COLUMNS}
        record.update(common_cache[row.block_id])
        record.update(audit_values)
        record["applied_audit_offset"] = offset
        record.update(summarize(evaluation_prediction, "evaluation_prediction"))
        records.append(record)

    features = pd.DataFrame(records)
    targets = config["conformity_targets"]
    labels = metrics[KEY_COLUMNS + ["conforms", "error_target_pass", "slope", "intercept", "r2", "rmse", "nrmse_percent"]].copy()
    labels["slope_pass"] = labels["slope"].between(targets["slope_min"], targets["slope_max"])
    labels["intercept_pass"] = labels["intercept"].between(
        targets["intercept_min_ug_m3"], targets["intercept_max_ug_m3"]
    )
    labels["r2_pass"] = labels["r2"].ge(targets["r2_min"])

    if features.duplicated(KEY_COLUMNS).any() or labels.duplicated(KEY_COLUMNS).any():
        raise ValueError("Gate keys are not unique")
    forbidden = {"reference_pm25", "conforms", "slope", "intercept", "r2", "rmse", "nrmse_percent"}
    leaked = forbidden.intersection(features.columns)
    if leaked:
        raise ValueError(f"Forbidden assessment fields in gate features: {sorted(leaked)}")
    if not features[KEY_COLUMNS].equals(labels[KEY_COLUMNS]):
        raise ValueError("Feature-label key order mismatch")

    feature_path = PROJECT_ROOT / "data" / "processed" / "gate_features.csv.gz"
    development_label_path = (
        PROJECT_ROOT / "data" / "private_labels" / "gate_labels_development.csv.gz"
    )
    external_label_path = (
        PROJECT_ROOT / "data" / "private_labels" / "gate_labels_external_locked.csv.gz"
    )
    audit_path = PROJECT_ROOT / "results" / "gate_dataset_audit.json"
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    development_label_path.parent.mkdir(parents=True, exist_ok=True)
    deterministic_gzip = {"method": "gzip", "mtime": 0}
    features.to_csv(feature_path, index=False, compression=deterministic_gzip)
    development_labels = labels[labels["role"].eq("development")].copy()
    external_labels = labels[labels["role"].eq("external_test")].copy()
    development_labels.to_csv(
        development_label_path,
        index=False,
        compression=deterministic_gzip,
    )
    external_labels.to_csv(
        external_label_path,
        index=False,
        compression=deterministic_gzip,
    )
    audit_report = {
        "rows": int(len(features)),
        "feature_columns": int(len(features.columns) - len(KEY_COLUMNS)),
        "key_columns": KEY_COLUMNS,
        "forbidden_feature_columns": sorted(leaked),
        "assessment_reference_in_features": False,
        "feature_label_key_match": True,
        "development_label_rows": int(len(development_labels)),
        "external_locked_label_rows": int(len(external_labels)),
        "labels_physically_separated_by_role": True,
        "audit_dates_checked": checked_audit_dates,
        "audit_dates_strictly_preassessment": True,
        "status": "PASS",
    }
    with audit_path.open("w", encoding="utf-8") as handle:
        json.dump(audit_report, handle, indent=2, ensure_ascii=True)
    print(json.dumps(audit_report, indent=2))


if __name__ == "__main__":
    main()
