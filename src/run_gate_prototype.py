"""Run leakage-controlled gate development and the locked external evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import NormalDist

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
NON_PREDICTOR_COLUMNS = {"block_id", "protocol", "role", "dataset", "site", "device"}
CATEGORICAL_COLUMNS = ["model", "adaptation"]
LEARNED_METHODS = ["logistic_gate", "hist_gradient_boosting_gate"]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_estimator(method: str, feature_columns: list[str], seed: int) -> Pipeline:
    categorical = [column for column in CATEGORICAL_COLUMNS if column in feature_columns]
    numeric = [column for column in feature_columns if column not in categorical]
    preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "one_hot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )
    if method == "logistic_gate":
        classifier = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=seed)
    elif method == "hist_gradient_boosting_gate":
        classifier = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=160,
            max_leaf_nodes=15,
            min_samples_leaf=30,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown learned method: {method}")
    return Pipeline([("preprocess", preprocessor), ("classifier", classifier)])


def grouped_oof_scores(
    method: str,
    frame: pd.DataFrame,
    labels: np.ndarray,
    groups: np.ndarray,
    feature_columns: list[str],
    seed: int,
    max_splits: int = 4,
) -> np.ndarray:
    unique_groups = np.unique(groups)
    splitter = GroupKFold(n_splits=min(max_splits, len(unique_groups)))
    scores = np.full(len(frame), np.nan, dtype=float)
    for fold, (train_index, test_index) in enumerate(splitter.split(frame, labels, groups)):
        estimator = build_estimator(method, feature_columns, seed + fold)
        estimator.fit(frame.iloc[train_index][feature_columns], labels[train_index])
        scores[test_index] = estimator.predict_proba(frame.iloc[test_index][feature_columns])[:, 1]
    if np.isnan(scores).any():
        raise ValueError("Grouped OOF scoring left unfilled rows")
    return scores


def wilson_upper(point_risk: float, effective_n: float, confidence: float) -> float:
    if effective_n <= 0:
        return 1.0
    z = NormalDist().inv_cdf(confidence)
    denominator = 1.0 + z * z / effective_n
    center = point_risk + z * z / (2.0 * effective_n)
    radius = z * np.sqrt(
        point_risk * (1.0 - point_risk) / effective_n
        + z * z / (4.0 * effective_n * effective_n)
    )
    return float(min(1.0, (center + radius) / denominator))


def risk_statistics(
    errors: np.ndarray,
    selected: np.ndarray,
    block_ids: np.ndarray,
    confidence: float,
    replicates: int,
    seed: int,
) -> dict:
    selected_count = int(selected.sum())
    if selected_count == 0:
        return {
            "selected_rows": 0,
            "selected_blocks": 0,
            "point_risk": np.nan,
            "bootstrap_upper": 1.0,
            "finite_sample_upper": 1.0,
            "upper_bound": 1.0,
        }
    selected_blocks, inverse = np.unique(block_ids[selected], return_inverse=True)
    selected_errors = errors[selected].astype(float)
    accepted_per_block = np.bincount(inverse, minlength=len(selected_blocks)).astype(float)
    errors_per_block = np.bincount(
        inverse,
        weights=selected_errors,
        minlength=len(selected_blocks),
    )
    point_risk = float(selected_errors.mean())
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(
        len(selected_blocks),
        np.full(len(selected_blocks), 1.0 / len(selected_blocks)),
        size=replicates,
    )
    bootstrap_selected = draws @ accepted_per_block
    bootstrap_errors = draws @ errors_per_block
    bootstrap_risk = np.divide(
        bootstrap_errors,
        bootstrap_selected,
        out=np.ones_like(bootstrap_errors, dtype=float),
        where=bootstrap_selected > 0,
    )
    bootstrap_upper = float(np.quantile(bootstrap_risk, confidence, method="higher"))
    effective_n = float(accepted_per_block.sum() ** 2 / np.square(accepted_per_block).sum())
    finite_sample_upper = wilson_upper(point_risk, effective_n, confidence)
    return {
        "selected_rows": selected_count,
        "selected_blocks": int(len(selected_blocks)),
        "point_risk": point_risk,
        "bootstrap_upper": bootstrap_upper,
        "finite_sample_upper": finite_sample_upper,
        "upper_bound": max(bootstrap_upper, finite_sample_upper),
        "effective_blocks": effective_n,
    }


def candidate_thresholds(scores: np.ndarray, direction: str) -> np.ndarray:
    finite = scores[np.isfinite(scores)]
    if not len(finite):
        return np.array([], dtype=float)
    quantiles = np.linspace(0.0, 1.0, 101)
    candidates = np.unique(np.quantile(finite, quantiles))
    return np.sort(candidates) if direction == "release" else np.sort(candidates)[::-1]


def calibrate_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    block_ids: np.ndarray,
    direction: str,
    risk_target: float,
    confidence: float,
    replicates: int,
    minimum_coverage: float,
    minimum_blocks: int,
    seed: int,
) -> dict:
    errors = (labels == 0) if direction == "release" else (labels == 1)
    best: dict | None = None
    for index, threshold in enumerate(candidate_thresholds(scores, direction)):
        selected = scores >= threshold if direction == "release" else scores <= threshold
        coverage = float(selected.mean())
        if coverage < minimum_coverage:
            continue
        stats = risk_statistics(
            errors,
            selected,
            block_ids,
            confidence,
            replicates,
            seed + index,
        )
        if stats["selected_blocks"] < minimum_blocks or stats["upper_bound"] > risk_target:
            continue
        candidate = {"threshold": float(threshold), "coverage": coverage, **stats}
        if best is None or candidate["coverage"] > best["coverage"]:
            best = candidate
    if best is None:
        return {
            "threshold": float("inf") if direction == "release" else float("-inf"),
            "coverage": 0.0,
            "selected_rows": 0,
            "selected_blocks": 0,
            "point_risk": np.nan,
            "bootstrap_upper": 1.0,
            "finite_sample_upper": 1.0,
            "upper_bound": 1.0,
            "feasible": False,
        }
    best["feasible"] = True
    return best


def calibrate_decisions(
    labels: np.ndarray,
    scores: np.ndarray,
    block_ids: np.ndarray,
    config: dict,
    seed: int,
) -> dict:
    gate = config["gate_decision"]
    release = calibrate_threshold(
        labels,
        scores,
        block_ids,
        "release",
        gate["false_acceptance_target"],
        gate["confidence_level"],
        gate["cluster_bootstrap_replicates"],
        gate["minimum_release_coverage"],
        gate["minimum_release_blocks"],
        seed,
    )
    withhold = calibrate_threshold(
        labels,
        scores,
        block_ids,
        "withhold",
        gate["false_rejection_target"],
        gate["confidence_level"],
        gate["cluster_bootstrap_replicates"],
        0.02,
        10,
        seed + 10000,
    )
    return {"release": release, "withhold": withhold}


def apply_decisions(scores: np.ndarray, thresholds: dict) -> np.ndarray:
    decisions = np.full(len(scores), "review", dtype=object)
    release_threshold = thresholds["release"]["threshold"]
    withhold_threshold = thresholds["withhold"]["threshold"]
    decisions[scores <= withhold_threshold] = "withhold"
    decisions[scores >= release_threshold] = "release"
    if withhold_threshold >= release_threshold:
        overlap = (scores <= withhold_threshold) & (scores >= release_threshold)
        decisions[overlap] = "review"
    return decisions


def summarize_decisions(frame: pd.DataFrame, confidence: float, replicates: int, seed: int) -> dict:
    release = frame["decision"].eq("release").to_numpy()
    withhold = frame["decision"].eq("withhold").to_numpy()
    labels = frame["conforms"].astype(bool).to_numpy()
    release_stats = risk_statistics(
        labels == 0,
        release,
        frame["block_id"].to_numpy(),
        confidence,
        replicates,
        seed,
    )
    withhold_stats = risk_statistics(
        labels == 1,
        withhold,
        frame["block_id"].to_numpy(),
        confidence,
        replicates,
        seed + 1,
    )
    return {
        "rows": int(len(frame)),
        "blocks": int(frame["block_id"].nunique()),
        "release_coverage": float(release.mean()),
        "review_coverage": float(frame["decision"].eq("review").mean()),
        "withhold_coverage": float(withhold.mean()),
        "release": release_stats,
        "withhold": withhold_stats,
    }


def heuristic_scores(
    method: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    train_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if method == "support_only":
        return (
            -train["support_mean_z_l2"].to_numpy(dtype=float),
            -test["support_mean_z_l2"].to_numpy(dtype=float),
            {},
        )
    if method == "audit_only":
        train_score = -train["audit_abs_error_mean"].to_numpy(dtype=float)
        test_score = -test["audit_abs_error_mean"].to_numpy(dtype=float)
        return (
            np.nan_to_num(train_score, nan=-np.inf),
            np.nan_to_num(test_score, nan=-np.inf),
            {},
        )
    if method == "variant_prior":
        keys = ["model", "reference_budget_days", "adaptation"]
        table = train[keys].copy()
        table["conforms"] = train_labels
        rates = table.groupby(keys, dropna=False)["conforms"].mean()
        overall = float(np.mean(train_labels))
        train_index = pd.MultiIndex.from_frame(train[keys])
        test_index = pd.MultiIndex.from_frame(test[keys])
        return (
            rates.reindex(train_index).fillna(overall).to_numpy(dtype=float),
            rates.reindex(test_index).fillna(overall).to_numpy(dtype=float),
            {"overall": overall, "rates": {"|".join(map(str, key)): float(value) for key, value in rates.items()}},
        )
    raise ValueError(f"Unknown heuristic method: {method}")


def merge_labels(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    merged = features.merge(
        labels[KEY_COLUMNS + ["conforms"]],
        on=KEY_COLUMNS,
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(labels):
        raise ValueError("Feature-label merge did not preserve every label row")
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "study_config.json")
    args = parser.parse_args()
    config = load_json(args.config)
    seed = int(config["random_seed"])

    feature_path = PROJECT_ROOT / "data" / "processed" / "gate_features.csv.gz"
    development_label_path = (
        PROJECT_ROOT / "data" / "private_labels" / "gate_labels_development.csv.gz"
    )
    external_label_path = (
        PROJECT_ROOT / "data" / "private_labels" / "gate_labels_external_locked.csv.gz"
    )
    features = pd.read_csv(feature_path)
    development_features = features[
        features["protocol"].eq("primary") & features["role"].eq("development")
    ].copy()
    external_features = features[
        features["protocol"].eq("primary") & features["role"].eq("external_test")
    ].copy()
    development_labels = pd.read_csv(development_label_path)
    development_labels = development_labels[development_labels["protocol"].eq("primary")].copy()
    development = merge_labels(development_features, development_labels)
    feature_columns = [column for column in development_features.columns if column not in NON_PREDICTOR_COLUMNS]
    forbidden = {"conforms", "slope", "intercept", "r2", "rmse", "nrmse_percent"}
    if forbidden.intersection(feature_columns):
        raise ValueError("Reference-derived target columns entered the predictor schema")

    labels = development["conforms"].astype(bool).to_numpy()
    groups = (development["site"].astype(str) + "|" + development["device"].astype(str)).to_numpy()
    block_ids = development["block_id"].to_numpy()
    outer = GroupKFold(n_splits=5)
    methods = LEARNED_METHODS + ["support_only", "audit_only", "variant_prior"]
    oof_parts: list[pd.DataFrame] = []
    fold_records: list[dict] = []

    for method_index, method in enumerate(methods):
        for fold, (train_index, test_index) in enumerate(outer.split(development, labels, groups)):
            train = development.iloc[train_index]
            test = development.iloc[test_index]
            train_labels = labels[train_index]
            if method in LEARNED_METHODS:
                train_scores = grouped_oof_scores(
                    method,
                    train,
                    train_labels,
                    groups[train_index],
                    feature_columns,
                    seed + method_index * 1000 + fold * 100,
                )
                estimator = build_estimator(method, feature_columns, seed + method_index * 1000 + fold)
                estimator.fit(train[feature_columns], train_labels)
                test_scores = estimator.predict_proba(test[feature_columns])[:, 1]
            else:
                train_scores, test_scores, _ = heuristic_scores(method, train, test, train_labels)
            thresholds = calibrate_decisions(
                train_labels,
                train_scores,
                train["block_id"].to_numpy(),
                config,
                seed + method_index * 100000 + fold * 10000,
            )
            decisions = apply_decisions(test_scores, thresholds)
            part = test[KEY_COLUMNS + ["conforms"]].copy()
            part["method"] = method
            part["outer_fold"] = fold
            part["score"] = test_scores
            part["decision"] = decisions
            oof_parts.append(part)
            fold_records.append(
                {
                    "method": method,
                    "outer_fold": fold,
                    "train_groups": int(np.unique(groups[train_index]).size),
                    "test_groups": int(np.unique(groups[test_index]).size),
                    "release": thresholds["release"],
                    "withhold": thresholds["withhold"],
                }
            )

    oof = pd.concat(oof_parts, ignore_index=True)
    gate = config["gate_decision"]
    internal_summary = {
        method: summarize_decisions(
            frame,
            gate["confidence_level"],
            gate["cluster_bootstrap_replicates"],
            seed + index * 100,
        )
        for index, (method, frame) in enumerate(oof.groupby("method", sort=False))
    }
    no_gate = development[KEY_COLUMNS + ["conforms"]].copy()
    no_gate["decision"] = "release"
    internal_summary["no_gate"] = summarize_decisions(
        no_gate,
        gate["confidence_level"],
        gate["cluster_bootstrap_replicates"],
        seed + 999,
    )

    frozen_methods: dict[str, dict] = {}
    model_dir = PROJECT_ROOT / "results" / "frozen_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    for method_index, method in enumerate(methods):
        if method in LEARNED_METHODS:
            development_scores = grouped_oof_scores(
                method,
                development,
                labels,
                groups,
                feature_columns,
                seed + 500000 + method_index * 1000,
                max_splits=5,
            )
            estimator = build_estimator(method, feature_columns, seed + 600000 + method_index)
            estimator.fit(development[feature_columns], labels)
            external_scores = estimator.predict_proba(external_features[feature_columns])[:, 1]
            model_path = model_dir / f"{method}.joblib"
            joblib.dump(estimator, model_path)
            state = {"model_path": str(model_path.relative_to(PROJECT_ROOT)), "model_sha256": sha256(model_path)}
        else:
            development_scores, external_scores, state = heuristic_scores(
                method,
                development,
                external_features,
                labels,
            )
        thresholds = calibrate_decisions(
            labels,
            development_scores,
            block_ids,
            config,
            seed + 700000 + method_index * 10000,
        )
        frozen_methods[method] = {
            "thresholds": thresholds,
            "state": state,
            "external_scores": external_scores.tolist(),
        }

    freeze_path = PROJECT_ROOT / "results" / "gate_freeze.json"
    freeze_report = {
        "status": "FROZEN_BEFORE_EXTERNAL_LABEL_ACCESS",
        "feature_artifact_sha256": sha256(feature_path),
        "development_label_artifact_sha256": sha256(development_label_path),
        "feature_columns": feature_columns,
        "grouping": "site|device",
        "outer_folds": 5,
        "inner_folds_max": 4,
        "risk_bound": "max(block-cluster bootstrap percentile, Wilson upper with Kish effective blocks)",
        "methods": {
            method: {key: value for key, value in values.items() if key != "external_scores"}
            for method, values in frozen_methods.items()
        },
    }
    with freeze_path.open("w", encoding="utf-8") as handle:
        json.dump(freeze_report, handle, indent=2, ensure_ascii=True, allow_nan=True)

    # The locked labels are first opened only after the complete freeze artifact exists.
    external_labels = pd.read_csv(external_label_path)
    external_labels = external_labels[external_labels["protocol"].eq("primary")].copy()
    external = merge_labels(external_features, external_labels)
    external_parts: list[pd.DataFrame] = []
    external_summary: dict[str, dict] = {}
    for index, method in enumerate(methods):
        scores = np.asarray(frozen_methods[method]["external_scores"], dtype=float)
        decisions = apply_decisions(scores, frozen_methods[method]["thresholds"])
        part = external[KEY_COLUMNS + ["conforms"]].copy()
        part["method"] = method
        part["score"] = scores
        part["decision"] = decisions
        external_parts.append(part)
        external_summary[method] = summarize_decisions(
            part,
            gate["confidence_level"],
            gate["cluster_bootstrap_replicates"],
            seed + 800000 + index * 100,
        )
    external_no_gate = external[KEY_COLUMNS + ["conforms"]].copy()
    external_no_gate["decision"] = "release"
    external_summary["no_gate"] = summarize_decisions(
        external_no_gate,
        gate["confidence_level"],
        gate["cluster_bootstrap_replicates"],
        seed + 899999,
    )
    external_predictions = pd.concat(external_parts, ignore_index=True)

    result_dir = PROJECT_ROOT / "results"
    deterministic_gzip = {"method": "gzip", "mtime": 0}
    oof.to_csv(result_dir / "gate_internal_oof.csv.gz", index=False, compression=deterministic_gzip)
    external_predictions.to_csv(
        result_dir / "gate_external_predictions.csv.gz",
        index=False,
        compression=deterministic_gzip,
    )
    report = {
        "status": "COMPLETE",
        "external_labels_opened_after_freeze": True,
        "development_rows": int(len(development)),
        "development_blocks": int(development["block_id"].nunique()),
        "development_groups": int(np.unique(groups).size),
        "external_rows": int(len(external)),
        "external_blocks": int(external["block_id"].nunique()),
        "internal_oof": internal_summary,
        "external_locked": external_summary,
        "fold_thresholds": fold_records,
    }
    report_path = result_dir / "gate_prototype_results.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=True, allow_nan=True)
    compact = {
        "internal": {
            method: {
                "release_coverage": values["release_coverage"],
                "false_acceptance": values["release"]["point_risk"],
                "false_acceptance_upper": values["release"]["upper_bound"],
            }
            for method, values in internal_summary.items()
        },
        "external": {
            method: {
                "release_coverage": values["release_coverage"],
                "false_acceptance": values["release"]["point_risk"],
            }
            for method, values in external_summary.items()
        },
    }
    print(json.dumps(compact, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
