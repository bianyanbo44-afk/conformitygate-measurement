"""Create deterministic, leakage-checked deployment-window manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def make_windows(
    daily: pd.DataFrame,
    config: dict,
    protocol: str,
    stride: int,
    greedy_nonoverlap: bool = False,
) -> pd.DataFrame:
    deployment_days = int(config["quality_control"]["deployment_days"])
    min_valid_days = int(config["quality_control"]["valid_days_per_deployment"])
    lookback = int(config["quality_control"]["audit_lookback_calendar_days"])
    budgets = sorted(int(value) for value in config["target_reference_budgets_days"])
    rows: list[dict] = []

    group_columns = ["dataset", "site", "device"]
    for (dataset, site, device), group in daily.groupby(group_columns, sort=True):
        group = group.sort_values("date").copy()
        valid_dates = pd.DatetimeIndex(group["date"].unique()).sort_values()
        if valid_dates.empty:
            continue

        first_start = valid_dates.min().normalize() + pd.Timedelta(days=lookback)
        last_start = valid_dates.max().normalize() - pd.Timedelta(days=deployment_days - 1)
        if first_start > last_start:
            continue

        next_allowed_start = first_start
        candidate_frequency = "1D" if greedy_nonoverlap else f"{stride}D"
        for start in pd.date_range(first_start, last_start, freq=candidate_frequency):
            if greedy_nonoverlap and start < next_allowed_start:
                continue
            end = start + pd.Timedelta(days=deployment_days - 1)
            evaluation_dates = valid_dates[(valid_dates >= start) & (valid_dates <= end)]
            if len(evaluation_dates) < min_valid_days:
                continue

            audit_start = start - pd.Timedelta(days=lookback)
            audit_pool = valid_dates[(valid_dates >= audit_start) & (valid_dates < start)]
            record = {
                "protocol": protocol,
                "dataset": dataset,
                "role": "external_test" if dataset == "CampmierIndia" else "development",
                "site": site,
                "device": device,
                "evaluation_start": start.date().isoformat(),
                "evaluation_end": end.date().isoformat(),
                "evaluation_valid_days": int(len(evaluation_dates)),
                "audit_pool_start": audit_start.date().isoformat(),
                "audit_pool_end": (start - pd.Timedelta(days=1)).date().isoformat(),
                "audit_pool_valid_days": int(len(audit_pool)),
            }
            for budget in budgets:
                chosen = audit_pool[-budget:] if budget else audit_pool[:0]
                record[f"eligible_b{budget}"] = bool(len(chosen) == budget)
                record[f"audit_dates_b{budget}"] = ";".join(day.date().isoformat() for day in chosen)
            rows.append(record)
            if greedy_nonoverlap:
                next_allowed_start = end + pd.Timedelta(days=1)

    manifest = pd.DataFrame(rows)
    if manifest.empty:
        return manifest
    manifest.insert(
        0,
        "block_id",
        [f"{protocol}_{index:04d}" for index in range(1, len(manifest) + 1)],
    )
    return manifest


def leakage_checks(manifest: pd.DataFrame, budgets: list[int]) -> dict:
    findings: list[str] = []
    for row in manifest.itertuples(index=False):
        start = pd.Timestamp(row.evaluation_start)
        end = pd.Timestamp(row.evaluation_end)
        if end - start != pd.Timedelta(days=29):
            findings.append(f"{row.block_id}: evaluation block is not 30 calendar days")
        for budget in budgets:
            dates_text = getattr(row, f"audit_dates_b{budget}")
            dates = [pd.Timestamp(value) for value in dates_text.split(";") if value]
            if any(value >= start for value in dates):
                findings.append(f"{row.block_id}: budget {budget} audit date overlaps evaluation")
            if len(dates) != budget and getattr(row, f"eligible_b{budget}"):
                findings.append(f"{row.block_id}: budget {budget} eligibility mismatch")

    primary = manifest[manifest["protocol"].eq("primary")].copy()
    for keys, group in primary.groupby(["dataset", "site", "device"]):
        group = group.sort_values("evaluation_start")
        previous_end = None
        for row in group.itertuples(index=False):
            start = pd.Timestamp(row.evaluation_start)
            end = pd.Timestamp(row.evaluation_end)
            if previous_end is not None and start <= previous_end:
                findings.append(f"{keys}: overlapping primary windows")
            previous_end = end

    role_mismatch = manifest[
        (manifest["dataset"].eq("CampmierIndia") & ~manifest["role"].eq("external_test"))
        | (~manifest["dataset"].eq("CampmierIndia") & ~manifest["role"].eq("development"))
    ]
    if not role_mismatch.empty:
        findings.append("dataset-role lock mismatch")
    return {"status": "PASS" if not findings else "FAIL", "findings": findings}


def summarize(manifest: pd.DataFrame, budgets: list[int], checks: dict) -> dict:
    summary: dict = {"rows": int(len(manifest)), "leakage_checks": checks, "by_protocol_role": {}}
    for (protocol, role), group in manifest.groupby(["protocol", "role"]):
        key = f"{protocol}:{role}"
        summary["by_protocol_role"][key] = {
            "blocks": int(len(group)),
            "sites": sorted(group["site"].unique().tolist()),
            "eligible_by_budget": {
                str(budget): int(group[f"eligible_b{budget}"].sum()) for budget in budgets
            },
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "study_config.json")
    args = parser.parse_args()
    config = load_config(args.config)
    daily_path = PROJECT_ROOT / config["derived_data"]["daily"]
    daily = pd.read_csv(daily_path, parse_dates=["date"])

    protocols = config["window_protocols"]
    primary = make_windows(
        daily,
        config,
        "primary",
        int(protocols["primary_stride_days"]),
        greedy_nonoverlap=True,
    )
    sensitivity = make_windows(daily, config, "sensitivity", int(protocols["sensitivity_stride_days"]))
    manifest = pd.concat([primary, sensitivity], ignore_index=True)
    budgets = sorted(int(value) for value in config["target_reference_budgets_days"])
    checks = leakage_checks(manifest, budgets)
    if checks["status"] != "PASS":
        raise RuntimeError(json.dumps(checks, indent=2))
    summary = summarize(manifest, budgets, checks)

    manifest_path = PROJECT_ROOT / "data" / "processed" / "deployment_windows.csv"
    summary_path = PROJECT_ROOT / "results" / "window_manifest_audit.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
