"""Harmonize the EPA U.S. 24-hour PurpleAir dataset without scoring outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_window_manifest import make_windows


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "epa_us"
    / "Data_DevelopmentUSPAcorrection_210408"
    / "Full24hrdataset.csv"
)
SENSOR_COLUMNS = ["PM25cfatm", "PM25cf1", "TempC", "RH"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def aggregate_daily(raw: pd.DataFrame, reference_rule: str, require_reference: bool = True) -> pd.DataFrame:
    rows: list[dict] = []
    for (sensor_id, date), group in raw.groupby(["ID", "Date"], sort=True):
        sensor = group[SENSOR_COLUMNS]
        for column in SENSOR_COLUMNS:
            finite = sensor[column].dropna()
            if len(finite) and float(finite.max() - finite.min()) > 1e-9:
                raise ValueError(f"Inconsistent {column} within {sensor_id}/{date.date()}")

        if reference_rule == "mean_all_regulatory":
            reference_rows = group
        elif reference_rule == "frm_priority":
            frm = group[group["FEMorFRM"].astype(str).str.upper().eq("FRM")]
            reference_rows = frm if not frm.empty else group
        else:
            raise ValueError(f"Unknown reference rule: {reference_rule}")

        regions = group["region"].dropna().astype(str).unique()
        if len(regions) != 1:
            raise ValueError(f"Ambiguous region within {sensor_id}/{date.date()}")
        rows.append(
            {
                "dataset": "EPA_US_24h",
                "site": str(sensor_id),
                "device": str(sensor_id),
                "date": date,
                # Native sub-hourly completeness is >=90%, but the exact count is unavailable.
                "valid_hours": np.nan,
                "cf1": float(group["PM25cf1"].iloc[0]),
                "atm": float(group["PM25cfatm"].iloc[0]),
                "rh": float(group["RH"].iloc[0]),
                "temperature_c": float(group["TempC"].iloc[0]),
                "reference_pm25": float(reference_rows["PM25FM"].mean()),
                "cf1_channel_gap": np.nan,
                "atm_channel_gap": np.nan,
                "region": regions[0],
                "reference_rule": reference_rule,
                "regulatory_values": int(len(group)),
                "reference_values_used": int(len(reference_rows)),
                "frm_available": bool(
                    group["FEMorFRM"].astype(str).str.upper().eq("FRM").any()
                ),
            }
        )
    frame = pd.DataFrame(rows)
    required = ["cf1", "atm", "rh", "temperature_c"]
    if require_reference:
        required.append("reference_pm25")
    return frame.dropna(subset=required).sort_values(["site", "date"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "config" / "study_config.json"
    )
    args = parser.parse_args()
    config = load_json(args.config)

    raw = pd.read_csv(args.input, low_memory=False)
    required = [
        "Date",
        "ID",
        "region",
        "FEMorFRM",
        "PM25FM",
        *SENSOR_COLUMNS,
    ]
    missing = sorted(set(required) - set(raw.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    raw = raw[required].copy()
    raw["Date"] = pd.to_datetime(raw["Date"], errors="raise").dt.normalize()
    for column in ["PM25FM", *SENSOR_COLUMNS]:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")

    primary = aggregate_daily(raw, "mean_all_regulatory")
    sensitivity = aggregate_daily(raw, "frm_priority")
    primary_keys = primary[["dataset", "site", "device", "date"]]
    sensitivity_keys = sensitivity[["dataset", "site", "device", "date"]]
    if not primary_keys.equals(sensitivity_keys):
        raise ValueError("Reference aggregation rules produced different daily keys")

    output_dir = PROJECT_ROOT / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    compression = {"method": "gzip", "mtime": 0}
    primary_path = output_dir / "epa_us_daily_primary.csv.gz"
    sensitivity_path = output_dir / "epa_us_daily_frm_priority.csv.gz"
    primary.to_csv(primary_path, index=False, compression=compression)
    sensitivity.to_csv(sensitivity_path, index=False, compression=compression)

    manifest_input = primary[
        ["dataset", "site", "device", "date"]
    ].copy()
    manifest = make_windows(
        manifest_input,
        config,
        protocol="epa_us_primary",
        stride=int(config["window_protocols"]["primary_stride_days"]),
        greedy_nonoverlap=True,
    )
    manifest["role"] = "post_review_external"
    manifest_path = output_dir / "epa_us_extension_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    report = {
        "status": "PREPARED_WITHOUT_OUTCOME_SCORING",
        "source_sha256": sha256(args.input),
        "reference_rules": {
            "primary": "mean_all_regulatory",
            "sensitivity": "frm_priority",
        },
        "daily_rows": int(len(primary)),
        "collocation_ids": int(primary["site"].nunique()),
        "eligible_ids": int(manifest["site"].nunique()),
        "primary_blocks": int(len(manifest)),
        "artifacts": {
            "primary_daily": {
                "path": str(primary_path.relative_to(PROJECT_ROOT)),
                "sha256": sha256(primary_path),
            },
            "frm_priority_daily": {
                "path": str(sensitivity_path.relative_to(PROJECT_ROOT)),
                "sha256": sha256(sensitivity_path),
            },
            "manifest": {
                "path": str(manifest_path.relative_to(PROJECT_ROOT)),
                "sha256": sha256(manifest_path),
            },
        },
        "outcomes_scored": False,
    }
    report_path = PROJECT_ROOT / "results" / "epa_us_extension_preparation.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=True)
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
