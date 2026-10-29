"""Build harmonized PurpleAir PM2.5 tables from the frozen raw datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def project_path(relative: str) -> Path:
    return PROJECT_ROOT / relative


def build_quant_hourly(config: dict, reference_join: str = "inner") -> pd.DataFrame:
    sensor_path = project_path(config["raw_data"]["quant_sensors"])
    reference_path = project_path(config["raw_data"]["quant_reference"])
    excluded_flags = set(config["quality_control"]["exclude_quant_flags"])

    selected_chunks: list[pd.DataFrame] = []
    usecols = [
        "time",
        "location",
        "instrument",
        "sensornumber",
        "version",
        "measurand",
        "measurement",
        "flag",
    ]
    for chunk in pd.read_csv(sensor_path, usecols=usecols, chunksize=500_000):
        purpleair = chunk["instrument"].astype(str).str.fullmatch(r"PA\d+")
        wanted_measurement = (
            (chunk["measurand"].eq("PM2.5") & chunk["version"].isin(["indoor", "atmospheric"]))
            | (
                chunk["measurand"].isin(["RelHumidity", "Temperature"])
                & chunk["version"].eq("out-of-box")
                & chunk["sensornumber"].eq(1)
            )
        )
        valid_flag = chunk["flag"].isna() | ~chunk["flag"].isin(excluded_flags)
        selected_chunks.append(chunk.loc[purpleair & wanted_measurement & valid_flag])

    long = pd.concat(selected_chunks, ignore_index=True)
    long["time"] = pd.to_datetime(long["time"], errors="raise")
    long["feature"] = np.select(
        [
            long["measurand"].eq("PM2.5") & long["version"].eq("indoor") & long["sensornumber"].eq(1),
            long["measurand"].eq("PM2.5") & long["version"].eq("indoor") & long["sensornumber"].eq(2),
            long["measurand"].eq("PM2.5") & long["version"].eq("atmospheric") & long["sensornumber"].eq(1),
            long["measurand"].eq("PM2.5") & long["version"].eq("atmospheric") & long["sensornumber"].eq(2),
            long["measurand"].eq("RelHumidity"),
            long["measurand"].eq("Temperature"),
        ],
        ["cf1_a", "cf1_b", "atm_a", "atm_b", "rh", "temperature_c"],
        default="",
    )
    long = long[long["feature"].ne("")]
    wide = long.pivot_table(
        index=["time", "location", "instrument"],
        columns="feature",
        values="measurement",
        aggfunc="mean",
    ).reset_index()
    wide.columns.name = None

    for prefix in ["cf1", "atm"]:
        channels = [f"{prefix}_a", f"{prefix}_b"]
        for column in channels:
            if column not in wide:
                wide[column] = np.nan
        wide[prefix] = wide[channels].mean(axis=1, skipna=True)
        wide[f"{prefix}_channel_gap"] = (wide[channels[0]] - wide[channels[1]]).abs()
        wide[f"{prefix}_n_channels"] = wide[channels].notna().sum(axis=1)

    reference = pd.read_csv(reference_path, parse_dates=["time"])
    reference = reference[
        reference["measurand"].eq("PM2.5") & reference["version"].eq("Ratified")
    ][["time", "location", "measurement"]].rename(columns={"measurement": "reference_pm25"})
    if reference.duplicated(["time", "location"]).any():
        raise ValueError("Ratified QUANT PM2.5 reference keys are not unique")

    if reference_join not in {"inner", "left"}:
        raise ValueError("reference_join must be inner (R0 reproduction) or left (sensor-only revision)")
    merged = wide.merge(reference, on=["time", "location"], how=reference_join, validate="many_to_one")
    merged = merged.rename(columns={"location": "site", "instrument": "device"})
    merged.insert(0, "dataset", "QUANT")
    return merged.sort_values(["site", "device", "time"]).reset_index(drop=True)


def build_india_hourly(config: dict) -> pd.DataFrame:
    workbook = project_path(config["raw_data"]["india_workbook"])
    frames: list[pd.DataFrame] = []
    for sheet in pd.ExcelFile(workbook).sheet_names:
        raw = pd.read_excel(workbook, sheet_name=sheet)
        temperature_column = next(column for column in raw if "Temperature" in str(column))
        frame = pd.DataFrame(
            {
                "dataset": "CampmierIndia",
                "site": "Bengaluru" if sheet == "Bangalore" else sheet,
                "device": f"PA_aggregate_{sheet}",
                "time": pd.to_datetime(raw["Index"], errors="raise"),
                "cf1": pd.to_numeric(raw["PA_CF1 (ug/m3)"], errors="coerce"),
                "atm": pd.to_numeric(raw["PA_ATM (ug/m3)"], errors="coerce"),
                "rh": pd.to_numeric(raw["PA_RH (%)"], errors="coerce"),
                "temperature_c": pd.to_numeric(raw[temperature_column], errors="coerce"),
                "reference_pm25": pd.to_numeric(raw["BAM (ug/m3)"], errors="coerce"),
            }
        )
        frame["cf1_a"] = np.nan
        frame["cf1_b"] = np.nan
        frame["atm_a"] = np.nan
        frame["atm_b"] = np.nan
        frame["cf1_channel_gap"] = np.nan
        frame["atm_channel_gap"] = np.nan
        frame["cf1_n_channels"] = np.nan
        frame["atm_n_channels"] = np.nan
        frames.append(frame)
    return pd.concat(frames, ignore_index=True).sort_values(["site", "time"]).reset_index(drop=True)


def build_daily(hourly: pd.DataFrame, min_hours: int) -> pd.DataFrame:
    required = ["cf1", "atm", "rh", "temperature_c", "reference_pm25"]
    clean = hourly.dropna(subset=required).copy()
    clean["date"] = clean["time"].dt.floor("D")
    daily = (
        clean.groupby(["dataset", "site", "device", "date"], as_index=False)
        .agg(
            valid_hours=("time", "nunique"),
            cf1=("cf1", "mean"),
            atm=("atm", "mean"),
            rh=("rh", "mean"),
            temperature_c=("temperature_c", "mean"),
            reference_pm25=("reference_pm25", "mean"),
            cf1_channel_gap=("cf1_channel_gap", "mean"),
            atm_channel_gap=("atm_channel_gap", "mean"),
        )
    )
    return daily[daily["valid_hours"].ge(min_hours)].reset_index(drop=True)


def make_audit(config: dict, quant: pd.DataFrame, india: pd.DataFrame, daily: pd.DataFrame) -> dict:
    raw_files = {
        name: project_path(path)
        for name, path in config["raw_data"].items()
    }

    def summarize_hourly(frame: pd.DataFrame) -> dict:
        return {
            "rows": int(len(frame)),
            "sites": sorted(frame["site"].unique().tolist()),
            "devices": int(frame["device"].nunique()),
            "time_min": frame["time"].min().isoformat(),
            "time_max": frame["time"].max().isoformat(),
            "complete_common_feature_rows": int(
                frame[["cf1", "atm", "rh", "temperature_c", "reference_pm25"]]
                .notna()
                .all(axis=1)
                .sum()
            ),
        }

    daily_groups = (
        daily.groupby(["dataset", "site", "device"], as_index=False)
        .agg(valid_days=("date", "nunique"), start=("date", "min"), end=("date", "max"))
    )
    daily_groups["start"] = daily_groups["start"].astype(str)
    daily_groups["end"] = daily_groups["end"].astype(str)
    return {
        "raw_files": {
            name: {"path": str(path.relative_to(PROJECT_ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for name, path in raw_files.items()
        },
        "quant_hourly": summarize_hourly(quant),
        "india_hourly": summarize_hourly(india),
        "valid_daily_rows": int(len(daily)),
        "daily_groups": daily_groups.to_dict(orient="records"),
        "limitations": [
            "QUANT duplicate-reference data contain NO, NO2, and O3 but no PM2.5.",
            "The India workbook contains site-level aggregate PurpleAir channels, so unit-level precision is unavailable.",
            "EPA targets are used as operational evaluation criteria; this retrospective analysis is not EPA certification.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "study_config.json")
    args = parser.parse_args()
    config = load_config(args.config)

    output_paths = {key: project_path(path) for key, path in config["derived_data"].items()}
    for path in output_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    quant = build_quant_hourly(config)
    india = build_india_hourly(config)
    common_columns = sorted(set(quant.columns) | set(india.columns))
    quant = quant.reindex(columns=common_columns)
    india = india.reindex(columns=common_columns)
    daily = build_daily(
        pd.concat([quant, india], ignore_index=True),
        config["quality_control"]["valid_hours_per_day"],
    )
    audit = make_audit(config, quant, india, daily)

    deterministic_gzip = {"method": "gzip", "mtime": 0}
    quant.to_csv(output_paths["quant_hourly"], index=False, compression=deterministic_gzip)
    india.to_csv(output_paths["india_hourly"], index=False, compression=deterministic_gzip)
    daily.to_csv(output_paths["daily"], index=False, compression=deterministic_gzip)
    with output_paths["audit"].open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, ensure_ascii=True)

    print(json.dumps(audit, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
