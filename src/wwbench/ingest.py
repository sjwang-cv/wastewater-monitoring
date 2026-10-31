"""Load raw public files into native, standardized, and performance tables."""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from wwbench.paths import ROOT
from wwbench.spec import (
    MDAS_CATEGORIES,
    UCI_DOCUMENTED_MISSING,
    UCI_FEATURES,
    UCI_KIND,
    UCI_LABELS,
)

DATE_RE = re.compile(r"D-(\d{1,2})/(\d{1,2})/(\d{2})")
RANGE_RE = re.compile(
    r"D-(\d{1,2})/(\d{1,2})/(\d{2})\s+to\s+D-(\d{1,2})/(\d{1,2})/(\d{2})"
)
STD_COLUMNS = [
    "dataset_id", "sample_id", "timestamp", "pH", "DO", "COD", "BOD",
    "NH4_N", "TN", "TP", "MLSS", "SS", "conductivity", "flow_rate",
    "temperature", "operating_status",
]


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "latin1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin1")


def _parse_uci_date(token: str) -> datetime:
    match = DATE_RE.fullmatch(token.strip())
    if not match:
        raise ValueError(f"Could not parse UCI date: {token}")
    day, month, year = (int(part) for part in match.groups())
    year += 1900 if year >= 50 else 2000
    return datetime(year, month, day)


def _class_dates(body: str) -> set[datetime]:
    found: set[datetime] = set()
    for match in RANGE_RE.finditer(body):
        start = _parse_uci_date(f"D-{match.group(1)}/{match.group(2)}/{match.group(3)}")
        end = _parse_uci_date(f"D-{match.group(4)}/{match.group(5)}/{match.group(6)}")
        if end < start:
            start, end = end, start
        cursor = start
        while cursor <= end:
            found.add(cursor)
            cursor += timedelta(days=1)
    masked = RANGE_RE.sub(" ", body)
    for match in DATE_RE.finditer(masked):
        found.add(_parse_uci_date(match.group(0)))
    return found


def load_uci_clusters(names_path: Path) -> dict[datetime, tuple[int, str, str]]:
    text = _read_text(names_path)
    section = text.split("9. Class Distribution", 1)[1].split("-- Comments", 1)[0]
    parts = re.split(r"--\s*Class\s+(\d+)\s*:\s*", section)
    mapping: dict[datetime, tuple[int, str, str]] = {}
    conflicts: list[str] = []
    for index in range(1, len(parts), 2):
        class_id = int(parts[index])
        body = parts[index + 1]
        name = body.strip().splitlines()[0].strip()
        kind = UCI_KIND[class_id]
        for day in _class_dates(body):
            if day in mapping and mapping[day][0] != class_id:
                conflicts.append(f"{day.date()} appears in both {mapping[day][0]} and {class_id}")
                continue
            mapping[day] = (class_id, name, kind)
    load_uci_clusters.conflicts = conflicts  # type: ignore[attr-defined]
    return mapping


def _empty_standard(n: int) -> pd.DataFrame:
    frame = pd.DataFrame({column: pd.Series([pd.NA] * n) for column in STD_COLUMNS})
    return frame


def _feature(name: str, unit: str, role: str, description: str, model: bool) -> dict:
    return {
        "feature_name": name,
        "canonical_name": name,
        "unit": unit,
        "role": role,
        "description": description,
        "use_in_model": model,
    }


def load_uci(data_path: Path, names_path: Path) -> dict:
    frame = pd.read_csv(
        data_path,
        header=None,
        names=["date_raw", *UCI_FEATURES],
        na_values=["?"],
        encoding="latin1",
    )
    frame["timestamp"] = frame["date_raw"].map(_parse_uci_date)
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    clusters = load_uci_clusters(names_path)
    assigned = frame["timestamp"].map(lambda value: clusters.get(value))
    frame["author_cluster_id"] = assigned.map(lambda item: item[0] if item else pd.NA)
    frame["author_cluster_name"] = assigned.map(lambda item: item[1] if item else pd.NA)
    frame["author_cluster_kind"] = assigned.map(lambda item: item[2] if item else "unlabeled")
    frame.insert(0, "dataset_id", "uci_water_treatment")
    frame.insert(1, "sample_id", frame["timestamp"].dt.strftime("uci-%Y%m%d"))

    standard = _empty_standard(len(frame))
    standard["dataset_id"] = "uci_water_treatment"
    standard["sample_id"] = frame["sample_id"]
    standard["timestamp"] = frame["timestamp"]
    standard["pH"] = frame["PH-E"]
    standard["COD"] = frame["DQO-E"]
    standard["BOD"] = frame["DBO-E"]
    standard["SS"] = frame["SS-E"]
    standard["conductivity"] = frame["COND-E"]
    standard["flow_rate"] = frame["Q-E"]
    standard["operating_status"] = pd.NA

    performance_rows = []
    pairings = [
        ("BOD", "DBO-E", "DBO-S", "RD-DBO-G", "mg/L"),
        ("COD", "DQO-E", "DQO-S", "RD-DQO-G", "mg/L"),
        ("SS", "SS-E", "SS-S", "RD-SS-G", "mg/L"),
        ("SED", "SED-E", "SED-S", "RD-SED-G", "not stated"),
    ]
    for pollutant, influent, effluent, removal, unit in pairings:
        for _, row in frame.iterrows():
            performance_rows.append(
                {
                    "sample_id": f"{row['sample_id']}-{pollutant}",
                    "dataset_id": "uci_water_treatment",
                    "treatment_method": "Municipal plant-wide treatment (global performance recorded by UCI)",
                    "pollutant": pollutant,
                    "input_parameters": "",
                    "influent_quality": json.dumps({influent: None if pd.isna(row[influent]) else row[influent]}, ensure_ascii=False),
                    "effluent_quality": json.dumps({effluent: None if pd.isna(row[effluent]) else row[effluent]}, ensure_ascii=False),
                    "influent_value": row[influent],
                    "effluent_value": row[effluent],
                    "concentration_unit": unit,
                    "removal_efficiency": row[removal],
                    "removal_unit": "%",
                }
            )
    performance = pd.DataFrame(performance_rows)
    observed = frame[UCI_FEATURES].isna().sum().to_dict()
    missing_check = []
    for name, documented in UCI_DOCUMENTED_MISSING.items():
        missing_check.append(
            {
                "feature": name,
                "documented": documented,
                "observed": int(observed[name]),
                "match": int(observed[name]) == int(documented),
            }
        )

    unit_for = {}
    for name in UCI_FEATURES:
        if name.startswith("PH"):
            unit_for[name] = "pH"
        elif name.startswith("RD-"):
            unit_for[name] = "%"
        elif name.startswith("Q-"):
            unit_for[name] = "m³/d (not stated in .names; assigned from the flow magnitude)"
        elif name.startswith("COND"):
            unit_for[name] = "µS/cm (not stated in .names; assigned from the conductivity magnitude)"
        elif name.startswith("SSV"):
            unit_for[name] = "% (not stated in .names; values mostly fall in 0–100)"
        elif name.startswith("SED") or "SED" in name:
            unit_for[name] = "not stated (magnitude is close to settled volume, mL/L)"
        else:
            unit_for[name] = "mg/L (not stated in .names; assigned as the usual concentration unit)"
    features = []
    for name in UCI_FEATURES:
        role = "performance" if name.startswith("RD-") else "measured"
        features.append(
            _feature(name, unit_for[name], role, UCI_LABELS[name], role == "measured")
        )
    labeled = int(frame["author_cluster_id"].notna().sum())
    return {
        "native": frame,
        "standardized": standard,
        "performance": performance,
        "features": features,
        "n_samples": int(len(frame)),
        "extra": {
            "cluster_labeled_days": labeled,
            "cluster_unlabeled_days": int(len(frame) - labeled),
            "cluster_conflicts": list(getattr(load_uci_clusters, "conflicts", [])),
            "missing_check": missing_check,
            "cluster_counts": frame["author_cluster_kind"].value_counts().to_dict(),
        },
    }


def load_bsm1(raw_dir: Path) -> dict:
    files = {
        "dry": raw_dir / "Inf_dry_2006.txt",
        "rain": raw_dir / "Inf_rain_2006.txt",
        "storm": raw_dir / "Inf_strm_2006.txt",
    }
    frames = []
    for weather, path in files.items():
        part = pd.read_csv(path, sep=r"\s+")
        part["weather"] = weather
        part["source_file"] = path.name
        frames.append(part)
    frame = pd.concat(frames, ignore_index=True)
    frame.insert(0, "dataset_id", "bsm1_influent")
    pieces = []
    for weather, part in frame.groupby("weather", sort=False):
        part = part.copy()
        part["sample_id"] = [f"bsm1-{weather}-{i:04d}" for i in range(len(part))]
        pieces.append(part)
    frame = pd.concat(pieces, ignore_index=True)
    cod = frame[["Si", "Ss", "Xi", "Xs", "Xbh", "Xba", "Xp"]].sum(axis=1)
    frame["COD_asm1"] = cod
    frame["NH4_N"] = frame["Snh"]
    anchor = pd.Timestamp("2006-01-01")
    frame["timestamp"] = anchor + pd.to_timedelta(frame["t"], unit="D")
    frame["time_day"] = frame["t"]

    standard = _empty_standard(len(frame))
    standard["dataset_id"] = "bsm1_influent"
    standard["sample_id"] = frame["sample_id"]
    standard["timestamp"] = frame["timestamp"]
    standard["COD"] = frame["COD_asm1"]
    standard["NH4_N"] = frame["NH4_N"]
    standard["flow_rate"] = frame["Q"]
    standard["operating_status"] = "influent-scenario:" + frame["weather"]

    descriptions = {
        "t": ("d", "simulated", "Simulation time", False),
        "Si": ("g COD/m³", "simulated", "Soluble inert organic matter", True),
        "Ss": ("g COD/m³", "simulated", "Readily biodegradable substrate", True),
        "Xi": ("g COD/m³", "simulated", "Particulate inert organic matter", True),
        "Xs": ("g COD/m³", "simulated", "Slowly biodegradable substrate", True),
        "Xbh": ("g COD/m³", "simulated", "Heterotrophic biomass", True),
        "Xba": ("g COD/m³", "simulated", "Autotrophic biomass, defined as 0 in the influent", False),
        "Xp": ("g COD/m³", "simulated", "Particulate products, defined as 0 in the influent", False),
        "So": ("g/m³", "simulated", "Dissolved oxygen, defined as 0 in the influent and not counted as DO coverage", False),
        "Sno": ("g N/m³", "simulated", "Nitrate nitrogen, defined as 0 in the influent", False),
        "Snh": ("g N/m³", "simulated", "Ammonia nitrogen", True),
        "Snd": ("g N/m³", "simulated", "Soluble biodegradable organic nitrogen", True),
        "Xnd": ("g N/m³", "simulated", "Particulate biodegradable organic nitrogen", True),
        "Salk": ("mol/m³", "simulated", "Alkalinity", True),
        "Q": ("m³/d", "simulated", "Influent flow", True),
        "COD_asm1": ("g COD/m³", "derived", "Sum of ASM1 components, not a laboratory COD", False),
        "weather": ("—", "label", "Influent-file scenario", False),
    }
    features = [
        _feature(name, unit, role, text, model)
        for name, (unit, role, text, model) in descriptions.items()
    ]
    return {
        "native": frame,
        "standardized": standard,
        "performance": None,
        "features": features,
        "n_samples": int(len(frame)),
        "extra": {"rows_by_weather": frame["weather"].value_counts().to_dict()},
    }


def load_dhi(path: Path) -> dict:
    frame = pd.read_csv(path)
    frame["timestamp"] = pd.to_datetime(frame["time"])
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    frame.insert(0, "dataset_id", "dhi_wwtp_inflow")
    frame.insert(1, "sample_id", frame["timestamp"].dt.strftime("dhi-%Y%m%d%H"))
    standard = _empty_standard(len(frame))
    standard["dataset_id"] = "dhi_wwtp_inflow"
    standard["sample_id"] = frame["sample_id"]
    standard["timestamp"] = frame["timestamp"]
    standard["flow_rate"] = frame["flow"]
    descriptions = {
        "flow": ("m³/h", "measured", "WWTP inflow", True),
        "acc_precip": ("mm", "meteorological", "Accumulated precipitation; not used by the anomaly model by default", False),
        "mean_pressure": ("hPa", "meteorological", "Air pressure", False),
        "mean_radiation": ("W/m²", "meteorological", "Radiation", False),
        "mean_relative_hum": ("%", "meteorological", "Relative humidity", False),
        "mean_temp": ("°C", "meteorological", "Air temperature, not wastewater temperature", False),
        "temp_grass": ("°C", "meteorological", "Grass temperature", False),
        "temp_soil_10": ("°C", "meteorological", "Soil temperature at 10 cm", False),
        "temp_soil_30": ("°C", "meteorological", "Soil temperature at 30 cm", False),
    }
    features = [_feature(name, *meta) for name, meta in descriptions.items()]
    return {
        "native": frame,
        "standardized": standard,
        "performance": None,
        "features": features,
        "n_samples": int(len(frame)),
        "extra": {},
    }


def load_koeln(path: Path) -> dict:
    frame = pd.read_csv(path, sep=";")
    frame = frame.rename(
        columns={
            "Timestamp": "timestamp",
            "rain_rate [mm/min]": "rain_rate_mm_min",
            "Level_1 [mNHN]": "level_1_m",
            "Level_2 [mNHN]": "level_2_m",
            "Level_3 [mNHN]": "level_3_m",
            "level_storage_sewer [mNHN]": "level_storage_m",
            "wtp_inflow [l/s]": "inflow_L_s",
        }
    )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    frame.insert(0, "dataset_id", "koeln_weiden")
    frame.insert(1, "sample_id", frame["timestamp"].dt.strftime("koeln-%Y%m%d%H%M"))
    hourly = (
        frame.set_index("timestamp")
        .resample("1h")
        .mean(numeric_only=True)
        .dropna(how="all")
        .reset_index()
    )
    hourly.insert(0, "dataset_id", "koeln_weiden")
    hourly.insert(1, "sample_id", hourly["timestamp"].dt.strftime("koeln-h-%Y%m%d%H"))
    standard = _empty_standard(len(frame))
    standard["dataset_id"] = "koeln_weiden"
    standard["sample_id"] = frame["sample_id"]
    standard["timestamp"] = frame["timestamp"]
    standard["flow_rate"] = frame["inflow_L_s"]
    descriptions = {
        "inflow_L_s": ("L/s", "measured", "WWTP inflow", True),
        "level_1_m": ("mNHN", "measured", "Sewer level 1", True),
        "level_2_m": ("mNHN", "measured", "Sewer level 2", True),
        "level_3_m": ("mNHN", "measured", "Sewer level 3", True),
        "level_storage_m": ("mNHN", "measured", "Storage level", True),
        "rain_rate_mm_min": ("mm/min", "meteorological", "Rain rate; not used by the anomaly model by default", False),
    }
    features = [_feature(name, *meta) for name, meta in descriptions.items()]
    return {
        "native": frame,
        "hourly": hourly,
        "standardized": standard,
        "performance": None,
        "features": features,
        "n_samples": int(len(frame)),
        "extra": {"hourly_rows": int(len(hourly))},
    }


def _recovery_frame(path: Path, pollutant: str, method: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame = frame.drop(columns=[column for column in frame.columns if str(column).lower().startswith("reference")])
    renamed = {}
    for column in frame.columns:
        text = str(column)
        lowered = text.lower().replace(" ", "")
        if "℃" in text or "°c" in lowered or lowered.startswith("temperature"):
            renamed[column] = "temperature_C"
        elif lowered.startswith("stir"):
            renamed[column] = "stir_rpm"
        elif lowered.startswith("t(min") or lowered == "time_min":
            renamed[column] = "time_min"
        elif lowered == "ph":
            renamed[column] = "pH"
        elif lowered.startswith("n:p"):
            renamed[column] = "n_to_p"
        elif lowered.startswith("mg"):
            renamed[column] = "mg_ratio"
        elif lowered.startswith("p_initial"):
            renamed[column] = "p_initial_mg_l"
        elif "recovery" in lowered:
            renamed[column] = "recovery_pct"
    frame = frame.rename(columns=renamed)
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(how="all").reset_index(drop=True)
    recovery_column = "recovery_pct"
    rows = []
    for index, row in frame.iterrows():
        parameters = {column: None if pd.isna(row[column]) else row[column] for column in frame.columns if column != recovery_column}
        initial_column = "p_initial_mg_l" if pollutant == "P" and "p_initial_mg_l" in frame.columns else None
        influent = None if initial_column is None else row[initial_column]
        rows.append(
            {
                "sample_id": f"aqua-{pollutant.lower()}-{index:04d}",
                "dataset_id": "aquafetch_nutrient_recovery",
                "treatment_method": method,
                "pollutant": pollutant,
                "input_parameters": json.dumps(parameters, ensure_ascii=False),
                "influent_quality": json.dumps({} if influent is None or pd.isna(influent) else {initial_column: influent}, ensure_ascii=False),
                "effluent_quality": "",
                "influent_value": influent,
                "effluent_value": np.nan,
                "concentration_unit": "mg/L" if initial_column else "",
                "removal_efficiency": row[recovery_column],
                "removal_unit": "%",
            }
        )
    numeric = frame.copy()
    numeric.insert(0, "dataset_id", "aquafetch_nutrient_recovery")
    numeric.insert(1, "pollutant", pollutant)
    numeric.insert(2, "sample_id", [f"aqua-{pollutant.lower()}-{i:04d}" for i in range(len(numeric))])
    return numeric, pd.DataFrame(rows)


def load_aquafetch(raw_dir: Path) -> dict:
    nitrogen, nitrogen_perf = _recovery_frame(
        raw_dir / "N_recovery.csv",
        "N",
        "Laboratory adsorption recovery of nitrogen",
    )
    phosphorus, phosphorus_perf = _recovery_frame(
        raw_dir / "P_recovery.csv",
        "P",
        "Laboratory adsorption recovery of phosphorus",
    )
    native = pd.concat([nitrogen, phosphorus], ignore_index=True, sort=False)
    performance = pd.concat([nitrogen_perf, phosphorus_perf], ignore_index=True)
    features = [
        _feature("pH", "pH", "measured", "Experimental pH", False),
        _feature("temperature_C", "°C", "measured", "Experimental temperature, not plant water temperature", False),
        _feature("p_initial_mg_l", "mg/L", "measured", "Initial phosphorus concentration", False),
        _feature("recovery_pct", "%", "performance", "Nitrogen or phosphorus recovery", False),
        _feature("stir_rpm", "rpm", "measured", "Stirring speed", False),
        _feature("time_min", "min", "measured", "Reaction time", False),
        _feature("n_to_p", "—", "measured", "Nitrogen to phosphorus ratio", False),
        _feature("mg_ratio", "—", "measured", "Magnesium ratio to nitrogen or phosphorus", False),
    ]
    return {
        "native": native,
        "standardized": None,
        "performance": performance,
        "features": features,
        "n_samples": int(len(performance)),
        "extra": {"n_recovery": int(len(nitrogen_perf)), "p_recovery": int(len(phosphorus_perf))},
    }


def load_mdas_index() -> pd.DataFrame:
    frame = pd.DataFrame(MDAS_CATEGORIES)
    frame.insert(0, "dataset_id", "mdas_2025")
    frame["image_id"] = ""
    frame["image_path"] = ""
    frame["bounding_boxes"] = "Annotated set is PASCAL VOC XML; images were not downloaded into this repository"
    frame["segmentation_mask"] = ""
    frame["magnification"] = "10× objective"
    return frame


def resolve(relative: str) -> Path:
    return ROOT / relative
