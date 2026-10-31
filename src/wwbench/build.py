"""Build the benchmark tables, profiles, catalog, and SQLite index."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from wwbench.ingest import load_aquafetch, load_bsm1, load_dhi, load_koeln, load_mdas_index, load_uci
from wwbench.paths import DATA, METADATA, PROCESSED, RAW, ROOT, ensure_dirs
from wwbench.spec import DATASETS, INDICATORS, TASKS, TASK_SUPPORT, coverage_state


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _profile(frame: pd.DataFrame, columns: list[str]) -> dict:
    profile = {"n_rows": int(len(frame)), "columns": {}}
    usable = []
    for column in columns:
        if column not in frame.columns or not pd.api.types.is_numeric_dtype(frame[column]):
            continue
        series = frame[column].astype(float)
        missing = float(series.isna().mean()) if len(series) else 0.0
        observed = series.dropna()
        entry = {"missing_rate": missing, "n_observed": int(observed.shape[0])}
        if len(observed):
            entry.update(
                {
                    "min": float(observed.min()),
                    "p25": float(observed.quantile(0.25)),
                    "median": float(observed.median()),
                    "p75": float(observed.quantile(0.75)),
                    "max": float(observed.max()),
                    "mean": float(observed.mean()),
                    "std": float(observed.std(ddof=1)) if len(observed) > 1 else 0.0,
                }
            )
            counts, edges = np.histogram(observed, bins=20)
            entry["hist_counts"] = counts.astype(int).tolist()
            entry["hist_edges"] = [float(edge) for edge in edges]
            if missing < 0.6 and observed.nunique() > 1:
                usable.append(column)
        profile["columns"][column] = entry
    corr_columns = usable[:10]
    if len(corr_columns) >= 2:
        matrix = frame[corr_columns].corr(numeric_only=True)
        profile["correlation"] = {
            "columns": corr_columns,
            "matrix": matrix.round(4).where(matrix.notna(), None).values.tolist(),
        }
    else:
        profile["correlation"] = {"columns": [], "matrix": []}
    if "timestamp" in frame.columns and frame["timestamp"].notna().any():
        stamp = pd.to_datetime(frame["timestamp"])
        profile["time_min"] = str(stamp.min())
        profile["time_max"] = str(stamp.max())
    return profile


def _json_matrix(matrix: list) -> list:
    cleaned = []
    for row in matrix:
        cleaned.append([None if value is None or (isinstance(value, float) and not np.isfinite(value)) else value for value in row])
    return cleaned


def build_all() -> dict:
    ensure_dirs()
    uci = load_uci(RAW / "uci" / "extracted" / "water-treatment.data", RAW / "uci" / "extracted" / "water-treatment.names")
    bsm1 = load_bsm1(RAW / "bsm1")
    dhi = load_dhi(RAW / "dhi" / "extracted" / "processed" / "data.csv")
    koeln = load_koeln(RAW / "koeln" / "wtp_network_rain.csv")
    aqua = load_aquafetch(RAW / "aquafetch")
    loaded = {
        "uci_water_treatment": uci,
        "bsm1_influent": bsm1,
        "dhi_wwtp_inflow": dhi,
        "koeln_weiden": koeln,
        "aquafetch_nutrient_recovery": aqua,
    }

    native_dir = PROCESSED / "native"
    standard_dir = PROCESSED / "standardized"
    for dataset_id, bundle in loaded.items():
        _write_frame(bundle["native"], native_dir / f"{dataset_id}.csv")
        if bundle.get("standardized") is not None:
            _write_frame(bundle["standardized"], standard_dir / f"{dataset_id}.csv")
    _write_frame(koeln["hourly"], native_dir / "koeln_weiden_hourly.csv")
    performance = pd.concat(
        [bundle["performance"] for bundle in loaded.values() if bundle.get("performance") is not None],
        ignore_index=True,
    )
    _write_frame(performance, PROCESSED / "performance_records.csv")
    _write_frame(load_mdas_index(), PROCESSED / "mdas_category_index.csv")

    profiles = {}
    for dataset_id, bundle in loaded.items():
        source = bundle["hourly"] if dataset_id == "koeln_weiden" else bundle["native"]
        columns = [feature["feature_name"] for feature in bundle["features"] if feature["feature_name"] in source.columns]
        profile = _profile(source, columns)
        if "correlation" in profile:
            profile["correlation"]["matrix"] = _json_matrix(profile["correlation"]["matrix"])
        profiles[dataset_id] = profile
    if uci["extra"].get("missing_check"):
        profiles["uci_water_treatment"]["documented_missing_check"] = uci["extra"]["missing_check"]
    (PROCESSED / "profiles.json").write_text(json.dumps(profiles, ensure_ascii=False), encoding="utf-8")

    catalog = []
    feature_rows = []
    task_rows = []
    for spec in DATASETS:
        item = dict(spec)
        bundle = loaded.get(spec["dataset_id"])
        if bundle is not None:
            item["n_samples"] = bundle["n_samples"]
            item["n_files"] = len(spec["raw_files"])
            item["extra"] = bundle["extra"]
            item["sha256"] = {relative: _sha256(ROOT / relative) for relative in spec["raw_files"]}
            profile = profiles[spec["dataset_id"]]
            item["time_min"] = profile.get("time_min")
            item["time_max"] = profile.get("time_max")
            for feature in bundle["features"]:
                feature_rows.append({"dataset_id": spec["dataset_id"], **feature})
        else:
            published = spec.get("published_counts", {})
            item["n_samples"] = None
            item["n_files"] = 0
            item["extra"] = published
            item["sha256"] = {}
        item["indicators"] = {key: coverage_state(spec["dataset_id"], key) for key, _ in INDICATORS}
        item["tasks"] = [
            {
                "task_id": task["task_id"],
                "name": task["name"],
                "support": TASK_SUPPORT[spec["dataset_id"]][task["task_id"]][0],
                "rationale": TASK_SUPPORT[spec["dataset_id"]][task["task_id"]][1],
            }
            for task in TASKS
        ]
        for task in item["tasks"]:
            task_rows.append(
                {
                    "dataset_id": spec["dataset_id"],
                    "dataset_name": spec["name"],
                    "task_id": task["task_id"],
                    "task_name": task["name"],
                    "support": task["support"],
                    "rationale": task["rationale"],
                    "bundled": spec["bundled"],
                }
            )
        catalog.append(item)

    (METADATA / "catalog.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(task_rows).to_csv(METADATA / "dataset_task_matrix.csv", index=False)
    pd.DataFrame(feature_rows).to_csv(METADATA / "feature_dictionary.csv", index=False)

    database = DATA / "benchmark.sqlite"
    if database.exists():
        database.unlink()
    connection = sqlite3.connect(database)
    pd.DataFrame(TASKS).to_sql("tasks", connection, index=False)
    catalog_table = pd.DataFrame(
        [
            {
                "dataset_id": item["dataset_id"],
                "name": item["name"],
                "category": item["category"],
                "modality": item["modality"],
                "domain": item["domain"],
                "bundled": int(item["bundled"]),
                "n_samples": item.get("n_samples"),
                "sampling_rate": item["sampling_rate"],
                "data_format": item["data_format"],
                "license": item["license"],
                "source": item["source"],
                "url": item["url"],
                "citation": item["citation"],
                "label_note": item["label_note"],
                "availability": item["availability"],
            }
            for item in catalog
        ]
    )
    catalog_table.to_sql("datasets", connection, index=False)
    pd.DataFrame(feature_rows).to_sql("features", connection, index=False)
    pd.DataFrame(task_rows).to_sql("task_support", connection, index=False)
    connection.close()

    bundled_samples = sum(item["n_samples"] for item in catalog if item["bundled"])
    summary = {
        "datasets": len(catalog),
        "bundled_datasets": sum(item["bundled"] for item in catalog),
        "bundled_samples": int(bundled_samples),
        "uci_cluster": uci["extra"],
        "counts": {item["dataset_id"]: item.get("n_samples") for item in catalog},
    }
    (METADATA / "build_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return {"catalog": catalog, "profiles": profiles, "summary": summary, "loaded_ids": list(loaded)}


def load_catalog() -> list[dict]:
    return json.loads((METADATA / "catalog.json").read_text(encoding="utf-8"))


def load_profiles() -> dict:
    return json.loads((PROCESSED / "profiles.json").read_text(encoding="utf-8"))


def native_frame(dataset_id: str, hourly: bool = False) -> pd.DataFrame:
    name = "koeln_weiden_hourly.csv" if dataset_id == "koeln_weiden" and hourly else f"{dataset_id}.csv"
    path = PROCESSED / "native" / name
    frame = pd.read_csv(path)
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame


def feature_dictionary(dataset_id: str | None = None) -> pd.DataFrame:
    frame = pd.read_csv(METADATA / "feature_dictionary.csv")
    if dataset_id:
        return frame[frame["dataset_id"] == dataset_id].reset_index(drop=True)
    return frame
