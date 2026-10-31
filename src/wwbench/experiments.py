"""Run screening, the synthetic test, and the BSM1 scenario classifier."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from wwbench.build import feature_dictionary, load_catalog, native_frame
from wwbench.modeling import (
    apply_reference,
    evaluate_synthetic,
    fit_isolation_forest,
    isolation_scores,
    model_features,
    screen,
    sweep_thresholds,
)
from wwbench.paths import RESULTS, ensure_dirs
from wwbench.spec import dataset_by_id


def monitoring_config(dataset_id: str) -> dict:
    spec = dataset_by_id(dataset_id)
    config = {
        "seasonal_mode": spec["seasonal_mode"],
        "train_frac": spec["train_frac"],
    }
    if spec.get("reference_column"):
        config["reference_column"] = spec["reference_column"]
        config["reference_value"] = spec["reference_value"]
    if dataset_id == "bsm1_influent":
        config["day_column"] = "time_day"
    return config


def _step_hours(dataset_id: str) -> float:
    return {"uci_water_treatment": 24.0, "bsm1_influent": 0.25, "dhi_wwtp_inflow": 1.0, "koeln_weiden": 1.0}[dataset_id]


def _modeling_frame(dataset_id: str) -> pd.DataFrame:
    spec = dataset_by_id(dataset_id)
    hourly = spec.get("model_resolution") == "1h"
    return native_frame(dataset_id, hourly=hourly)


def run_screening() -> dict:
    ensure_dirs()
    summaries = {}
    for spec in load_catalog():
        if not spec.get("monitor"):
            continue
        dataset_id = spec["dataset_id"]
        frame = _modeling_frame(dataset_id)
        features = feature_dictionary(dataset_id).to_dict(orient="records")
        columns = model_features(frame, features)
        result = screen(frame, columns, monitoring_config(dataset_id))
        table = result["table"].copy()
        for column in ("weather", "author_cluster_kind", "author_cluster_name"):
            if column in result["frame"].columns:
                table[column] = result["frame"][column].to_numpy()
        table.to_csv(RESULTS / f"screen_{dataset_id}.csv", index=False)
        reference = {key: value for key, value in result["reference"].items()}
        (RESULTS / f"reference_{dataset_id}.json").write_text(
            json.dumps(reference, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        joblib.dump(result["model"], RESULTS / "models" / f"if_{dataset_id}.joblib")
        summary = {
            "rows": int(len(table)),
            "features": result["reference"]["feature_cols"],
            "flagged_z": int(table["flag_z"].sum()),
            "flagged_if": int(table["flag_if"].sum()),
            "seasonal_mode": result["reference"]["seasonal_mode"],
        }
        if "author_cluster_kind" in table.columns:
            summary["cluster_overlap"] = _cluster_overlap(table)
        summaries[dataset_id] = summary
    return summaries


def _cluster_overlap(table: pd.DataFrame) -> dict:
    overlap = {}
    for method, column in (("zscore", "flag_z"), ("isolation_forest", "flag_if")):
        flags = table[column].astype(bool)
        kind = table["author_cluster_kind"].fillna("unlabeled")
        overlap[method] = {}
        for name in ("routine", "nonroutine", "unlabeled"):
            mask = kind == name
            overlap[method][name] = {
                "days": int(mask.sum()),
                "flagged": int((mask & flags).sum()),
            }
    overlap["note"] = "The authors' conceptual clusters are not a fault gold standard. This only reports how many days the screening flags overlap those clusters."
    return overlap


def run_synthetic() -> dict:
    output = {}
    for dataset_id in ("uci_water_treatment", "bsm1_influent"):
        frame = _modeling_frame(dataset_id)
        features = feature_dictionary(dataset_id).to_dict(orient="records")
        columns = model_features(frame, features)
        result = evaluate_synthetic(
            frame,
            columns,
            monitoring_config(dataset_id),
            step_hours=_step_hours(dataset_id),
        )
        scores = result.pop("test_scores")
        y_true = np.asarray(scores["y"])
        z_scores = np.asarray(scores["zscore"], dtype=float)
        result["sensitivity_zscore"] = sweep_thresholds(z_scores, y_true, [2.0, 2.5, 3.0, 3.5, 4.0])
        sensitivity_if = []
        for contamination in (0.02, 0.05, 0.1):
            rerun = evaluate_synthetic(
                frame,
                columns,
                monitoring_config(dataset_id),
                contamination=contamination,
                step_hours=_step_hours(dataset_id),
            )
            metrics = rerun["metrics"]["isolation_forest"]
            metrics["contamination"] = contamination
            sensitivity_if.append(metrics)
        result["sensitivity_isolation_forest"] = sensitivity_if
        output[dataset_id] = result
    (RESULTS / "synthetic_eval.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def run_bsm1_classifier() -> dict:
    frame = native_frame("bsm1_influent").sort_values(["weather", "time_day"]).reset_index(drop=True)
    features = feature_dictionary("bsm1_influent")
    columns = model_features(frame, features.to_dict(orient="records"))
    config = monitoring_config("bsm1_influent")
    dry = frame["weather"] == "dry"
    ref_pos = np.flatnonzero(dry.to_numpy())
    ref_pos = ref_pos[: int(len(ref_pos) * 0.7)]
    from wwbench.modeling import fit_reference

    reference = fit_reference(frame, columns, ref_pos, config["seasonal_mode"], config.get("day_column"))
    z = apply_reference(frame, reference)
    day = np.floor(frame["time_day"]).astype(int)
    rng = np.random.default_rng(42)
    test_days = {}
    for weather, part in frame.groupby("weather"):
        days = np.array(sorted(day[part.index].unique()))
        holdout = max(1, int(round(len(days) * 0.3)))
        test_days[weather] = set(rng.choice(days, size=holdout, replace=False).tolist())
    test_mask = np.array([int(day_value) in test_days[weather] for day_value, weather in zip(day, frame["weather"])])
    train_mask = ~test_mask
    model = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=1,
    )
    model.fit(z.loc[train_mask, reference["feature_cols"]], frame.loc[train_mask, "weather"])
    predicted = model.predict(z.loc[test_mask, reference["feature_cols"]])
    actual = frame.loc[test_mask, "weather"].to_numpy()
    labels = ["dry", "rain", "storm"]
    result = {
        "task": "Classify the influent-file scenario. The label is the file identity, not instantaneous weather and not an in-plant fault.",
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, average="macro", labels=labels)),
        "confusion_matrix": confusion_matrix(actual, predicted, labels=labels).tolist(),
        "labels": labels,
        "train_rows": int(train_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "features": reference["feature_cols"],
    }
    (RESULTS / "bsm1_scenario_classification.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def run_experiments() -> dict:
    screening = run_screening()
    synthetic = run_synthetic()
    scenario = run_bsm1_classifier()
    uci = synthetic["uci_water_treatment"]["metrics"]
    summary = {
        "screening": screening,
        "synthetic_f1": {name: values["f1"] for name, values in uci.items()},
        "primary_model": "isolation_forest",
        "primary_f1_synthetic": uci["isolation_forest"]["f1"],
        "bsm1_scenario_macro_f1": scenario["macro_f1"],
        "label_policy": "Every displayed F1 comes from a synthetic anomaly test or a simulated scenario classification. None of them is real wastewater-plant fault detection performance.",
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def load_summary() -> dict:
    return json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
