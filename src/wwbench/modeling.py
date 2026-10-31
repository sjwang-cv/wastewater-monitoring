"""Lightweight screening and a separately reported synthetic-anomaly test."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)


def model_features(frame: pd.DataFrame, features: list[dict]) -> list[str]:
    names = []
    for feature in features:
        if feature.get("use_in_model") and feature["feature_name"] in frame.columns:
            series = frame[feature["feature_name"]]
            if pd.api.types.is_numeric_dtype(series) and series.nunique(dropna=True) > 1:
                names.append(feature["feature_name"])
    return names


def _time_key(frame: pd.DataFrame, mode: str, day_column: str | None) -> pd.Series:
    if mode == "none":
        return pd.Series(0, index=frame.index)
    if mode == "time_of_day":
        if day_column and day_column in frame.columns:
            hour = np.floor((frame[day_column] % 1) * 24).astype(int) % 24
            return hour
        return frame["timestamp"].dt.hour.astype(int)
    if mode == "hour_of_week":
        stamp = frame["timestamp"]
        return (stamp.dt.dayofweek * 24 + stamp.dt.hour).astype(int)
    raise ValueError(mode)


def reference_positions(frame: pd.DataFrame, config: dict) -> tuple[np.ndarray, np.ndarray]:
    if config.get("reference_column"):
        pool = np.flatnonzero(frame[config["reference_column"]].to_numpy() == config["reference_value"])
    else:
        pool = np.arange(len(frame))
    cut = int(len(pool) * config.get("train_frac", 0.7))
    cut = min(max(cut, 1), len(pool) - 1)
    return pool[:cut], pool[cut:]


def fit_reference(
    frame: pd.DataFrame,
    feature_cols: list[str],
    ref_pos: np.ndarray,
    seasonal_mode: str,
    day_column: str | None,
) -> dict[str, Any]:
    keys = _time_key(frame, seasonal_mode, day_column)
    ref = frame.iloc[ref_pos]
    ref_keys = keys.iloc[ref_pos]
    baselines: dict[str, dict[str, float]] = {}
    global_median: dict[str, float] = {}
    sigma: dict[str, float] = {}
    kept = []
    for column in feature_cols:
        observed = ref[column]
        median = float(observed.median()) if observed.notna().any() else 0.0
        global_median[column] = median
        grouped = observed.groupby(ref_keys).median()
        baselines[column] = {str(int(key)): float(value) for key, value in grouped.dropna().items()}
        base = ref_keys.map(lambda key, table=baselines[column], fallback=median: table.get(str(int(key)), fallback))
        residual = observed - base
        scale = float(residual.std(skipna=True))
        if not np.isfinite(scale) or scale < 1e-8:
            continue
        sigma[column] = scale
        kept.append(column)
    if not kept:
        raise ValueError("No variables are available for modeling.")
    return {
        "feature_cols": kept,
        "seasonal_mode": seasonal_mode,
        "day_column": day_column,
        "baselines": {column: baselines[column] for column in kept},
        "global_median": {column: global_median[column] for column in kept},
        "sigma": sigma,
        "ref_rows": int(len(ref_pos)),
    }


def apply_reference(frame: pd.DataFrame, reference: dict[str, Any]) -> pd.DataFrame:
    keys = _time_key(frame, reference["seasonal_mode"], reference.get("day_column"))
    columns = reference["feature_cols"]
    z = pd.DataFrame(index=frame.index)
    for column in columns:
        table = reference["baselines"][column]
        fallback = reference["global_median"][column]
        base = keys.map(lambda key, table=table, fallback=fallback: table.get(str(int(key)), fallback)).astype(float)
        actual = frame[column].astype(float)
        filled = actual.fillna(base)
        residual = filled - base
        # Missing measurements contribute no deviation.
        residual = residual.where(actual.notna(), 0.0)
        z[column] = residual / reference["sigma"][column]
    return z


def zscore_flags(z: pd.DataFrame, threshold: float) -> tuple[pd.Series, pd.Series]:
    score = z.abs().max(axis=1)
    return score, score >= threshold


def fit_isolation_forest(z_train: pd.DataFrame, contamination: float, n_estimators: int, seed: int) -> IsolationForest:
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=min(max(contamination, 0.001), 0.5),
        random_state=seed,
        n_jobs=1,
    )
    model.fit(z_train.to_numpy())
    return model


def isolation_scores(model: IsolationForest, z: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    raw = model.decision_function(z.to_numpy())
    score = pd.Series(-raw, index=z.index)
    flag = pd.Series(model.predict(z.to_numpy()) == -1, index=z.index)
    return score, flag


def isolation_contributions(model: IsolationForest, z_row: pd.Series) -> pd.Series:
    order = list(z_row.index)
    base = float(-model.decision_function(z_row.to_numpy().reshape(1, -1))[0])
    values = {}
    for column in order:
        altered = z_row.copy()
        altered[column] = 0.0
        score = float(-model.decision_function(altered.to_numpy().reshape(1, -1))[0])
        values[column] = base - score
    return pd.Series(values).sort_values(ascending=False)


def inject_synthetic(
    frame: pd.DataFrame,
    columns: list[str],
    positions: np.ndarray,
    sigma: dict[str, float],
    seed: int,
    event_fraction: float = 0.06,
) -> tuple[pd.DataFrame, np.ndarray, list[dict]]:
    rng = np.random.default_rng(seed)
    out = frame.copy()
    for column in columns:
        out[column] = pd.to_numeric(out[column], errors="coerce").astype(float)
    labels = np.zeros(len(frame), dtype=int)
    pool = list(map(int, positions))
    rng.shuffle(pool)
    occupied: set[int] = set()
    events = []
    target = max(4, int(round(event_fraction * len(positions))))
    ph_cols = {column for column in columns if "ph" in column.lower()}
    for start in pool:
        if len(events) >= target:
            break
        kind = "spike" if len(events) % 2 == 0 else "drift"
        span = 1 if kind == "spike" else 4
        slots = list(range(start, start + span))
        if slots[-1] not in set(pool) or any(slot in occupied for slot in slots):
            continue
        usable = [column for column in columns if column in sigma and sigma[column] > 0]
        if not usable:
            break
        n_cols = 1 if len(usable) == 1 else int(rng.integers(1, min(3, len(usable) + 1)))
        chosen = list(rng.choice(usable, size=n_cols, replace=False))
        for column in chosen:
            direction = float(rng.choice([-1.0, 1.0])) if column in ph_cols else 1.0
            magnitude = float(rng.uniform(3.5, 6.0) if kind == "spike" else 2.5)
            shift = direction * magnitude * sigma[column]
            for slot in slots:
                current = out.at[slot, column]
                if pd.isna(current):
                    continue
                value = float(current) + shift
                if column in ph_cols:
                    value = min(max(value, 0.0), 14.0)
                elif direction > 0:
                    value = max(value, 0.0)
                out.at[slot, column] = value
        for slot in slots:
            labels[slot] = 1
            occupied.add(slot)
        events.append({"start": start, "end": slots[-1], "kind": kind, "columns": chosen})
    return out, labels, events


def classification_metrics(y_true: np.ndarray, flags: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, flags, average="binary", zero_division=0
    )
    matrix = confusion_matrix(y_true, flags, labels=[0, 1]).tolist()
    auc = None
    curve = None
    if len(np.unique(y_true)) == 2 and np.isfinite(scores).all():
        try:
            auc = float(roc_auc_score(y_true, scores))
            fpr, tpr, _ = roc_curve(y_true, scores)
            step = max(1, len(fpr) // 80)
            curve = {"fpr": [float(v) for v in fpr[::step]], "tpr": [float(v) for v in tpr[::step]]}
        except ValueError:
            auc = None
            curve = None
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": auc,
        "support_positive": int(y_true.sum()),
        "support_negative": int((y_true == 0).sum()),
        "confusion_matrix": matrix,
        "roc": curve,
    }


def event_latency(events: list[dict], flags: np.ndarray, step_hours: float) -> dict[str, Any]:
    detected = []
    missed = 0
    for event in events:
        window = flags[event["start"] : event["end"] + 1]
        hits = np.flatnonzero(window)
        if len(hits) == 0:
            missed += 1
            continue
        steps = int(hits[0])
        detected.append(steps)
    mean_steps = float(np.mean(detected)) if detected else None
    return {
        "events": len(events),
        "detected_events": len(detected),
        "missed_events": missed,
        "mean_latency_steps": mean_steps,
        "mean_latency_hours": None if mean_steps is None else mean_steps * step_hours,
    }


def screen(
    frame: pd.DataFrame,
    feature_cols: list[str],
    config: dict,
    threshold: float = 3.0,
    contamination: float = 0.05,
    n_estimators: int = 200,
    seed: int = 42,
) -> dict[str, Any]:
    ordered = frame.sort_values("timestamp").reset_index(drop=True)
    ref_pos, test_pos = reference_positions(ordered, config)
    reference = fit_reference(
        ordered,
        feature_cols,
        ref_pos,
        config.get("seasonal_mode", "none"),
        config.get("day_column"),
    )
    z = apply_reference(ordered, reference)
    z_score, z_flag = zscore_flags(z, threshold)
    model = fit_isolation_forest(z.iloc[ref_pos], contamination, n_estimators, seed)
    if_score, if_flag = isolation_scores(model, z)
    table = pd.DataFrame(
        {
            "sample_id": ordered["sample_id"] if "sample_id" in ordered.columns else np.arange(len(ordered)),
            "timestamp": ordered["timestamp"],
            "score_z": z_score,
            "flag_z": z_flag.astype(bool),
            "score_if": if_score,
            "flag_if": if_flag.astype(bool),
            "top_variable_z": z.abs().idxmax(axis=1),
        }
    )
    reference.update(
        {
            "threshold": threshold,
            "contamination": contamination,
            "n_estimators": n_estimators,
            "ref_end_timestamp": str(ordered.iloc[ref_pos[-1]]["timestamp"]),
            "test_start_timestamp": str(ordered.iloc[test_pos[0]]["timestamp"]),
        }
    )
    return {
        "frame": ordered,
        "z": z,
        "table": table,
        "reference": reference,
        "model": model,
        "ref_pos": ref_pos,
        "test_pos": test_pos,
    }


def evaluate_synthetic(
    frame: pd.DataFrame,
    feature_cols: list[str],
    config: dict,
    threshold: float = 3.0,
    contamination: float = 0.05,
    n_estimators: int = 200,
    seed: int = 42,
    step_hours: float = 24.0,
) -> dict[str, Any]:
    ordered = frame.sort_values("timestamp").reset_index(drop=True)
    ref_pos, test_pos = reference_positions(ordered, config)
    clean_ref = fit_reference(
        ordered,
        feature_cols,
        ref_pos,
        config.get("seasonal_mode", "none"),
        config.get("day_column"),
    )
    columns = clean_ref["feature_cols"]
    injected, labels, events = inject_synthetic(
        ordered, columns, test_pos, clean_ref["sigma"], seed=seed
    )
    z = apply_reference(injected, clean_ref)
    z_score, z_flag = zscore_flags(z, threshold)
    if_model = fit_isolation_forest(apply_reference(ordered, clean_ref).iloc[ref_pos], contamination, n_estimators, seed)
    if_score, if_flag = isolation_scores(if_model, z)

    train_injected, train_labels, _ = inject_synthetic(
        ordered, columns, ref_pos, clean_ref["sigma"], seed=seed + 17
    )
    z_train = apply_reference(train_injected, clean_ref).iloc[ref_pos]
    rf_score_test = np.zeros(len(test_pos))
    rf_flag_test = np.zeros(len(test_pos), dtype=bool)
    importance = {}
    rf_threshold = None
    rf_train_f1 = None
    train_y = train_labels[ref_pos]
    if len(np.unique(train_y)) > 1:
        forest = RandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=1,
        )
        forest.fit(z_train.to_numpy(), train_y)
        if 1 in list(forest.classes_):
            positive = list(forest.classes_).index(1)
            train_proba = forest.predict_proba(z_train.to_numpy())[:, positive]
            best_threshold = 0.5
            best_train_f1 = -1.0
            for candidate in np.linspace(0.05, 0.95, 19):
                _, _, train_f1, _ = precision_recall_fscore_support(
                    train_y,
                    train_proba >= candidate,
                    average="binary",
                    zero_division=0,
                )
                if train_f1 > best_train_f1:
                    best_train_f1 = float(train_f1)
                    best_threshold = float(candidate)
            rf_score_test = forest.predict_proba(z.iloc[test_pos].to_numpy())[:, positive]
            rf_flag_test = rf_score_test >= best_threshold
            importance = {
                column: float(value)
                for column, value in zip(columns, forest.feature_importances_)
            }
            rf_threshold = best_threshold
            rf_train_f1 = best_train_f1

    y_test = labels[test_pos]
    metrics = {
        "zscore": classification_metrics(y_test, z_flag.iloc[test_pos].to_numpy(), z_score.iloc[test_pos].to_numpy()),
        "isolation_forest": classification_metrics(y_test, if_flag.iloc[test_pos].to_numpy(), if_score.iloc[test_pos].to_numpy()),
        "random_forest": classification_metrics(y_test, rf_flag_test, rf_score_test),
    }
    metrics["random_forest"]["decision_threshold"] = rf_threshold
    metrics["random_forest"]["train_f1"] = rf_train_f1
    latency = {
        "zscore": event_latency(events, z_flag.to_numpy(), step_hours),
        "isolation_forest": event_latency(events, if_flag.to_numpy(), step_hours),
    }
    rf_full = np.zeros(len(ordered), dtype=bool)
    rf_full[test_pos] = rf_flag_test
    latency["random_forest"] = event_latency(events, rf_full, step_hours)
    return {
        "label": "synthetic",
        "protocol": (
            "Synthetic anomalies are injected only in the test segment after the reference period. "
            "Point anomalies are 3.5–6 reference standard deviations; short drifts are 4 consecutive points at 2.5 standard deviations. "
            "The Z-score threshold and Isolation Forest contamination are preset and are not chosen on the test labels. "
            "Random Forest is trained on a separate set of training synthetic labels, and its decision threshold is chosen by F1 on that training set only."
        ),
        "metrics": metrics,
        "latency": latency,
        "events": len(events),
        "threshold": threshold,
        "contamination": contamination,
        "feature_cols": columns,
        "importance": importance,
        "test_scores": {
            "y": y_test.astype(int).tolist(),
            "zscore": z_score.iloc[test_pos].astype(float).tolist(),
            "isolation_forest": if_score.iloc[test_pos].astype(float).tolist(),
        },
    }


def sweep_thresholds(scores: np.ndarray, y_true: np.ndarray, thresholds: list[float]) -> list[dict]:
    rows = []
    for threshold in thresholds:
        flags = scores >= threshold
        metrics = classification_metrics(y_true, flags, scores)
        metrics["threshold"] = threshold
        rows.append(metrics)
    return rows
