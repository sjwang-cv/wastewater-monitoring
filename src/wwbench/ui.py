"""Shared layout, labels, and charts for the Streamlit app."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from wwbench.build import feature_dictionary, load_catalog, load_profiles, native_frame
from wwbench.paths import PROCESSED, RESULTS
from wwbench.spec import INDICATORS, UCI_LABELS

SUPPORT_ZH = {"full": "Full", "partial": "Partial", "none": "Not supported"}
COVERAGE_ZH = {"plant": "Plant measured", "lab": "Laboratory", "simulated": "Simulated", "absent": "Absent"}
CATEGORY_ZH = {
    "A": "A Process monitoring",
    "B": "B Microscopy",
    "C": "C Industrial sensors",
    "D": "D Treatment performance",
}
MODEL_ZH = {
    "zscore": "Z-score",
    "isolation_forest": "Isolation Forest",
    "random_forest": "Random Forest",
}


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        .ww-kicker { color:#0c6b63; font-weight:700; letter-spacing:.04em; font-size:.8rem; margin-bottom:.15rem; }
        .ww-lead { color:#4d6360; font-size:1.02rem; max-width:46rem; line-height:1.55; margin-bottom:.6rem; }
        div[data-testid="stMetric"] {
            background:#fff; border:1px solid #d7e3df; border-left:4px solid #0c6b63;
            border-radius:10px; padding:.55rem .8rem;
        }
        div[data-testid="stMetricValue"] { color:#123c3a; }
        .ww-note { background:#fff; border:1px solid #d7e3df; border-radius:10px; padding:.8rem 1rem; color:#334443; }
        .ww-warn { background:#fff7ed; border:1px solid #fed7aa; border-radius:10px; padding:.8rem 1rem; color:#7c2d12; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(kicker: str, title: str, lead: str) -> None:
    apply_theme()
    st.markdown(f'<div class="ww-kicker">{kicker}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="ww-lead">{lead}</div>', unsafe_allow_html=True)


def note(text: str, warn: bool = False) -> None:
    kind = "ww-warn" if warn else "ww-note"
    st.markdown(f'<div class="{kind}">{text}</div>', unsafe_allow_html=True)


def style_fig(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Segoe UI, Microsoft YaHei, sans-serif", "color": "#1c2b2b", "size": 13},
        margin={"l": 48, "r": 20, "t": 48, "b": 40},
        legend={"orientation": "h", "y": 1.14},
        colorway=["#0c6b63", "#b45309", "#1d4e89", "#6b4c7a", "#3f7d4e", "#9f1239"],
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e4eeeb", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#e4eeeb", zeroline=False)
    return fig


@st.cache_data(show_spinner=False)
def catalog() -> list[dict]:
    return load_catalog()


@st.cache_data(show_spinner=False)
def profiles() -> dict:
    return load_profiles()


@st.cache_data(show_spinner=False)
def frame_for(dataset_id: str, hourly: bool = False) -> pd.DataFrame:
    return native_frame(dataset_id, hourly=hourly)


@st.cache_data(show_spinner=False)
def labels_for(dataset_id: str) -> dict[str, str]:
    if dataset_id == "uci_water_treatment":
        base = dict(UCI_LABELS)
    else:
        base = {}
    table = feature_dictionary(dataset_id)
    for row in table.itertuples(index=False):
        base.setdefault(row.feature_name, row.description)
    return base


def nice(dataset_id: str, column: str) -> str:
    return labels_for(dataset_id).get(column, column)


def monitor_datasets() -> list[dict]:
    return [item for item in catalog() if item.get("monitor")]


def dataset_map() -> dict[str, dict]:
    return {item["dataset_id"]: item for item in catalog()}


@st.cache_data(show_spinner=False)
def screen_table(dataset_id: str) -> pd.DataFrame:
    frame = pd.read_csv(RESULTS / f"screen_{dataset_id}.csv")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame


@st.cache_data(show_spinner=False)
def reference_for(dataset_id: str) -> dict:
    return json.loads((RESULTS / f"reference_{dataset_id}.json").read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def synthetic_results() -> dict:
    return json.loads((RESULTS / "synthetic_eval.json").read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def scenario_results() -> dict:
    return json.loads((RESULTS / "bsm1_scenario_classification.json").read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def run_summary() -> dict:
    return json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))


def performance_table() -> pd.DataFrame:
    return pd.read_csv(PROCESSED / "performance_records.csv")


def coverage_figure() -> go.Figure:
    items = catalog()
    z = []
    hover = []
    for item in items:
        row = []
        tips = []
        for key, _label in INDICATORS:
            state = item["indicators"].get(key, "absent")
            row.append({"plant": 3, "lab": 2, "simulated": 1, "absent": 0}[state])
            tips.append(f"{item['name']}<br>{_label}: {COVERAGE_ZH[state]}")
        z.append(row)
        hover.append(tips)
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=[label for _, label in INDICATORS],
            y=[item["name"] for item in items],
            text=hover,
            hoverinfo="text",
            colorscale=[
                [0.0, "#edf2f0"],
                [0.16, "#edf2f0"],
                [0.17, "#b45309"],
                [0.49, "#b45309"],
                [0.50, "#1d4e89"],
                [0.82, "#1d4e89"],
                [0.83, "#0c6b63"],
                [1.0, "#0c6b63"],
            ],
            zmin=0,
            zmax=3,
            showscale=False,
            xgap=3,
            ygap=3,
        )
    )
    return style_fig(fig, height=420)


def support_figure() -> go.Figure:
    items = catalog()
    tasks = items[0]["tasks"]
    code = {"none": 0, "partial": 1, "full": 2}
    z = [[code[task["support"]] for task in item["tasks"]] for item in items]
    hover = [
        [f"{item['name']}<br>{task['name']}: {SUPPORT_ZH[task['support']]}<br>{task['rationale']}" for task in item["tasks"]]
        for item in items
    ]
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=[task["name"] for task in tasks],
            y=[item["name"] for item in items],
            text=hover,
            hoverinfo="text",
            colorscale=[[0, "#edf2f0"], [0.49, "#edf2f0"], [0.5, "#b45309"], [0.99, "#b45309"], [1, "#0c6b63"]],
            zmin=0,
            zmax=2,
            showscale=False,
            xgap=3,
            ygap=3,
        )
    )
    return style_fig(fig, height=460)


def histogram(profile: dict, column: str, title: str) -> go.Figure:
    entry = profile["columns"][column]
    edges = entry["hist_edges"]
    counts = entry["hist_counts"]
    centers = [(edges[i] + edges[i + 1]) / 2 for i in range(len(counts))]
    widths = [edges[i + 1] - edges[i] for i in range(len(counts))]
    fig = go.Figure(go.Bar(x=centers, y=counts, width=widths, marker_color="#0c6b63", name=title))
    fig.update_layout(title=title, xaxis_title=title, yaxis_title="Samples", bargap=0.02)
    return style_fig(fig, 380)


def correlation_figure(profile: dict, rename: dict[str, str]) -> go.Figure | None:
    corr = profile.get("correlation") or {}
    columns = corr.get("columns") or []
    if len(columns) < 2:
        return None
    labels = [rename.get(column, column) for column in columns]
    matrix = corr["matrix"]
    fig = go.Figure(
        go.Heatmap(
            z=matrix,
            x=labels,
            y=labels,
            zmin=-1,
            zmax=1,
            colorscale="Tealgrn",
            hovertemplate="%{y} vs %{x}<br>correlation %{z:.2f}<extra></extra>",
        )
    )
    return style_fig(fig, 480)


def line_chart(
    frame: pd.DataFrame,
    x: str,
    columns: list[str],
    rename: dict[str, str],
    flags: pd.Series | None = None,
    title: str = "",
) -> go.Figure:
    fig = go.Figure()
    for column in columns:
        fig.add_trace(
            go.Scatter(
                x=frame[x],
                y=frame[column],
                mode="lines",
                name=rename.get(column, column),
                line={"width": 1.6},
            )
        )
    if flags is not None and flags.any():
        marked = frame.loc[flags]
        y_column = columns[0]
        fig.add_trace(
            go.Scatter(
                x=marked[x],
                y=marked[y_column],
                mode="markers",
                name="Flagged points",
                marker={"color": "#9f1239", "size": 8, "symbol": "diamond"},
            )
        )
    fig.update_layout(title=title, xaxis_title="", yaxis_title="")
    return style_fig(fig, 440)


def choose_monitor_dataset() -> str:
    options = monitor_datasets()
    labels = {item["dataset_id"]: item["name"] for item in options}
    current = st.session_state.get("active_dataset", "uci_water_treatment")
    ids = list(labels)
    if current not in ids:
        current = ids[0]
    selected = st.selectbox("Dataset", ids, index=ids.index(current), format_func=lambda key: labels[key])
    st.session_state["active_dataset"] = selected
    return selected


def modeling_view(dataset_id: str) -> pd.DataFrame:
    spec = dataset_map()[dataset_id]
    return frame_for(dataset_id, hourly=spec.get("model_resolution") == "1h")


def window_frame(frame: pd.DataFrame, dataset_id: str) -> pd.DataFrame:
    spec = dataset_map()[dataset_id]
    if spec.get("time_axis") == "simulation_day" and "weather" in frame.columns:
        weather = st.selectbox("Influent scenario", ["dry", "rain", "storm"], format_func=lambda value: {"dry": "Dry", "rain": "Rain", "storm": "Storm"}[value])
        part = frame[frame["weather"] == weather].copy()
        return part
    if "timestamp" not in frame.columns or frame["timestamp"].isna().all():
        return frame
    stamp = pd.to_datetime(frame["timestamp"])
    start_default = stamp.min().date()
    end_default = stamp.max().date()
    if (stamp.max() - stamp.min()).days > 28:
        end_default = (stamp.min() + pd.Timedelta(days=28)).date()
    show_all = st.checkbox("Show the full time range", value=False)
    if show_all:
        return frame
    picked = st.date_input("Time range", value=(start_default, end_default))
    if not isinstance(picked, tuple) or len(picked) != 2:
        return frame
    start, end = picked
    mask = (stamp.dt.date >= start) & (stamp.dt.date <= end)
    return frame.loc[mask].copy()


def score_column(method: str) -> tuple[str, str]:
    if method == "zscore":
        return "score_z", "flag_z"
    return "score_if", "flag_if"


def top_anomalies(table: pd.DataFrame, method: str, limit: int = 15) -> pd.DataFrame:
    score, flag = score_column(method)
    part = table.loc[table[flag].astype(bool)].sort_values(score, ascending=False).head(limit)
    return part


def report_markdown(dataset_name: str, method: str, table: pd.DataFrame, features: list[str]) -> str:
    score, flag = score_column(method)
    flagged = table.loc[table[flag].astype(bool)].sort_values(score, ascending=False)
    lines = [
        "# Wastewater monitoring screening report",
        "",
        f"- Dataset: {dataset_name}",
        f"- Model: {MODEL_ZH.get(method, method)}",
        f"- Samples: {len(table)}",
        f"- Flagged outliers: {len(flagged)}",
        f"- Variables: {', '.join(features)}",
        "",
        "These points are statistical outliers relative to the reference period. They are not confirmed equipment faults or effluent violations.",
        "",
        "## Largest deviations",
        "",
    ]
    for row in flagged.head(10).itertuples(index=False):
        lines.append(f"- {getattr(row, 'timestamp')}, score {getattr(row, score):.3f}, top variable {getattr(row, 'top_variable_z', '')}")
    return "\n".join(lines)
