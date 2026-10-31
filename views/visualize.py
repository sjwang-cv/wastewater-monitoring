import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from wwbench.ui import (
    catalog,
    frame_for,
    labels_for,
    line_chart,
    note,
    page_header,
    performance_table,
    style_fig,
    window_frame,
)


def page() -> None:
    items = [item for item in catalog() if item["bundled"]]
    page_header("Stage 1", "Visualization", "Choose a dataset and variables to see the actual records. Simulation clocks, experiment tables, and plant calendars are never drawn on one time axis.")
    labels = {item["dataset_id"]: item["name"] for item in items}
    dataset_id = st.selectbox("Dataset", list(labels), format_func=lambda key: labels[key])
    item = next(entry for entry in items if entry["dataset_id"] == dataset_id)
    rename = labels_for(dataset_id)

    if dataset_id == "aquafetch_nutrient_recovery":
        _aqua()
        return
    if not item["bundled"]:
        note("This dataset has no local data file.")
        return

    hourly = item.get("model_resolution") == "1h"
    frame = frame_for(dataset_id, hourly=hourly).copy()
    if hourly:
        st.caption("Cologne charts use hourly means. The 15-minute values are in data/processed/native/koeln_weiden.csv.")
    view = window_frame(frame, dataset_id)
    numeric = [column for column in view.columns if pd.api.types.is_numeric_dtype(view[column]) and column not in {"time_day"}]
    if not numeric:
        note("There is no numeric column to plot.")
        return
    x = "time_day" if item.get("time_axis") == "simulation_day" else "timestamp"
    chosen = st.multiselect("Variables", numeric, default=numeric[:2], format_func=lambda key: rename.get(key, key))
    if not chosen:
        return
    st.plotly_chart(line_chart(view, x, chosen, rename), use_container_width=True)

    if len(chosen) >= 2:
        st.subheader("Relationship")
        fig = go.Figure(
            go.Scatter(
                x=view[chosen[0]],
                y=view[chosen[1]],
                mode="markers",
                marker={"color": "#0c6b63", "size": 7, "opacity": 0.7},
            )
        )
        fig.update_layout(xaxis_title=rename.get(chosen[0], chosen[0]), yaxis_title=rename.get(chosen[1], chosen[1]))
        st.plotly_chart(style_fig(fig, 380), use_container_width=True)

    if dataset_id == "bsm1_influent":
        _bsm1_overlay(frame)
    if dataset_id == "uci_water_treatment":
        _uci_removal()
    if dataset_id in {"dhi_wwtp_inflow", "koeln_weiden"}:
        _diurnal(frame, dataset_id, rename)


def _aqua() -> None:
    table = performance_table()
    table = table[table["dataset_id"] == "aquafetch_nutrient_recovery"].copy()
    pollutant = st.selectbox("Recovery target", ["N", "P"], format_func=lambda value: "Nitrogen" if value == "N" else "Phosphorus")
    part = table[table["pollutant"] == pollutant].copy()
    params = part["input_parameters"].map(lambda text: json.loads(text) if isinstance(text, str) and text.startswith("{") else {})
    part["pH"] = params.map(lambda item: item.get("pH"))
    part["temperature"] = params.map(lambda item: item.get("temperature_C"))
    fig = go.Figure(
        go.Scatter(
            x=part["pH"],
            y=part["removal_efficiency"],
            mode="markers",
            marker={"color": part["temperature"], "colorscale": "Tealgrn", "showscale": True, "colorbar": {"title": "Temperature °C"}},
        )
    )
    fig.update_layout(title="Experimental pH and recovery", xaxis_title="pH", yaxis_title="Recovery %")
    st.plotly_chart(style_fig(fig, 420), use_container_width=True)
    note("These are laboratory adsorption-recovery experiments, not plant effluent removal.")


def _bsm1_overlay(frame) -> None:
    st.subheader("Flow in the three influent scenarios")
    fig = go.Figure()
    names = {"dry": "Dry", "rain": "Rain", "storm": "Storm"}
    for weather, part in frame.groupby("weather"):
        fig.add_trace(go.Scatter(x=part["time_day"], y=part["Q"], mode="lines", name=names.get(weather, weather)))
    fig.update_layout(xaxis_title="Simulation time (days)", yaxis_title="Flow m³/d")
    st.plotly_chart(style_fig(fig, 400), use_container_width=True)
    note("Each line uses its own file clock. The first half of the rain and storm files resembles dry weather. That is how the benchmark files are constructed.")


def _uci_removal() -> None:
    st.subheader("UCI plant-wide removal")
    table = performance_table()
    table = table[table["dataset_id"] == "uci_water_treatment"]
    fig = go.Figure()
    for pollutant, part in table.groupby("pollutant"):
        fig.add_trace(go.Box(y=part["removal_efficiency"], name=pollutant, marker_color="#0c6b63"))
    fig.update_layout(yaxis_title="Removal recorded in the original file (%)")
    st.plotly_chart(style_fig(fig, 380), use_container_width=True)


def _diurnal(frame, dataset_id: str, rename: dict) -> None:
    column = "flow" if dataset_id == "dhi_wwtp_inflow" else "inflow_L_s"
    if column not in frame.columns:
        return
    hour = pd.to_datetime(frame["timestamp"]).dt.hour
    profile = frame.groupby(hour)[column].median()
    fig = go.Figure(go.Scatter(x=profile.index, y=profile.values, mode="lines+markers", line={"color": "#0c6b63"}))
    fig.update_layout(title=f"Hour-of-day median · {rename.get(column, column)}", xaxis_title="Hour", yaxis_title="Original units")
    st.plotly_chart(style_fig(fig, 340), use_container_width=True)
