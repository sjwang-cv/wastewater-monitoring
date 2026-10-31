import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from wwbench.modeling import apply_reference
from wwbench.ui import (
    choose_monitor_dataset,
    dataset_map,
    labels_for,
    modeling_view,
    note,
    page_header,
    reference_for,
    score_column,
    screen_table,
    style_fig,
    window_frame,
)


def page() -> None:
    page_header(
        "Stage 2",
        "Process monitoring",
        "The reference baseline and flagged outliers are shown together. The band is a statistical interval from the reference standard deviation. It is not an effluent limit or a process control limit.",
    )
    dataset_id = choose_monitor_dataset()
    spec = dataset_map()[dataset_id]
    frame = modeling_view(dataset_id).reset_index(drop=True)
    view = window_frame(frame, dataset_id)
    rename = labels_for(dataset_id)
    reference = reference_for(dataset_id)
    columns = [column for column in reference["feature_cols"] if column in view.columns]
    chosen = st.multiselect("Monitored variables", columns, default=columns[:1], format_func=lambda key: rename.get(key, key))
    method = st.radio(
        "Outlier marker",
        ["zscore", "isolation_forest"],
        format_func=lambda key: "Z-score" if key == "zscore" else "Isolation Forest",
        horizontal=True,
    )
    if not chosen:
        return
    if spec.get("model_resolution") == "1h":
        st.caption("Detection and the curve use hourly means.")
    if spec.get("time_axis") == "simulation_day":
        st.caption("The reference is the dry-weather diurnal pattern. Rain or storm deviations from dry weather are marked as scenario differences, not fault labels.")

    z = apply_reference(frame, reference).copy()
    z.columns = [f"z__{column}" for column in z.columns]
    z["sample_id"] = frame["sample_id"].to_numpy()
    table = screen_table(dataset_id)
    aligned = view.merge(
        table[["sample_id", "score_z", "flag_z", "score_if", "flag_if"]],
        on="sample_id",
        how="left",
    ).merge(z, on="sample_id", how="left")
    column = chosen[0]
    sigma = float(reference["sigma"][column])
    deviation = aligned[f"z__{column}"]
    base = aligned[column] - deviation * sigma
    x = "time_day" if spec.get("time_axis") == "simulation_day" else "timestamp"
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=aligned[x], y=base + 3 * sigma, mode="lines", line={"width": 0}, showlegend=False, hoverinfo="skip"))
    fig.add_trace(
        go.Scatter(
            x=aligned[x],
            y=base - 3 * sigma,
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor="rgba(12,107,99,0.12)",
            name="Baseline ± 3 reference standard deviations",
            hoverinfo="skip",
        )
    )
    fig.add_trace(go.Scatter(x=aligned[x], y=aligned[column], mode="lines", name=rename.get(column, column), line={"color": "#123c3a", "width": 1.7}))
    _, flag_name = score_column(method)
    flags = aligned[flag_name].fillna(False).astype(bool)
    if flags.any():
        fig.add_trace(
            go.Scatter(
                x=aligned.loc[flags, x],
                y=aligned.loc[flags, column],
                mode="markers",
                name="Outliers",
                marker={"color": "#9f1239", "size": 8, "symbol": "diamond"},
            )
        )
    for extra in chosen[1:]:
        fig.add_trace(go.Scatter(x=aligned[x], y=aligned[extra], mode="lines", name=rename.get(extra, extra)))
    st.plotly_chart(style_fig(fig, 460), use_container_width=True)
    model_name = "Z-score" if method == "zscore" else "Isolation Forest"
    note(f"In the current window, {model_name} flags {int(flags.sum())} points. Series with a daily or weekly cycle first subtract the reference median at the same time of day or week.")
