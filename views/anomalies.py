import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from wwbench.modeling import model_features, screen
from wwbench.spec import dataset_by_id
from wwbench.ui import (
    MODEL_ZH,
    choose_monitor_dataset,
    dataset_map,
    labels_for,
    modeling_view,
    note,
    page_header,
    reference_for,
    report_markdown,
    score_column,
    screen_table,
    style_fig,
    top_anomalies,
)


def page() -> None:
    page_header(
        "Stage 2",
        "Anomaly detection",
        "Default results come from a Z-score and an Isolation Forest fitted on the reference period. An uploaded CSV uses the same procedure. An anomaly here is a statistical outlier.",
    )
    source = st.radio("Data source", ["Benchmark dataset", "Upload CSV"], horizontal=True)
    if source == "Upload CSV":
        _upload()
        return

    dataset_id = choose_monitor_dataset()
    spec = dataset_map()[dataset_id]
    rename = labels_for(dataset_id)
    reference = reference_for(dataset_id)
    features = st.multiselect(
        "Variables used by the model",
        reference["feature_cols"],
        default=reference["feature_cols"],
        format_func=lambda key: rename.get(key, key),
    )
    method = st.radio("Model", ["zscore", "isolation_forest"], format_func=lambda key: MODEL_ZH[key], horizontal=True)
    threshold = st.slider("Z-score threshold", 2.0, 5.0, float(reference["threshold"]), 0.5)
    contamination = st.select_slider("Isolation Forest contamination", options=[0.02, 0.05, 0.1], value=float(reference["contamination"]))
    live = st.session_state.get("live")
    use_live = live and live.get("dataset_id") == dataset_id and live.get("features") == tuple(features)
    if st.button("Run with these settings", type="primary"):
        frame = modeling_view(dataset_id)
        config = {
            "seasonal_mode": spec["seasonal_mode"],
            "train_frac": spec["train_frac"],
        }
        if spec.get("reference_column"):
            config["reference_column"] = spec["reference_column"]
            config["reference_value"] = spec["reference_value"]
        if dataset_id == "bsm1_influent":
            config["day_column"] = "time_day"
        with st.spinner("Fitting on the reference period"):
            result = screen(frame, features, config, threshold=threshold, contamination=contamination, n_estimators=100)
        st.session_state["live"] = {
            "dataset_id": dataset_id,
            "table": result["table"],
            "frame": result["frame"],
            "z": result["z"],
            "reference": result["reference"],
            "model": result["model"],
            "features": tuple(features),
        }
        use_live = True
        live = st.session_state["live"]

    table = live["table"] if use_live else screen_table(dataset_id)
    if not use_live:
        note("These are the precomputed default results. Change the variables or settings, then run again to update them.")
    score_name, flag_name = score_column(method)
    flagged = int(table[flag_name].astype(bool).sum()) if flag_name in table.columns else 0
    a, b, c = st.columns(3)
    a.metric("Samples", f"{len(table):,}")
    b.metric("Outliers", f"{flagged:,}")
    c.metric("Variables", len(features))
    fig = go.Figure(go.Scatter(x=table["timestamp"], y=table[score_name], mode="lines", line={"color": "#0c6b63", "width": 1.2}, name="Anomaly score"))
    marked = table.loc[table[flag_name].astype(bool)]
    fig.add_trace(go.Scatter(x=marked["timestamp"], y=marked[score_name], mode="markers", name="Outliers", marker={"color": "#9f1239", "size": 7}))
    fig.update_layout(title="Anomaly score", yaxis_title="Score (larger means farther from the reference)")
    st.plotly_chart(style_fig(fig, 380), use_container_width=True)
    st.subheader("Largest deviations")
    top = top_anomalies(table, method)
    show = top.copy()
    show["Time"] = show["timestamp"].astype(str)
    show["Score"] = show[score_name].map(lambda value: round(float(value), 3))
    show["Top variable"] = show["top_variable_z"].map(lambda key: rename.get(key, key))
    columns = ["Time", "Score", "Top variable"]
    if "weather" in show.columns:
        show["Scenario"] = show["weather"]
        columns.append("Scenario")
    if "author_cluster_kind" in show.columns:
        show["Documented cluster"] = show["author_cluster_kind"]
        columns.append("Documented cluster")
    st.dataframe(show[columns], use_container_width=True, hide_index=True)
    report = report_markdown(spec["name"], method, table, features)
    st.download_button("Download monitoring report", report, file_name=f"{dataset_id}-screening.md")


def _upload() -> None:
    file = st.file_uploader("CSV monitoring data", type=["csv"])
    if file is None:
        note("The header must be readable by pandas. Choose a time column and numeric columns, then run.")
        return
    frame = pd.read_csv(file)
    st.dataframe(frame.head(8), use_container_width=True)
    time_options = ["(use row number as time)", *list(frame.columns)]
    time_col = st.selectbox("Time column", time_options)
    numeric = [column for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])]
    features = st.multiselect("Numeric variables", numeric, default=numeric[: min(6, len(numeric))])
    seasonal = st.selectbox("Seasonal correction", ["none", "time_of_day", "hour_of_week"], format_func=lambda key: {"none": "None", "time_of_day": "Time of day", "hour_of_week": "Hour of week"}[key])
    if not features or not st.button("Run uploaded data", type="primary"):
        return
    work = frame.copy()
    work["sample_id"] = [f"upload-{index:05d}" for index in range(len(work))]
    if time_col == "(use row number as time)":
        work["timestamp"] = pd.date_range("2000-01-01", periods=len(work), freq="h")
    else:
        work["timestamp"] = pd.to_datetime(work[time_col], errors="coerce")
        work = work.dropna(subset=["timestamp"])
    specs = [{"feature_name": column, "use_in_model": True} for column in features]
    columns = model_features(work, specs)
    with st.spinner("Detecting"):
        result = screen(work, columns, {"seasonal_mode": seasonal, "train_frac": 0.7}, n_estimators=100)
    st.session_state["live"] = {
        "dataset_id": "upload",
        "table": result["table"],
        "frame": result["frame"],
        "z": result["z"],
        "reference": result["reference"],
        "model": result["model"],
        "features": tuple(columns),
        "rename": {column: column for column in columns},
    }
    table = result["table"]
    note(f"Uploaded data has {len(table)} rows. Z-score flags {int(table['flag_z'].sum())} rows and Isolation Forest flags {int(table['flag_if'].sum())} rows.")
    st.dataframe(table.sort_values("score_if", ascending=False).head(20), use_container_width=True, hide_index=True)
