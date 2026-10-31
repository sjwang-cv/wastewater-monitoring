import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from wwbench.ui import MODEL_ZH, note, page_header, run_summary, scenario_results, style_fig, synthetic_results


def page() -> None:
    page_header(
        "Stage 2",
        "Model evaluation",
        "The headline metrics come from synthetic anomaly tests. No anomalies are injected into the reference period, and thresholds are not chosen on the test labels. These scores are not real-fault detection performance.",
    )
    note("The UCI data file has no fault labels. The classes in the .names file are the authors' conceptual clusters. Precision, recall, and F1 below apply only to injected point anomalies and short drifts.", warn=True)
    results = synthetic_results()
    dataset_id = st.selectbox(
        "Synthetic-test dataset",
        list(results),
        format_func=lambda key: "UCI plant daily monitoring" if key == "uci_water_treatment" else "BSM1 dry-weather influent",
    )
    block = results[dataset_id]
    st.caption(block["protocol"])
    cols = st.columns(3)
    for column, (name, metrics) in zip(cols, block["metrics"].items()):
        column.metric(MODEL_ZH[name], f"{metrics['f1']:.3f}")
        column.caption(f"Precision {metrics['precision']:.3f} · Recall {metrics['recall']:.3f}")
        if metrics.get("roc_auc") is not None:
            column.caption(f"ROC-AUC {metrics['roc_auc']:.3f}")
    rf = block["metrics"]["random_forest"]
    if rf.get("decision_threshold") is not None:
        st.caption(
            f"The Random Forest decision threshold {rf['decision_threshold']:.2f} is chosen only on the training synthetic labels, where train F1 is {rf['train_f1']:.3f}. "
            "The test score is usually lower: the model memorized the training injections rather than learning a stable fault rule."
        )
    left, right = st.columns(2)
    with left:
        st.subheader("Confusion matrix")
        model_name = st.radio("Model", list(block["metrics"]), format_func=lambda key: MODEL_ZH[key], horizontal=True)
        matrix = np.array(block["metrics"][model_name]["confusion_matrix"])
        fig = go.Figure(
            go.Heatmap(
                z=matrix,
                x=["Predicted normal", "Predicted anomaly"],
                y=["Actual normal", "Actual anomaly"],
                text=matrix,
                texttemplate="%{text}",
                colorscale="Teal",
                showscale=False,
            )
        )
        st.plotly_chart(style_fig(fig, 340), use_container_width=True)
    with right:
        st.subheader("Detection latency")
        latency_rows = []
        for name, latency in block["latency"].items():
            latency_rows.append(
                {
                    "Model": MODEL_ZH[name],
                    "Events": latency["events"],
                    "Detected": latency["detected_events"],
                    "Missed": latency["missed_events"],
                    "Mean latency (steps)": None if latency["mean_latency_steps"] is None else round(latency["mean_latency_steps"], 2),
                    "Mean latency (hours)": None if latency["mean_latency_hours"] is None else round(latency["mean_latency_hours"], 2),
                }
            )
        st.dataframe(pd.DataFrame(latency_rows), use_container_width=True, hide_index=True)
        st.caption("Latency is counted from the start of a synthetic event to the first detection inside its window. Missed events are excluded from the mean.")

    st.subheader("ROC")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line={"dash": "dot", "color": "#94a3a0"}, name="Chance"))
    for name, metrics in block["metrics"].items():
        curve = metrics.get("roc")
        if curve:
            fig.add_trace(go.Scatter(x=curve["fpr"], y=curve["tpr"], mode="lines", name=MODEL_ZH[name]))
    fig.update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate")
    st.plotly_chart(style_fig(fig, 380), use_container_width=True)

    st.subheader("Parameter sensitivity")
    st.caption("The threshold or contamination is changed on the same synthetic test labels. This is not a new independent test.")
    sens = block["sensitivity_zscore"]
    fig = go.Figure(go.Scatter(x=[row["threshold"] for row in sens], y=[row["f1"] for row in sens], mode="lines+markers", name="Z-score F1", line={"color": "#0c6b63"}))
    fig.update_layout(xaxis_title="Z-score threshold", yaxis_title="F1")
    c1, c2 = st.columns(2)
    c1.plotly_chart(style_fig(fig, 320), use_container_width=True)
    sens_if = block["sensitivity_isolation_forest"]
    fig = go.Figure(go.Bar(x=[str(row["contamination"]) for row in sens_if], y=[row["f1"] for row in sens_if], marker_color="#b45309"))
    fig.update_layout(xaxis_title="contamination", yaxis_title="F1")
    c2.plotly_chart(style_fig(fig, 320), use_container_width=True)

    st.subheader("BSM1 influent-scenario classification")
    scenario = scenario_results()
    a, b = st.columns(2)
    a.metric("Accuracy", f"{scenario['accuracy']:.3f}")
    b.metric("Macro F1", f"{scenario['macro_f1']:.3f}")
    note(scenario["task"])
    labels = scenario["labels"]
    matrix = np.array(scenario["confusion_matrix"])
    fig = go.Figure(
        go.Heatmap(
            z=matrix,
            x=[f"Predicted {name}" for name in labels],
            y=[f"File {name}" for name in labels],
            text=matrix,
            texttemplate="%{text}",
            colorscale="Teal",
            showscale=False,
        )
    )
    st.plotly_chart(style_fig(fig, 360), use_container_width=True)
    st.caption("Train and test are split by day. The first half of the rain and storm files resembles dry weather, so the file-level label is mixed.")

    summary = run_summary()
    overlap = summary["screening"]["uci_water_treatment"].get("cluster_overlap")
    if overlap:
        st.subheader("Overlap with the UCI conceptual clusters")
        note(overlap["note"])
        rows = []
        for method, groups in overlap.items():
            if method == "note":
                continue
            for kind, values in groups.items():
                rows.append({"Model": MODEL_ZH.get(method, method), "Cluster": kind, "Days": values["days"], "Flagged as outliers": values["flagged"]})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
