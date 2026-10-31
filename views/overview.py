import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from wwbench.spec import INDICATORS
from wwbench.ui import CATEGORY_ZH, COVERAGE_ZH, catalog, coverage_figure, note, page_header, run_summary, style_fig


def page() -> None:
    items = catalog()
    bundled = [item for item in items if item["bundled"]]
    samples = sum(item["n_samples"] or 0 for item in bundled)
    page_header(
        "Stage 1",
        "Wastewater Monitoring Benchmark",
        "Public datasets that can be downloaded, processed, and licensed clearly share one catalog. Different sampling rates and units stay in separate tables and are linked only by metadata and column contracts.",
    )
    left, mid, right, far = st.columns(4)
    left.metric("Catalogued datasets", len(items))
    mid.metric("Processable datasets", len(bundled))
    right.metric("Processable samples", f"{samples:,}")
    far.metric("Indicator slots", len(INDICATORS))
    st.caption("Sample counts include only files that were downloaded and standardized. MD-AS-2025 and SWaT stay in the catalog, but their files are not in this repository.")

    col1, col2 = st.columns(2)
    modality = pd.Series([item["modality"] for item in items]).value_counts()
    fig = go.Figure(go.Pie(labels=modality.index.tolist(), values=modality.values.tolist(), hole=0.55, marker={"colors": ["#0c6b63", "#1d4e89", "#b45309"]}))
    fig.update_layout(title="Modality")
    col1.plotly_chart(style_fig(fig, 340), use_container_width=True)
    counts = pd.Series([CATEGORY_ZH[item["category"]] for item in items]).value_counts()
    fig = go.Figure(go.Bar(x=counts.index.tolist(), y=counts.values.tolist(), marker_color="#0c6b63"))
    fig.update_layout(title="Four data categories", yaxis_title="Datasets")
    col2.plotly_chart(style_fig(fig, 340), use_container_width=True)

    st.subheader("Indicator coverage")
    note("Teal is plant measurement, blue is laboratory, ochre is simulation, and grey means the dataset does not have that indicator. BSM1 influent dissolved oxygen is defined as zero and is not counted as dissolved-oxygen coverage.")
    st.plotly_chart(coverage_figure(), use_container_width=True)
    st.caption(" · ".join(COVERAGE_ZH[key] for key in ("plant", "lab", "simulated", "absent")))

    st.subheader("Where the samples come from")
    fig = go.Figure(go.Bar(x=[item["n_samples"] for item in bundled], y=[item["name"] for item in bundled], orientation="h", marker_color="#1d4e89"))
    fig.update_layout(title="Processable samples", xaxis_title="Samples (log scale)", xaxis_type="log")
    st.plotly_chart(style_fig(fig, 380), use_container_width=True)
    note("The Cologne series is recorded every 15 minutes, so it has far more rows than the daily UCI record. The log axis keeps water-quality and hydraulic data visible together.")

    summary = run_summary()
    st.subheader("Link to Stage 2")
    note(
        f"The monitoring app uses the time series stored here. On UCI, Isolation Forest synthetic-test F1 is {summary['primary_f1_synthetic']:.3f}. "
        "That number describes synthetic anomalies only, not real faults."
    )
