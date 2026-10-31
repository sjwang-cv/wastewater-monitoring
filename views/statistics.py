import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from wwbench.ui import catalog, correlation_figure, histogram, labels_for, note, page_header, profiles, style_fig


def page() -> None:
    items = [item for item in catalog() if item["bundled"]]
    page_header("Stage 1", "Statistics", "Missingness, distributions, and correlations come from the bundled tables. Datasets with different units are never concatenated before these calculations.")
    labels = {item["dataset_id"]: item["name"] for item in items}
    dataset_id = st.selectbox("Dataset", list(labels), format_func=lambda key: labels[key])
    item = next(entry for entry in items if entry["dataset_id"] == dataset_id)
    profile = profiles()[dataset_id]
    if item.get("time_min"):
        st.caption(f"Time range: {item['time_min']} to {item['time_max']}. Cologne charts use hourly means; the original 15-minute table is kept.")
    rename = labels_for(dataset_id)
    columns = list(profile["columns"])
    missing = pd.DataFrame(
        {
            "Variable": [rename.get(column, column) for column in columns],
            "Missing rate": [profile["columns"][column]["missing_rate"] for column in columns],
        }
    ).sort_values("Missing rate", ascending=False)
    fig = go.Figure(go.Bar(x=missing["Missing rate"], y=missing["Variable"], orientation="h", marker_color="#0c6b63"))
    fig.update_layout(title="Missing rate", xaxis_tickformat=".0%", xaxis_title="Missing fraction")
    st.plotly_chart(style_fig(fig, max(320, 28 * len(columns))), use_container_width=True)

    checks = profile.get("documented_missing_check")
    if checks:
        matched = sum(item["match"] for item in checks)
        note(f"The UCI .names file lists a missing count for each column. {matched}/{len(checks)} columns in the current file match that document.")
        if matched != len(checks):
            bad = [row for row in checks if not row["match"]]
            st.dataframe(pd.DataFrame(bad), hide_index=True)

    numeric = [column for column in columns if "median" in profile["columns"][column]]
    if not numeric:
        note("This dataset has no numeric distribution to plot.")
        return
    column = st.selectbox("Distribution", numeric, format_func=lambda key: rename.get(key, key))
    entry = profile["columns"][column]
    a, b, c, d = st.columns(4)
    a.metric("Median", f"{entry['median']:.3g}")
    b.metric("Mean", f"{entry['mean']:.3g}")
    c.metric("Minimum", f"{entry['min']:.3g}")
    d.metric("Maximum", f"{entry['max']:.3g}")
    st.plotly_chart(histogram(profile, column, rename.get(column, column)), use_container_width=True)
    corr = correlation_figure(profile, rename)
    if corr is not None:
        st.subheader("Correlation")
        st.caption("Pearson correlation on pairwise complete samples. Correlation is not causation.")
        st.plotly_chart(corr, use_container_width=True)
