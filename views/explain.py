import joblib
import plotly.graph_objects as go
import streamlit as st

from wwbench.modeling import apply_reference, isolation_contributions
from wwbench.paths import RESULTS
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
    top_anomalies,
)


def page() -> None:
    page_header(
        "Stage 2",
        "Explanation",
        "Each variable is compared with the reference standard deviation. Isolation Forest also reports how much the anomaly score drops when that variable is returned to the reference.",
    )
    live = st.session_state.get("live")
    if live and live.get("dataset_id") == "upload":
        _explain_live(live, live.get("rename") or {})
        return
    dataset_id = choose_monitor_dataset()
    if live and live.get("dataset_id") == dataset_id:
        rename = labels_for(dataset_id)
        _explain_live(live, rename)
        return
    rename = labels_for(dataset_id)
    method = st.radio("Model used for the explanation", ["zscore", "isolation_forest"], format_func=lambda key: "Z-score" if key == "zscore" else "Isolation Forest", horizontal=True)
    table = screen_table(dataset_id)
    top = top_anomalies(table, method, limit=30)
    if top.empty:
        note("The current model did not flag any outliers.")
        return
    labels = [f"{row.timestamp} · {rename.get(row.top_variable_z, row.top_variable_z)}" for row in top.itertuples(index=False)]
    pick = st.selectbox("Outlier", labels)
    row = top.iloc[labels.index(pick)]
    frame = modeling_view(dataset_id).reset_index(drop=True)
    reference = reference_for(dataset_id)
    z = apply_reference(frame, reference)
    z.index = frame["sample_id"].to_numpy()
    z_row = z.loc[row.sample_id]
    _render(dataset_id, row, z_row, frame, reference, rename, method)


def _explain_live(live: dict, rename: dict) -> None:
    method = st.radio("Model", ["zscore", "isolation_forest"], format_func=lambda key: "Z-score" if key == "zscore" else "Isolation Forest", horizontal=True, key="explain_live_method")
    score_name, flag_name = score_column(method)
    table = live["table"]
    top = table.loc[table[flag_name].astype(bool)].sort_values(score_name, ascending=False).head(30)
    if top.empty:
        note("This run did not flag any outliers.")
        return
    labels = [f"{row.timestamp} · {rename.get(row.top_variable_z, row.top_variable_z)}" for row in top.itertuples(index=False)]
    pick = st.selectbox("Outlier", labels, key="explain_live_pick")
    row = top.iloc[labels.index(pick)]
    z = live["z"].copy()
    z.index = live["frame"]["sample_id"].to_numpy()
    _render(live["dataset_id"], row, z.loc[row.sample_id], live["frame"], live["reference"], rename, method, model=live["model"])


def _render(dataset_id, row, z_row, frame, reference, rename, method, model=None) -> None:
    order = z_row.abs().sort_values(ascending=False)
    names = [rename.get(name, name) for name in order.index]
    fig = go.Figure(go.Bar(x=order.values, y=names, orientation="h", marker_color=["#9f1239" if value >= 0 else "#1d4e89" for value in z_row.loc[order.index]]))
    fig.update_layout(title="Z-score against the reference baseline", xaxis_title="Reference standard deviations", yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(style_fig(fig, max(360, 28 * len(order))), use_container_width=True)
    bits = []
    for name, value in order.head(3).items():
        direction = "above" if value > 0 else "below"
        bits.append(f"{rename.get(name, name)} is {abs(value):.1f} standard deviations {direction} the reference baseline")
    note(". ".join(bits) + ".")
    actual = frame.set_index("sample_id").loc[row.sample_id]
    rows = []
    for name, value in order.items():
        sigma = reference["sigma"][name]
        measured = actual[name]
        baseline = None if pd_isna(measured) else float(measured) - float(value) * float(sigma)
        rows.append(
            {
                "Variable": rename.get(name, name),
                "Observed": None if pd_isna(measured) else round(float(measured), 4),
                "Reference baseline": None if baseline is None else round(baseline, 4),
                "Z-score": round(float(value), 2),
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)
    if method == "isolation_forest":
        if model is None and dataset_id != "upload":
            model = joblib.load(RESULTS / "models" / f"if_{dataset_id}.joblib")
        if model is not None:
            contrib = isolation_contributions(model, z_row[reference["feature_cols"]])
            fig = go.Figure(
                go.Bar(
                    x=contrib.values,
                    y=[rename.get(name, name) for name in contrib.index],
                    orientation="h",
                    marker_color="#b45309",
                )
            )
            fig.update_layout(title="Isolation Forest: how much this variable raises the score", xaxis_title="Score drop after returning the variable to 0", yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(style_fig(fig, max(360, 28 * len(contrib))), use_container_width=True)
            st.caption("A positive value means this variable pushes the sample away from the reference period. It is not a process diagnosis.")


def pd_isna(value) -> bool:
    try:
        return bool(value != value) or value is None
    except Exception:
        return False
