import pandas as pd
import streamlit as st

from wwbench.ui import SUPPORT_ZH, catalog, note, page_header, support_figure


def page() -> None:
    items = catalog()
    page_header(
        "Stage 1",
        "Task catalog",
        "Support describes whether a dataset can carry the task, not a model score. Full means the variables and labels are sufficient. Partial means only a proxy task is possible, or a gold-standard label is missing.",
    )
    st.plotly_chart(support_figure(), use_container_width=True)
    st.caption("Teal is full support, ochre is partial support, and grey is not supported.")
    task_names = [task["name"] for task in items[0]["tasks"]]
    task_name = st.selectbox("Task", task_names)
    rows = []
    for item in items:
        task = next(entry for entry in item["tasks"] if entry["name"] == task_name)
        rows.append(
            {
                "Dataset": item["name"],
                "Support": SUPPORT_ZH[task["support"]],
                "Bundled": "Yes" if item["bundled"] else "No",
                "Rationale": task["rationale"],
            }
        )
    order = {"Full": 0, "Partial": 1, "Not supported": 2}
    table = pd.DataFrame(rows).sort_values("Support", key=lambda series: series.map(order))
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.subheader("What public data still lacks")
    note("Microorganism recognition has MD-AS-2025, but the annotated set is about 2 GB and the images are not copied into this repository.")
    note("Anomaly detection with attack labels has SWaT. Redistribution is forbidden, and it is a water-treatment testbed, not a municipal wastewater plant.")
    note("Among the processable municipal series, there is no plant time series with measured dissolved oxygen, MLSS, or total phosphorus. Ammonia nitrogen appears only in the simulated BSM1 influent.")
    note("UCI supports process-performance analysis and unsupervised screening. Treating its conceptual clusters or synthetic points as real faults would overstate how well the system works.")
