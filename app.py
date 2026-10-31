import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

st.set_page_config(
    page_title="Wastewater Monitoring Benchmark",
    layout="wide",
    initial_sidebar_state="expanded",
)

from views.anomalies import page as anomalies_page
from views.evaluation import page as evaluation_page
from views.explain import page as explain_page
from views.explorer import page as explorer_page
from views.monitor import page as monitor_page
from views.overview import page as overview_page
from views.statistics import page as statistics_page
from views.tasks import page as tasks_page
from views.visualize import page as visualize_page

st.navigation(
    {
        "Stage 1  Benchmark": [
            st.Page(overview_page, title="Overview", url_path="overview", default=True),
            st.Page(explorer_page, title="Datasets", url_path="datasets"),
            st.Page(statistics_page, title="Statistics", url_path="statistics"),
            st.Page(tasks_page, title="Tasks", url_path="tasks"),
            st.Page(visualize_page, title="Visualization", url_path="visualize"),
        ],
        "Stage 2  Monitoring": [
            st.Page(monitor_page, title="Process monitoring", url_path="monitor"),
            st.Page(anomalies_page, title="Anomaly detection", url_path="anomalies"),
            st.Page(explain_page, title="Explanation", url_path="explain"),
            st.Page(evaluation_page, title="Evaluation", url_path="evaluation"),
        ],
    }
).run()
