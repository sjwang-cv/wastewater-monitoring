import pandas as pd
import streamlit as st

from wwbench.build import feature_dictionary
from wwbench.ui import CATEGORY_ZH, SUPPORT_ZH, catalog, note, page_header


def page() -> None:
    items = catalog()
    page_header(
        "Stage 1",
        "Datasets",
        "Filter by modality, domain, and task. Each record states the source, license, what the labels can support, and whether the files are bundled.",
    )
    modalities = ["All", *sorted({item["modality"] for item in items})]
    domains = ["All", *sorted({item["domain"] for item in items})]
    categories = ["All", *sorted({item["category"] for item in items})]
    tasks = ["All", *{task["name"] for item in items for task in item["tasks"]}]
    c1, c2, c3, c4 = st.columns(4)
    modality = c1.selectbox("Modality", modalities)
    domain = c2.selectbox("Domain", domains)
    category = c3.selectbox("Category", categories, format_func=lambda value: value if value == "All" else CATEGORY_ZH[value])
    task_name = c4.selectbox("At least partially supported task", tasks)
    bundled_only = st.checkbox("Bundled data only", value=False)

    chosen = []
    for item in items:
        if modality != "All" and item["modality"] != modality:
            continue
        if domain != "All" and item["domain"] != domain:
            continue
        if category != "All" and item["category"] != category:
            continue
        if bundled_only and not item["bundled"]:
            continue
        if task_name != "All":
            matched = [task for task in item["tasks"] if task["name"] == task_name and task["support"] != "none"]
            if not matched:
                continue
        chosen.append(item)
    if not chosen:
        note("No dataset matches the current filters.")
        return

    table = pd.DataFrame(
        [
            {
                "Name": item["name"],
                "Category": CATEGORY_ZH[item["category"]],
                "Modality": item["modality"],
                "Domain": item["domain"],
                "Samples": "Not bundled" if item["n_samples"] is None else f"{item['n_samples']:,}",
                "License": item["license"],
                "Bundled": "Yes" if item["bundled"] else "No",
            }
            for item in chosen
        ]
    )
    st.dataframe(table, use_container_width=True, hide_index=True)
    names = [item["name"] for item in chosen]
    selected_name = st.selectbox("Details", names)
    item = next(entry for entry in chosen if entry["name"] == selected_name)
    st.subheader(item["name"])
    note(item["description"])
    a, b, c = st.columns(3)
    a.metric("Sampling", item["sampling_rate"])
    b.metric("Format", item["data_format"])
    c.metric("Status", item["availability"])
    st.markdown(f"**Source**  \n{item['source']}")
    st.markdown(f"**Citation**  \n{item['citation']}")
    st.markdown(f"**Labels**  \n{item['label_note']}")
    st.link_button("Open source", item["url"])
    if item.get("time_min"):
        st.caption(f"Time range: {item['time_min']} to {item['time_max']}")
    extra = item.get("extra") or {}
    if item["dataset_id"] == "mdas_2025":
        st.write(
            f"The paper reports {extra.get('annotated_images', 4000):,} annotated images and {extra.get('cropped_images', 14257):,} crops. "
            "The annotated archive is about 2.0 GB, so this repository keeps only the category index."
        )
    if item["dataset_id"] == "swat_a1":
        st.write(
            f"Published A1 characteristics: {extra.get('normal_days')} normal days, {extra.get('attack_days')} attack days, "
            f"{extra.get('channels')} channels, and {extra.get('attacks')} attacks. The terms forbid redistribution."
        )
    st.subheader("Task support")
    st.dataframe(
        pd.DataFrame(
            [{"Task": task["name"], "Support": SUPPORT_ZH[task["support"]], "Rationale": task["rationale"]} for task in item["tasks"]]
        ),
        use_container_width=True,
        hide_index=True,
    )
    if item["bundled"]:
        st.subheader("Fields")
        features = feature_dictionary(item["dataset_id"])
        show = features.rename(
            columns={
                "feature_name": "Field",
                "unit": "Unit",
                "role": "Role",
                "description": "Meaning",
                "use_in_model": "Used in the default model",
            }
        )
        st.dataframe(show[["Field", "Meaning", "Unit", "Role", "Used in the default model"]], use_container_width=True, hide_index=True)
