import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Evaluation",
    page_icon=":lower_left_fountain_pen:",
    layout="wide",
)
st.sidebar.success("Select a page above.")
st.markdown("# Candidate Evaluation")

st.write(
    """
Here you'll evaluate your candidate based on the topics which appear in the coefficient table.
In order to be as unbiased as possible, the coefficient won't be visible in this step of the process.
Don't forget to submit your evaluation at the end!
"""
)

if "clicked_save_eval" not in st.session_state:
    st.session_state.clicked_save_eval = False

if "random_eval_key" not in st.session_state:
    st.session_state["random_eval_key"] = 0

if (
    "rerun_editing_eval" not in st.session_state
    or "current_rerun_editing_eval" not in st.session_state
):
    st.session_state["rerun_editing_eval"] = 0
    st.session_state["current_rerun_editing_eval"] = 0


def check_unique_true(df):
    for index, row in df.iterrows():
        if row.sum() != 1:
            return False
    return True


if "levels" not in st.session_state:
    st.session_state["levels"] = ["Junior", "Confirmed", "Expert"]
if "edited_df" not in st.session_state:
    st.warning(
        " It seems that you didn't modify or save your own coefficient setting. Default template will be used.",
        icon="⚠️",
    )
    df = st.session_state["default_coeff"]
else:
    df = st.session_state["edited_df"]
subdf = df[df.columns[:2]]
subdf[st.session_state["levels"]] = np.full(
    (len(subdf), len(st.session_state["levels"])), False
)
st.session_state["empty_eval"] = subdf

columns = subdf.columns
edited_evaluation = st.data_editor(
    st.session_state["empty_eval"],
    column_config={
        columns[0]: st.column_config.TextColumn(disabled=True),
        columns[1]: st.column_config.TextColumn(disabled=True),
        columns[2]: st.column_config.TextColumn(disabled=True),
        "Junior": st.column_config.CheckboxColumn(
            None,
            help="Is the candidate Junior?",
        ),
    },
    use_container_width=True,
    key=st.session_state["random_eval_key"],
)


col0, col1, col2, col3 = st.columns([1, 1, 1, 1])
with col1:
    if st.button("Save Evaluation"):
        st.session_state.clicked_save_eval = True

with col2:
    if st.button("Reset Values"):
        st.cache_data.clear()
        st.session_state["random_eval_key"] += 1
        if "Final_Evaluation" in st.session_state:
            del st.session_state["Final_Evaluation"]
            st.session_state["rerun_editing_eval"] += 1
        else:
            st.session_state["rerun_editing_eval"] += 1

        if "Final_df" in st.session_state:
            del st.session_state["Final_df"]
        st.rerun()


if st.session_state.clicked_save_eval:
    st.session_state["Final_Evaluation"] = edited_evaluation
    is_valid = check_unique_true(edited_evaluation[columns[2:]])

    if not is_valid:
        # st.session_state["trigger_invalid_data"] = True
        st.warning(
            " Invalid Data, there should be 1 level selected for each topic.",
            icon="⚠️",
        )
    else:
        st.info("Coefficient saved.", icon="ℹ️")
        final_df = df.reset_index().merge(edited_evaluation.reset_index())
        st.session_state["Final_df"] = final_df
        st.dataframe(final_df.set_index("Sujet"))

    st.session_state.clicked_save_eval = False


if (
    st.session_state["rerun_editing_eval"]
    == st.session_state["current_rerun_editing_eval"] + 1
):
    st.session_state["current_rerun_editing_eval"] += 1
    st.info("Values have been reset.", icon="ℹ️")


# st.dataframe(final_df.set_index("Sujet"))
