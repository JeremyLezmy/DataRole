import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Coefficient Setting",
    page_icon=":lower_left_fountain_pen:",
    layout="wide",
)
st.sidebar.success("Select a page above.")
st.markdown(
    """
    # Coefficient Setting

This page allows you to interact with the default coefficient template by modifying them, add/remove rows, or even modifying the topic that you want to evaluate.
Don't forget to _save the edited dataframe_ to use your custom version.
"""
)

if "clicked_save_coeff" not in st.session_state:
    st.session_state.clicked_save_coeff = False

if "random_key" not in st.session_state:
    st.session_state["random_key"] = 0

if (
    "rerun_editing_coeff" not in st.session_state
    or "current_rerun_editing_coeff" not in st.session_state
):
    st.session_state["rerun_editing_coeff"] = 0
    st.session_state["current_rerun_editing_coeff"] = 0


@st.cache_data
def load_template():
    df = pd.read_csv("default_template.csv", index_col=2)
    return df


# default_df = load_template()
if "default_coeff" not in st.session_state:
    st.session_state["default_coeff"] = load_template()

if "edited_df" in st.session_state:
    edited_df = st.data_editor(
        st.session_state["edited_df"],
        num_rows="dynamic",
        use_container_width=True,
        key=st.session_state["random_key"],
    )
else:
    edited_df = st.data_editor(
        st.session_state["default_coeff"],
        num_rows="dynamic",
        use_container_width=True,
        key=st.session_state["random_key"],
    )

col0, col1, col2, col3 = st.columns([1, 1, 1, 1])
with col1:
    if st.button("Save Coefficients"):
        st.session_state.clicked_save_coeff = True

with col2:
    if st.button("Reset Values"):
        st.cache_data.clear()
        st.session_state["random_key"] += 1
        # st.session_state["edited_df"] = st.session_state["default_coeff"]
        if "edited_df" in st.session_state:
            del st.session_state["edited_df"]
            st.session_state["rerun_editing_coeff"] += 1
        else:
            st.session_state["rerun_editing_coeff"] += 1
        st.rerun()


if st.session_state.clicked_save_coeff:

    st.session_state["edited_df"] = edited_df
    st.info("Coefficient saved.", icon="ℹ️")
    st.session_state.clicked_save_coeff = False


if (
    st.session_state["rerun_editing_coeff"]
    == st.session_state["current_rerun_editing_coeff"] + 1
):
    st.session_state["current_rerun_editing_coeff"] += 1
    st.info("Values have been reset.", icon="ℹ️")
