import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Topic Description",
    page_icon=":scroll:",
    layout="wide",
)
# st.sidebar.success("Select a page above.")


@st.cache_data
def load_description_template():
    df = pd.read_csv(
        "data/Description_DataRole_template_EN.csv",
        index_col=0,
    )
    return df


descr_df = load_description_template()
st.table(descr_df)  # , use_container_width=True, hide_index=True)


# default_df = load_template()
if "default_description" not in st.session_state:
    st.session_state["default_description"] = descr_df
