import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Coefficient Setting",
    page_icon=":lower_left_fountain_pen:",
    layout="wide",
)

@st.cache_data
def load_template():
    df = pd.read_csv("default_template.csv", index_col=2)
    return df


df = load_template()
edited_df = st.data_editor(df, num_rows='dynamic',use_container_width=True)

if st.button("Save Coefficients"):
    st.session_state["coeff_df"] = edited_df

