import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Home",
    page_icon="👋",
    layout="wide",
)

st.sidebar.success("Select a page above.")

st.markdown(
    """
    # Data Role Evaluation App

Welcome to the Data Role Evaluation App!

This app allows you to evaluate candidates for different data roles based on their skills and experience.

Here the steps to follow:
"""
)

st.markdown(
    """
## Coefficient setting

A unique coefficient is attributed for each topic (such as Python, SQL, ETL, Dataiku, etc).  
We suggest a default coefficient distribution (see below), but you can set and save your own configuration in the Coefficient Setting page.

        """
)

st.markdown(
    """
### Default Coefficients:
"""
)


@st.cache_data
def load_template():
    df = pd.read_csv("default_template.csv", index_col=2)
    return df


default_df = load_template()
st.dataframe(default_df, use_container_width=True)

if st.button("Set your coefficients here!"):
    st.switch_page("pages/01_Coefficient_Setting.py")

if "default_coeff" not in st.session_state:
    st.session_state["default_coeff"] = default_df

if "levels" not in st.session_state:
    st.session_state["levels"] = ["Junior", "Confirmed", "Expert"]

st.markdown(
    """ 
            ## Evaluation  

Based on these coefficients, the interviewer will select a status for the candidate among Junior, Intermediate, Confirmed and Expert.  
The status will act as a multiplicator over the coefficients.
This step is processed in the _Evaluation_ page, and must be finalized by using the _Submit_ button at the bottom. 
"""
)

st.markdown(
    """
            ## Results

After the Evaluation step, the Results page will display some analysis, such as a spider graph showing a score for each data role for the candidate, or a summary dataframe.
            The user will be allowed to download a report.


            """
)

st.markdown(
    """
            ## Descriptions

            This page describes the expected knowledges and experiences for each evaluated topic and for each status (junion, intermediate, confirmed, expert).
            """
)
