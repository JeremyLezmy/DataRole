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
    # Data Role Assessment App

Welcome to the Data Role Assessment App!

This app allows you to evaluate candidates for various data roles based on their skills and experiences.

Here the steps to follow:
"""
)

st.markdown(
    f"""
## Coefficient setting

A unique coefficient is attributed for each topic (such as Python, SQL, ETL, Dataiku, etc).  
We suggest a default coefficient distribution (see below), but you can set and save your own configuration in the 
        _Coefficient Setting_ page.
        """,
    unsafe_allow_html=True,
)

st.markdown(
    """
### Default Coefficients:
"""
)


@st.cache_data
def load_template():
    df = pd.read_csv("data/default_template.csv", index_col=2)
    return df


default_df = load_template()
st.dataframe(default_df, use_container_width=True)

if st.button("Set your coefficients here!"):
    st.switch_page("pages/01_Coefficient_Setting.py")

if "default_coeff" not in st.session_state:
    st.session_state["default_coeff"] = default_df

if "levels" not in st.session_state:
    st.session_state["levels"] = ["No Experience", "Junior", "Confirmed", "Senior"]

st.markdown(
    f""" 
            ## Assessment  

Based on these coefficients, the interviewer will select a level for the candidate among Junior, Confirmed or Senior.  
The level will act as a multiplicator over the coefficients.
This step is processed in the _Assessment_ page, and must be finalized by using the _Submit_ button at the bottom. 
""",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
            ## Topic description

            The _Topic Description_ page arbitrary describes the expected knowledges and experiences for each evaluated topic and for each level (Junior, Confirmed, Senior).
            """,
    unsafe_allow_html=True,
)


st.markdown(
    f"""
            ## Groq Assistant

            Here you'll find an AI assistant implemented using [_Groq AI_](https://groq.com/).  

            You'll be able to use any of their implemented open-source LLM as core AI,
            from Mistral to the last in date LLama 3 70B model!   
            This page becomes fully unlocked once you saved and submitted your assessment:
            more than a chatbot, the LLM is pretrained with the assessment results.  
            You'll be able to get insights, speak about the results, and even
            ask him to plot some figures! We indeed implemented a code detector in the AI responses that you can execute directly in the application.  

            As you might guess, this feature is API powered. Currently Groq allows the free use of their API services, so feel free to use it. However, this feature
            might not be indefinitely availbale.

            """,
    unsafe_allow_html=True,
)
