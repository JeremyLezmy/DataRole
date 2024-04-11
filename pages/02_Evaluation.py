import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Evaluation",
    page_icon=":lower_left_fountain_pen:",
    layout="wide",
)

st.markdown("# Candidate Evaluation")

st.write(
    """
Here you'll evaluate your candidate based on the topics which appear in the coefficient table.
In order to be as unbiased as possible, the coefficient won't be visible in this step of the process.
Don't forget to submit your evaluation at the end!
"""
)
