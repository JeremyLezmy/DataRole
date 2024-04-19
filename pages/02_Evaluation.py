import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Evaluation",
    page_icon=":lower_left_fountain_pen:",
    layout="wide",
)

# st.sidebar.success("Select a page above.")
with st.sidebar:
    junior = st.slider("Junior bonus?", 0, 5, 1)
    confirmed = st.slider("Confirmed bonus?", 0, 5, 2)
    senior = st.slider("Senior Bonus?", 0, 5, 4)

    if not (senior > confirmed and confirmed > junior):
        st.warning(
            " Your bonus distribution is odd, you should respect Senior > Confirmed > Junior.",
            icon="⚠️",
        )
st.markdown("# Candidate Evaluation")

st.write(
    """
Here you'll evaluate your candidate based on the topics which appear in the coefficient table.
In order to be as unbiased as possible, the coefficient won't be visible in this step of the process.
Don't forget to submit your evaluation at the end!
"""
)

if "eval_submitted" not in st.session_state:
    st.session_state.eval_submitted = False

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
    st.session_state["levels"] = ["Junior", "Confirmed", "Senior"]

if "positions" not in st.session_state:
    st.session_state["positions"] = ["Data Viz", "Data Eng", "Data Scientist", "DBA"]

if "edited_df" not in st.session_state:
    st.warning(
        " It seems that you didn't modify or save your own coefficient setting. Default template will be used.",
        icon="⚠️",
    )
    df = st.session_state["default_coeff"]
else:
    df = st.session_state["edited_df"]


if "Final_Evaluation" in st.session_state:
    subdf = st.session_state["Final_Evaluation"]
else:
    subdf = df[df.columns[:2]]
    subdf[st.session_state["levels"]] = np.full(
        (len(subdf), len(st.session_state["levels"])), False
    )
    st.session_state["empty_eval"] = subdf

columns = subdf.columns
edited_evaluation = st.data_editor(
    subdf,
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


cols = st.columns([2, 1, 1, 1, 2])
with cols[1]:
    if st.button("Save Evaluation"):
        st.session_state.clicked_save_eval = True

with cols[3]:
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

        st.session_state.eval_submitted = False
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
        # st.dataframe(final_df.set_index("Sujet"))

    st.session_state.clicked_save_eval = False


if (
    st.session_state["rerun_editing_eval"]
    == st.session_state["current_rerun_editing_eval"] + 1
):
    st.session_state["current_rerun_editing_eval"] += 1
    st.info("Values have been reset.", icon="ℹ️")


##### GET RESULTS ######
import plotly.express as px

# Sample data for the radar chart


def show_final_df(df):
    st.dataframe(df.set_index("Sujet"))


st.markdown(
    """
    <style>
    
    #button-after {
        display: none;
    }
    .element-container:has(#button-after) {
        display: none;
    }
    .element-container:has(#button-after) + div button {
        background-color: green;
        color: white;
        border-radius: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)
# st.button("button1")
st.markdown('<span id="button-after"></span>', unsafe_allow_html=True)
colss = st.columns([2, 1, 1, 1, 2])
with colss[2]:

    if st.button("Submit and get results"):
        st.session_state.eval_submitted = True

# st.button("button2")


if st.session_state.eval_submitted:
    if "Final_df" not in st.session_state:
        st.warning(
            " Evaluation not saved, please fill the interactive DataFrame above.",
            icon="⚠️",
        )
        st.session_state.eval_submitted = False
    else:
        st.html(
            '<div style="text-align: center; font-size: larger;"> Summary Evaluation Dataframe: </div>'
        )
        show_final_df(st.session_state["Final_df"])

        with st.container():
            # data = {
            #     "Category": ["Data Viz", "Data Eng", "Data Scientist", "DBA"],
            #     "Value": [4, 3, 2, 5, 4],
            # }
            dff = st.session_state["Final_df"]

            max_bonus = {"Junior": junior, "Confirmed": confirmed, "Senior": senior}

            # Compute maximum points for each position
            summary_df = pd.DataFrame(
                columns=["Position", "Max Points", "Current Points", "Percentage Match"]
            )
            summary_df["Position"] = st.session_state.positions
            summary_df["Max Points"] = list(
                df[st.session_state.positions].sum(axis=0) * max(max_bonus.values())
            )

            result_df = dff[["Sujet"]].copy()
            for position in ["Data Viz", "Data Eng", "Data Scientist", "DBA"]:
                result_df[position] = dff[position] * (
                    dff["Junior"] * max_bonus["Junior"]
                    + dff["Confirmed"] * max_bonus["Confirmed"]
                    + dff["Senior"] * max_bonus["Senior"]
                )

            summary_df["Current Points"] = list(result_df.sum(axis=0)[1:])

            summary_df["Percentage Match"] = (
                summary_df["Current Points"] / summary_df["Max Points"]
            ) * 100

            st.dataframe(summary_df)

            # Create radar chart using Plotly Express
            fig = px.line_polar(
                summary_df,
                r="Percentage Match",
                theta="Position",
                line_close=True,
                range_r=(0, 100),
            )
            fig.update_traces(fill="toself")

            plot = st.plotly_chart(fig, use_container_width=True)

        # st.session_state.eval_submitted = False
