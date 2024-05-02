import streamlit as st
import pandas as pd
import numpy as np
import os
from io import StringIO

st.set_page_config(
    page_title="Evaluation",
    page_icon=":lower_left_fountain_pen:",
    layout="wide",
)

if "use_uploaded_file" not in st.session_state:
    st.session_state.use_uploaded_file = False
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
    st.session_state["levels_bonus"] = {
        "no_experience": 0,
        "junior": junior,
        "confirmed": confirmed,
        "senior": senior,
    }
    st.markdown("##")
    target_level = st.selectbox(
        "**What is the desired level for the candidate?**",
        [k.title().replace("_", " ") for k in st.session_state["levels_bonus"].keys()],
        index=2,
    )
    st.markdown("##")

    uploaded_file = st.file_uploader("**Upload a previous assessment**")
    if uploaded_file is not None:
        # To read file as bytes:
        bytes_data = uploaded_file.getvalue()
        # st.write(bytes_data)

        # To convert to a string based IO:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        # st.write(stringio)

        # To read file as string:
        string_data = stringio.read()
        # st.write(string_data)

        # Can be used wherever a "file-like" object is accepted:
        uploaded_df = pd.read_csv(uploaded_file, sep=None, index_col=0)
        st.session_state["uploaded_file"] = uploaded_df
        if st.button("Use as data file"):
            st.session_state.eval_submitted = False
            if "Final_Evaluation" in st.session_state:
                del st.session_state["Final_Evaluation"]
            st.session_state.use_uploaded_file = True

            # del st.session_state["Final_Evaluation"]
            # st.session_state["rerun_editing_eval"] += 1

    else:
        if "uploaded_file" in st.session_state:
            del st.session_state["uploaded_file"]


st.markdown("# Candidate Assessment")

st.write(
    """
Here you'll evaluate your candidate based on the topics which appear in the coefficient table.
In order to be as unbiased as possible, the coefficient won't be visible in this step of the process.  
If you alread have a previous saved assessment, you can upload it (see sidebar) and re-use these data as starting point.  
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
    st.session_state["levels"] = ["No Experience", "Junior", "Confirmed", "Senior"]

if "positions" not in st.session_state or "skills_group" not in st.session_state:
    st.session_state["positions"] = [
        "Data Viz",
        "Data Engineer",
        "Data Scientist",
        "DBA",
    ]
    st.session_state["skills_group"] = ["Methods/Culture", "English"]

if "uploaded_file" in st.session_state:
    df = st.session_state.uploaded_file[
        [
            k
            for k in st.session_state.uploaded_file
            if k not in st.session_state["levels"]
        ]
    ]
elif "edited_df" not in st.session_state:
    st.warning(
        " It seems that you didn't modify or save your own coefficient setting. Default template will be used.",
        icon="⚠️",
    )
    df = st.session_state["default_coeff"]
else:
    df = st.session_state["edited_df"]


if "Final_Evaluation" in st.session_state:
    subdf = st.session_state["Final_Evaluation"]

elif "uploaded_file" in st.session_state and st.session_state.use_uploaded_file:
    subdf = st.session_state["uploaded_file"][
        [
            k
            for k in st.session_state.uploaded_file
            if k not in st.session_state["positions"]
        ]
    ]


else:
    subdf = df[df.columns[:2]]

    subdf[st.session_state["levels"]] = np.full(
        (len(subdf), len(st.session_state["levels"])), False
    )

    st.session_state["empty_eval"] = subdf
#
columns = subdf.columns
edited_evaluation = st.data_editor(
    subdf,
    column_config={
        "_index": st.column_config.TextColumn(disabled=True),
        columns[0]: st.column_config.TextColumn(disabled=True),
        columns[1]: st.column_config.TextColumn(disabled=True),
        # columns[2]: st.column_config.TextColumn(disabled=True),
        # columns[3]: st.column_config.TextColumn(disabled=True),
        "No Experience": st.column_config.CheckboxColumn(
            None,
            help="Is this topic unknown to the candidate?",
        ),
        "Junior": st.column_config.CheckboxColumn(
            None,
            help="Is the candidate Junior?",
        ),
        "Confirmed": st.column_config.CheckboxColumn(
            None,
            help="Is the candidate Confirmed?",
        ),
        "Senior": st.column_config.CheckboxColumn(
            None,
            help="Is the candidate Senior?",
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
        st.session_state.use_uploaded_file = False
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

    is_valid = check_unique_true(edited_evaluation[columns[2:]])

    if not is_valid:
        # st.session_state["trigger_invalid_data"] = True
        st.error(
            " Invalid Data, there should be 1 level selected for each topic.",
            icon="🚨",
        )

    else:
        st.session_state["Final_Evaluation"] = edited_evaluation
        st.info("Coefficient saved.", icon="ℹ️")
        final_df = df.reset_index().merge(edited_evaluation.reset_index())
        st.session_state["Final_df"] = final_df
        # st.dataframe(final_df.set_index("Topic"))

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
    st.dataframe(df.set_index("Topic"))


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
        st.markdown("##")

        st.markdown(
            '<h3  style="text-align: center; "> Summary Assessment Dataframe </h3>',
            unsafe_allow_html=True,
        )
        st.markdown("##")
        col_res = st.columns([7, 1])
        with col_res[0]:
            show_final_df(st.session_state["Final_df"])
        with col_res[1]:
            st.write(
                """<style>
            [data-testid="stHorizontalBlock"] {
                align-items: center;
            }
            </style>
            """,
                unsafe_allow_html=True,
            )

            @st.cache_data
            def convert_df(df):
                return df.to_csv(index=False, sep=";").encode("utf-8")

            main_csv = convert_df(st.session_state["Final_df"])
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
                border: 2px solid green;
            }
            .element-container:has(#button-after) + div button:hover {
                border-color: #ff4b4b; /* Border color on hover */
                
            }
                }
            </style>
            """,
                unsafe_allow_html=True,
            )
            st.markdown('<span id="button-after"></span>', unsafe_allow_html=True)
            st.download_button(
                "Download as CSV",
                main_csv,
                "main_result.csv",
                "text/csv",
                key="download-csv",
            )

        with st.container():
            # data = {
            #     "Category": ["Data Viz", "Data Eng", "Data Scientist", "DBA"],
            #     "Value": [4, 3, 2, 5, 4],
            # }
            dff = st.session_state["Final_df"]

            max_bonus = {
                "No Experience": 0,
                "Junior": junior,
                "Confirmed": confirmed,
                "Senior": senior,
            }

            # Compute maximum points for each position
            summary_df = pd.DataFrame(
                columns=[
                    "Qualification",
                    "Max Points",
                    "Current Points",
                    "Percentage Match",
                ]
            )
            summary_df["Qualification"] = (
                st.session_state.positions + st.session_state.skills_group
            )

            # summary_df["Max Points"] = list(
            #     dff[st.session_state.positions].sum(axis=0) * (max_bonus[target_level])
            # )

            summary_df["Max Points"] = (
                list(
                    dff[dff["Category"] == "Technical"][st.session_state.positions].sum(
                        axis=0
                    )
                    * (max_bonus[target_level])
                )
                + [
                    len(dff[~dff["Category"].isin(["Technical", "English"])])
                    * (max_bonus[target_level])
                ]
                + [len(dff[dff["Category"] == "English"]) * (max_bonus[target_level])]
            )

            # (df['Points'] * df['Experience Level'].map(max_bonus)).sum()

            result_df = dff[["Topic"]].copy()
            for position in ["Data Viz", "Data Engineer", "Data Scientist", "DBA"]:
                result_df[position] = dff[dff["Category"] == "Technical"][position] * (
                    dff["No Experience"] * max_bonus["No Experience"]
                    + dff["Junior"] * max_bonus["Junior"]
                    + dff["Confirmed"] * max_bonus["Confirmed"]
                    + dff["Senior"] * max_bonus["Senior"]
                )

            tmp_table_others = (
                dff[~dff["Category"].isin(["Technical", "English"])][columns[-4:]]
                .sum(axis=0)
                .reset_index()
            )

            tmp_table_eng = (
                dff[dff["Category"] == "English"][columns[-4:]]
                .sum(axis=0)
                .reset_index()
            )

            summary_df["Current Points"] = (
                list(result_df.sum(axis=0)[1:])
                + [
                    (
                        tmp_table_others[0] * tmp_table_others["index"].map(max_bonus)
                    ).sum()
                ]
                + [(tmp_table_eng[0] * tmp_table_eng["index"].map(max_bonus)).sum()]
            )

            if target_level != "No Experience":
                summary_df["Percentage Match"] = [
                    min([100, k])
                    for k in (summary_df["Current Points"] / summary_df["Max Points"])
                    * 100
                ]

            else:
                summary_df["Percentage Match"] = [100] * len(summary_df)
            st.session_state["summary_df"] = summary_df

            st.markdown("##")
            st.markdown("##")

            st.markdown(
                """
                        ### Basic analysis
                        
                        Below you'll find the main result based on the previously filled assessment:  
                        The dataframe on the left represents the match ratio for each
                        relevant Qualification.  
                        This percentage level is computed regarding the maximum obtainable points and the obtained points.  
                        The first value depends of both the desired level for the candidate, which will set the upper bond, and the multiplicator bonus for each level. 
                        These parameters can be customized in the sidebar.  
                        It's important to notate that the Job position results are computed only regarding the technical topics. 
                        The other ones (for instance English or Methods/Culture) are separately analyzed.  
                        On the right you'll see a radar chart highlighting the Percentage Match for each Qualification.

                        """
            )
            col1, col2 = st.columns(2)

            with col1:

                st.dataframe(summary_df)

            with col2:

                # Create radar chart using Plotly Express
                fig = px.line_polar(
                    summary_df,
                    r="Percentage Match",
                    theta="Qualification",
                    line_close=True,
                    range_r=(0, max([100, summary_df["Percentage Match"].max()])),
                    markers=dict(color="royalblue", symbol="square", size=80),
                )
                # fig.update_traces(fill="toself")
                fig.update_layout(
                    title="Percentage Match for each Qualification (%)",
                    title_x=0.25,
                    title_y=0.05,
                    font_size=15,
                    showlegend=True,
                    polar=dict(
                        # bgcolor = "rgb(223, 223, 223)",
                        bgcolor="#e9ecf2",
                        angularaxis=dict(
                            linewidth=2,
                            showline=True,
                            linecolor="black",
                            gridcolor="grey",
                            thetaunit="radians",
                            color="#ff4b4b",
                        ),
                        radialaxis=dict(
                            side="counterclockwise",
                            showline=True,
                            linewidth=1,
                            gridcolor="grey",
                            gridwidth=1,
                            angle=0,
                            color="#ff4b4b",
                            dtick=25,
                            tickfont=dict(
                                size=18,
                            ),
                        ),
                    ),
                    # paper_bgcolor="#ffffff",
                )
                fig.update_traces(
                    line_color="#ff4b4b",
                    line_width=5,
                    marker=dict(  # Customize marker properties
                        symbol="circle",  # Marker shape
                        size=10,  # Marker size
                        # color="lightcoral",  # Marker color
                    ),
                )
                plot = st.plotly_chart(fig, use_container_width=True)

            # @st.cache_data

        # st.session_state.eval_submitted = False
