import streamlit as st
from langchain_groq.chat_models import ChatGroq
from dotenv import load_dotenv
import os
import pandas as pd
from pandasai import Agent
from pandasai import SmartDatalake, SmartDataframe
from pandasai.responses.response_parser import ResponseParser
import numpy as np
import re

load_dotenv()

try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except KeyError:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "Oups")


with st.sidebar:
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
        background-color: red;
        color: white;
        border-radius: 20px;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )
    st.markdown('<span id="button-after"></span>', unsafe_allow_html=True)
    reset_chat = st.button("Clear Chat History")

    models = {
        "gemma-7b-it": {"name": "Gemma-7b-it", "tokens": 8192, "developer": "Google"},
        "llama2-70b-4096": {
            "name": "LLaMA2-70b-chat",
            "tokens": 4096,
            "developer": "Meta",
        },
        "llama3-70b-8192": {
            "name": "LLaMA3-70b-8192",
            "tokens": 8192,
            "developer": "Meta",
        },
        "llama3-8b-8192": {
            "name": "LLaMA3-8b-8192",
            "tokens": 8192,
            "developer": "Meta",
        },
        "mixtral-8x7b-32768": {
            "name": "Mixtral-8x7b-Instruct-v0.1",
            "tokens": 32768,
            "developer": "Mistral",
        },
    }

    if "selected_model" not in st.session_state:
        st.session_state.selected_model = None

    if "selected_model" in st.session_state and st.session_state.selected_model != None:
        def_index = list(models.keys()).index(st.session_state.selected_model)
    else:
        def_index = 4

    model_option = st.selectbox(
        "Choose a model:",
        options=list(models.keys()),
        format_func=lambda x: models[x]["name"],
        index=def_index,
    )
    max_tokens_range = models[model_option]["tokens"]
    max_tokens = st.slider(
        "Max Tokens:",
        min_value=512,  # Minimum value to allow some flexibility
        max_value=max_tokens_range,
        # Default value or max allowed if less
        value=min(32768, max_tokens_range),
        step=512,
        help=f"Adjust the maximum number of tokens (words) for the model's response. Max for selected model: {max_tokens_range}",
    )

    temperature = st.slider(
        "Temperature:",
        min_value=0.0,  # Minimum value to allow some flexibility
        max_value=1.0,
        # Default value or max allowed if less
        value=0.5,
        step=0.05,
        help=f"Adjust the Temperature for the model's response.",
    )


def extract_value_from_result(code_block):
    """
    Extracts the value from the 'result' line in the code block,
    removing any lines containing potentially dangerous commands.
    """
    # Define a regular expression pattern to match dangerous commands
    dangerous_commands = [
        "pip install",
        "rm",
        "del",
        "shutdown",
        "import os",
        "os.system",
    ]
    dangerous_pattern = "|".join(re.escape(cmd) for cmd in dangerous_commands)
    dangerous_pattern = rf"\b({dangerous_pattern})\b"

    # Remove lines containing dangerous commands
    cleaned_code_block = re.sub(dangerous_pattern, "", code_block)

    # Define a regular expression pattern to match the result line
    pattern = r"result\s*=\s*{'type':\s*'[^']*',\s*'value':\s*(.*)}"

    # Use re.sub() to replace the result line with just the value content
    cleaned_code_block = re.sub(pattern, r"\1", cleaned_code_block)

    return cleaned_code_block


if "Final_df" not in st.session_state or "summary_df" not in st.session_state:
    st.warning(
        "You haven't carried out an assessment or have forgotten to submit it. This is a mandatory step to use your AI assistant :",
        icon="⚠️",
    )
    st.session_state.eval_submitted = False
    if st.button("Perform the assessment here!"):
        st.switch_page("pages/02_Assessment.py")

else:

    class StreamlitCallback:
        def __init__(self, container) -> None:
            """Initialize callback handler."""
            self.container = container

        def on_code(self, response: str):
            self.container.code(response)

    class StreamlitResponse(ResponseParser):
        def __init__(self, context) -> None:
            super().__init__(context)

        def format_dataframe(self, result):
            st.dataframe(result["value"])
            return

        def format_plot(self, result):
            st.image(result["value"])
            return

        def format_other(self, result):
            st.write(result["value"])
            return

    st.text("")
    st.markdown(
        """  
        This is an entertainment purpose : you can use any available LLM model from Groq to discuss about the assessment results.  
        This feature is an AI agent built using [_PandasAI_](https://docs.pandas-ai.com/en/latest/).
        """
    )
    st.markdown("***")
    global_df = st.session_state["Final_df"]
    summary_df = st.session_state["summary_df"]
    dfs = [global_df, summary_df]

    if "pandasai_messages" not in st.session_state:
        st.session_state.pandasai_messages = []

    llm = ChatGroq(
        model_name=model_option,
        api_key=GROQ_API_KEY,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    agent = SmartDatalake(
        [global_df, summary_df],
        # memory_size=10,
        config={
            "llm": llm,
            "response_parser": StreamlitResponse,
            "verbose": True,
            "save_charts": False,
            "open_charts": False,
        },
        # description="""
        #     You are working with two pandas dataframe in Python named 'global_df' and 'summary_df'.
        #     'global_df' shows the global result with the topics (first column), the Data positions (Data Viz, Data Engineer, Data Scientist and DBA),
        #     and finally the level for a given subject among Junior, Confirmed and Senior.
        #     'summary_df' shows the summary for each position, including 'Max Points', 'Current Points', 'Percentage Match'.
        #     """,
    )

    query = st.chat_input("Ask me any question about the dataframe")

    for message in st.session_state.pandasai_messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant_code":
                st.write(f"```{message['content']}")
                try:
                    exec(message["content"])
                except Exception as e:
                    st.warning(
                        "Something went wrong when trying to execute the code provided by the AI Assistant:",
                        icon="⚠️",
                    )
                    st.write(e)
            elif message["role"] == "user":
                st.markdown(message["content"])

            else:
                st.code(message["content"])

    if query:

        with st.chat_message("user"):
            st.markdown(query)

        st.session_state.pandasai_messages.append({"role": "user", "content": query})

        with st.chat_message("assistant"):
            with st.spinner("Generating reply...."):
                answer = agent.chat(query)

                st.write(answer)
                st.code(answer)

            st.session_state.pandasai_messages.append(
                {"role": "assistant", "content": agent.last_code_executed}
            )

        # container.code(agent.last_code_executed)
        # container.write(answer)
    # st.session_state.pandasai_messages

    if reset_chat or st.session_state.selected_model != model_option:
        st.session_state.pandasai_messages = []

        for message in st.session_state.pandasai_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        message_placeholder = st.empty()
        st.session_state.selected_model = model_option
        st.rerun()
        # st.session_state.selected_model = model_option
