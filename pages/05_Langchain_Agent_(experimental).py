import streamlit as st
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_groq.chat_models import ChatGroq
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import MessagesPlaceholder
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from dotenv import load_dotenv
import os
import pandas as pd
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


if "Final_df" not in st.session_state or "summary_df" not in st.session_state:
    st.warning(
        "You haven't carried out an assessment or have forgotten to submit it. This is a mandatory step to use your AI assistant :",
        icon="⚠️",
    )
    st.session_state.eval_submitted = False
    if st.button("Perform the assessment here!"):
        st.switch_page("pages/02_Assessment.py")

else:
    st.text("")
    st.markdown(
        """  
        This is an entertainment purpose : you can use any available LLM model from Groq to discuss about the assessment results.  
    This feature is basically an AI agent built using [_langchain_](https://python.langchain.com/docs/modules/agents/), in particular [_langchain_experimental_](%s).
        """
        % "https://api.python.langchain.com/en/latest/experimental_api_reference.html"
    )
    st.markdown("***")
    df = st.session_state["Final_df"]
    df1 = st.session_state["summary_df"]

    llm = ChatGroq(
        model_name=model_option,
        api_key=GROQ_API_KEY,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    ########## TRY PANDAS AI ###############

    ###########################################

    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history")

    if "df_messages" not in st.session_state:
        st.session_state.df_messages = []

    msgs = StreamlitChatMessageHistory()
    # if "memory_test" not in st.session_state:

    st.session_state.memory_test = ConversationBufferWindowMemory(
        memory_key="chat_history", chat_memory=msgs, k=10, return_messages=True
    )
    agent_kwargs = {
        "extra_prompt_messages": [MessagesPlaceholder(variable_name="chat_history")],
    }

    PREFIX = f"""
    You are working with two pandas dataframe in Python named 'df' and 'df1'.
    'df' shows the global result with the topics (first column), the Data positions (Data Viz, Data Engineer, Data Scientist and DBA),
    and finally the level for a given subject among Junior, Confirmed and Senior. 
    'df1' shows the summary for each position, including 'Max Points', 'Current Points', 'Percentage Match'.
    You should use the conversation history below to answer the question posed of you:

    History of the whole conversation:
    {msgs}
    Always import streamlit as st first.
    Always use st.write() instead of print() to display a global result.
    Always use st.table() to display a dataframe/table.
    Never use the `set_page_config()` method from streamlit.
    Never execute any code, instead provide it following these instructions:
    Generate the code <code> for plotting the previous data in plotly,
    in the format requested. The solution should be given using plotly
    and only plotly. Do not use matplotlib. Be careful about the name used for the dataframe (df != df1).
    Return the code <code> in the following
    format ```python <code>```
    """

    FORMAT_INSTRUCTIONS = """
    You should use the tools below to answer the question posed of you:

    python_repl_ast: A Python shell. Use this to execute python commands. Input should be a valid python command. When using this tool, sometimes output is abbreviated - make sure it does not look abbreviated before using it in your answer.

    ALWAYS Use the following format in the respected order:

    Question: the input question you must answer
    Thought: you should always think about what to do, what action to take
    Action: the tool name, should be one of [python_repl_ast]
    Action Input: the input to the action, never add backticks "`" around the action input. Always use st.write() instead of print() to display a global result.
    Always use st.table() to display a dataframe/table.
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: MUST BE : I now know the final answer
    Final Answer: the final answer to the original input question. 
    """
    PythonAstREPLTool_init = PythonAstREPLTool.__init__

    def PythonAstREPLTool_init_wrapper(self, *args, **kwargs):
        PythonAstREPLTool_init(self, *args, **kwargs)
        self.globals = self.locals

    PythonAstREPLTool.__init__ = PythonAstREPLTool_init_wrapper

    agent = create_pandas_dataframe_agent(
        llm,
        [df, df1],
        prefix=PREFIX,
        verbose=True,
        agent_executor_kwargs={
            "handle_parsing_errors": True,
            "extra_prompt_messages": [
                MessagesPlaceholder(variable_name="chat_history")
            ],
            "memory": st.session_state.memory_test,
            "format_instructions": FORMAT_INSTRUCTIONS,
        },
        max_iterations=8,
        # include_df_in_prompt=True,
        number_of_head_rows=-1,
        # memory=st.session_state.memory_test,
        # st.session_state.memory,
    )

    # code extracted from the src.llm_utils module
    def extract_python_code(text):
        pattern = r"```python\s(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        if not matches:
            return None
        else:
            return matches[0]

    question = st.chat_input("Ask me any question about the dataframe")

    for message in st.session_state.df_messages:
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
            else:
                st.markdown(message["content"])

    if question:

        with st.chat_message("user"):
            st.markdown(question)

        st.session_state.df_messages.append({"role": "user", "content": question})

        with st.chat_message("assistant"):

            st_callback = StreamlitCallbackHandler(st.container())
            message_placeholder = st.empty()
            full_response = ""

            agent_response = agent.run(
                question,
                callbacks=[st_callback],
            )

            st.markdown(agent_response)

            for chunk in agent_response.split():
                full_response += chunk + " "

            message_placeholder.markdown(full_response + "▌")

            message_placeholder.info(full_response)

            st.session_state.df_messages.append(
                {"role": "assistant", "content": full_response}
            )

            code = extract_python_code(agent_response)
            if code != None:
                code.replace("streamlit", "st")
                code.replace("<code>", "").replace("</code>", "")
                if "fig.show()" in code:
                    code = code.replace("fig.show()", "")
                    code += """st.plotly_chart(fig, theme='streamlit', use_container_width=True)"""
                if "print(" in code:
                    code = code.replace("print(", "st.write(")

                st.write(f"```{code}")

                try:
                    exec(code)
                    st.session_state.df_messages.append(
                        {"role": "assistant_code", "content": code}
                    )
                except Exception as e:
                    st.warning(
                        "Something went wrong when trying to execute the code provided by the AI Assistant:",
                        icon="⚠️",
                    )
                    st.write(e)
        st.session_state.memory.save_context(
            {"input": question},
            {"outputs": full_response.replace("{", "{{{").replace("}", "}}}")},
        )
        msgs.messages[-1].content = (
            msgs.messages[-1].content.replace("{", "{{{").replace("}", "}}}")
        )
        # st.session_state.memory_test.save_context(
        #     {"input": question}, {"outputs": full_response}
        # )

        # st.session_state.memory

    # Show me something relevant about the dataframe
    # msgs
    if reset_chat or st.session_state.selected_model != model_option:
        st.session_state.df_messages = []
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history")
        msgs.clear()
        st.session_state.memory_test = ConversationBufferWindowMemory(
            memory_key="chat_history", chat_memory=msgs, k=10, return_messages=True
        )
        for message in st.session_state.df_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        message_placeholder = st.empty()
        st.session_state.selected_model = model_option
        st.rerun()
        # st.session_state.selected_model = model_option
