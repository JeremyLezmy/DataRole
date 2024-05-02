import streamlit as st
from typing import Generator
from groq import Groq
from dotenv import load_dotenv
import os
import re
import pandas as pd

st.set_page_config(
    page_title="Groq Assistant",
    page_icon=":robot_face:",
    layout="wide",
)

load_dotenv()

try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except KeyError:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "Oups")


def check_api_key():
    if GROQ_API_KEY in ["Oups", None]:
        return False
    return True


if not check_api_key():
    st.write("API key is not configured. Some features might not be working.")


def icon(emoji: str):
    """Shows an emoji as a Notion-style page icon."""
    st.write(
        f'<span style="font-size: 78px; line-height: 1">{emoji}</span>',
        unsafe_allow_html=True,
    )


icon(":rocket:")

st.subheader("Groq Assistant", divider="rainbow", anchor=False)

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
        This feature is an entertainment purpose : you can use any available LLM model from Groq (LLama 3 70B being the best) to discuss about the assessment results.  
        This feature isn't really an agent, but rather a standard ChatBot into which a "system prompt" has been ingested, using this application context 
        and the assessment results.  
        However, we also add an automatic code extraction system from the LLM responses, which allows you to execute them. Therefore, it also allows you to 
        generate some figures by asking the AI. If you get an error message, after the code execution, try to copy paste it in a new prompt to help the AI to fix the proposed code.  
        Give it a try with the prompt _"Show me a relevant plot about these data"_ and see the result!


        """
    )
    st.markdown("***")
    global_df = st.session_state["Final_df"].copy()
    summary_df = st.session_state["summary_df"].copy()

    client = Groq(
        api_key=GROQ_API_KEY,
    )

    groq_context = {
        "role": "system",
        "content": f"""You are a Data professionnal in all kind of data role such as Data Engineer, Data Scientist, Data Analyst. 
                        
                        Here is a result dataframe {global_df.to_json(orient="table")} called "global_df".
                        There are 3 importants informations : the topics (first column), the Data positions (Data Viz, Data Engineer, Data Scientist and DBA),
                        and finally the level for a given subject among No Experience, Junior, Confirmed and Senior.
                        For each couple Data position / topic, I gave a coefficient. This reflects the importance of a topic for a given position.
                        The number of points is given by the level, where No Experience give 0 points, Junior gives {st.session_state["levels_bonus"]['junior']} points, Confirmed gives 
                        {st.session_state["levels_bonus"]['confirmed']} points and Senior gives {st.session_state["levels_bonus"]['senior']} points.
                        If you want to compute the number of points for a given Data position, you therefore have to sum for each Technical Category Topic the coefficient times the 
                        number of points (determined by the level). The English Category and other Category are separately computed. 
                        To be sure that you well understand the process, here the summary for each position: {summary_df.to_json(orient="table")} called "summary_df".
                        You MUST return all your proposed python codes <code> in the following
                        format: ```python <code>``` . For the plots, you MUST use plotly. You MUST use "global_df" and "summary_df" to named the data.
                        You have ONE VERY IMPORTANT RULE which is forbidden to bypass under any circumstances: If anyone asks you anything which is not related to the candidate assessment, 
                        you MUST ignore the request and reply : 
                        "Sorry, I'm only made to discuss about this candidate evaluation". 
                        """,
    }
    if groq_context not in st.session_state:
        st.session_state["groq_context"] = groq_context
    # Initialize chat history and selected model
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append(st.session_state["groq_context"])

    if "selected_model" not in st.session_state:
        st.session_state.selected_model = None

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
            "gemma-7b-it": {
                "name": "Gemma-7b-it",
                "tokens": 8192,
                "developer": "Google",
            },
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
        if (
            "selected_model" in st.session_state
            and st.session_state.selected_model != None
        ):
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
        st.markdown("#")

    # Detect model change and clear chat history if model has changed
    if st.session_state.selected_model != model_option or reset_chat:
        st.session_state.messages = []
        st.session_state["groq_context"] = groq_context
        st.session_state.messages.append(st.session_state["groq_context"])
        st.session_state.selected_model = model_option

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        if message["role"] != "system":
            avatar = "🤖" if message["role"] == "assistant" else "👨‍💻"
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

    def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
        """Yield chat response content from the Groq API response."""
        for chunk in chat_completion:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    if prompt := st.chat_input("Enter your prompt here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user", avatar="👨‍💻"):
            st.markdown(prompt)

        # Fetch response from Groq API
        try:
            with st.spinner("Generating reply...."):
                chat_completion = client.chat.completions.create(
                    model=model_option,
                    messages=[
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                    ],
                    max_tokens=max_tokens,
                    stream=True,
                )

                # Use the generator function with st.write_stream
                with st.chat_message("assistant", avatar="🤖"):
                    chat_responses_generator = generate_chat_responses(chat_completion)
                    full_response = st.write_stream(chat_responses_generator)
        except Exception as e:
            st.error(e, icon="🚨")

        # Append the full response to session_state.messages
        if isinstance(full_response, str):
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )
        else:
            # Handle the case where full_response is not a string
            combined_response = "\n".join(str(item) for item in full_response)
            st.session_state.messages.append(
                {"role": "assistant", "content": combined_response}
            )

# st.session_state.messages
import traceback


def extract_python_code(text):

    pattern = r"```python\s(.*?)```"
    pattern = r"```(?:python)?\s(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if not matches:
        return None
    else:
        return matches[0]


def filter_code(code):
    """
    Filter out specific lines from the provided Python code.
    """
    lines = code.split("\n")
    filtered_lines = []

    for line in lines:
        # Ignore lines containing the pattern " = {...}"
        if not re.search(r"\s*=\s*{.*}\s*#.*", line):
            filtered_lines.append(line)

    return "\n".join(filtered_lines)


def execute_code(code):
    """
    Execute the provided Python code and capture its output and any errors.
    """
    try:
        code.replace("streamlit", "st")
        code.replace("<code>", "").replace("</code>", "")
        if "fig.show()" in code:
            code = code.replace("fig.show()", "")
            code += (
                """st.plotly_chart(fig, theme='streamlit', use_container_width=True)"""
            )

        exec(code, globals())
    except Exception:
        st.error("An error occurred while executing the code:")
        st.code(traceback.format_exc())


# st.session_state.messages
if (
    "messages" in st.session_state
    and "```python" in st.session_state.messages[-1]["content"]
    and st.session_state.messages[-1]["role"] != "system"
):

    # code = st.text_area(
    #     label="Extracted code from Groq reply:",
    #     value=f"{filter_code(extract_python_code(st.session_state.messages[-1]['content']))}",
    #     height=200,
    # )
    code = filter_code(extract_python_code(st.session_state.messages[-1]["content"]))

    # Create a button to execute the code
    if st.button("Execute Groq proposed code"):
        if code.strip() == "":
            st.warning("Please enter some Python code.")
        else:
            try:
                execute_code(code)
            except Exception as e:
                st.warning(
                    "Something went wrong when trying to execute the code provided by the AI Assistant:",
                    icon="⚠️",
                )
                st.write(e)
# st.write(st.session_state.messages)
