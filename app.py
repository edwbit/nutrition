import streamlit as st
from typing import Generator
from groq import Groq

# Set up the page configuration
st.set_page_config(page_icon="🚀", layout="centered", page_title="Groq Nutribot")

# Add Sidebar Menu
st.sidebar.title("Groq Nutribot")  # App name
st.sidebar.caption("App created by AI")
api_key = st.sidebar.text_input("Enter your API key and press Enter", type="password")

if st.sidebar.button("New Chat"):
    st.session_state.messages = []  # Clear the chat history

# Initialize the Groq client with the provided API key
client = Groq(api_key=api_key)

st.subheader("Nutribot", divider="rainbow", anchor="false")

# Initialize chat history and selected model
if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

# Define model details
models = {
    "openai/gpt-oss-120b": {"name": "openai/gpt-oss-120b", "tokens": 	
32,766},
    "moonshotai/kimi-k2-instruct": {"name": "moonshotai/kimi-k2-instruct", "tokens": 	
16,384},
    "qwen/qwen3-32b": {"name": "qwen/qwen3-32b", "tokens": 40960},
    "meta-llama/llama-4-maverick-17b-128e-instruct" : {"name": "meta-llama/llama-4-maverick-17b-128e-instruct", "tokens": 8192},
}

# Layout for model selection and max token slider
model_option = st.selectbox(
    "Choose a model:",
    options=list(models.keys()),
    format_func=lambda x: models[x]["name"],
    index=0
)

# Detect model change and clear chat history if the model has changed
if st.session_state.selected_model != model_option:
    st.session_state.messages = []
    st.session_state.selected_model = model_option

max_tokens_range = models[model_option]["tokens"]

max_tokens = st.slider(
    "Max Tokens:",
    min_value=1024,
    max_value=max_tokens_range,
    value=min(8192, max_tokens_range),
    step=1024,
    help=f"Adjust the maximum number of tokens (words) for the model's response. Max for selected model: {max_tokens_range}"
)

# Display chat messages from history in a scrollable container if there are messages
if st.session_state.messages:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for message in st.session_state.messages:
        avatar = '👨‍⚕️' if message["role"] == "assistant" else '🦲'
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.write("No chat history yet. Start a conversation by typing a message.")

# Function to generate chat responses
def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

# Handle new chat input
if prompt := st.chat_input("Type the vegetable or fruit name"):
    #task requirements
    task_requirements = f""" 
    A. Provide the SHOrt History and Scientific Name
    B. Nutritional Facts Table with description
    C. Table of Medicinal Properties
    D. Table of Diseases it can prevent or cure
    E. Glycemic index compared with white rice. 
    F. Best time to take
    G. PH level (acidic, neutral, alkaline)
    H. Preparation and consumption
    I. Contraindications of {prompt}"""

    st.session_state.messages.append({"role": "user", "content": f"{task_requirements}"})

    with st.chat_message("user", avatar='🦲'):
        st.markdown(prompt)

    try:
        chat_completion = client.chat.completions.create(
            model=model_option,
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            max_tokens=max_tokens,
            stream=True,
           
        )
        with st.chat_message("assistant", avatar="👨‍⚕️"):
            chat_responses_generator = generate_chat_responses(chat_completion)
            full_response = st.write_stream(chat_responses_generator)
    
        if isinstance(full_response, str):
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        else:
            # Handle case where full_response is an iterable (list/generator)
            combined_response = "\n".join(str(item) for item in full_response)
            st.session_state.messages.append({"role": "assistant", "content": combined_response})
    except Exception as e:
        st.error(e, icon="🚨")
