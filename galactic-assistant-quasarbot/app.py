import streamlit as st
import replicate
import os
from dotenv import load_dotenv
from reportlab.pdfgen import canvas
import tempfile

# Load environment variables from .env file
load_dotenv()

# App title
st.set_page_config(page_title="🤖💬 Galactic Assistant QuasarBot")

# Replicate Credentials
with st.sidebar:
    st.title('🤖💬 Galactic Assistant: QuasarBot')
    st.write('This chatbot is created using the open-source Llama 3 LLM model from Meta.')
    replicate_api = os.getenv('REPLICATE_API_TOKEN')
    if replicate_api:
        st.success('API key already provided!', icon='✅')
    else:
        replicate_api = st.text_input('Enter Replicate API token:', type='password')
        if not (replicate_api.startswith('r8_') and len(replicate_api) == 40):
            st.warning('Please enter your credentials!', icon='⚠️')
        else:
            st.success('Proceed to entering your prompt message!', icon='👉')
    os.environ['REPLICATE_API_TOKEN'] = replicate_api

    st.subheader('Models and parameters')
    selected_model = st.sidebar.selectbox('Choose a Llama3 model', ['Llama3-70B'], key='selected_model')
    if selected_model == 'Llama3-70B':
        llm = 'meta/meta-llama-3-70b-instruct'
    temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=1.0, value=0.6, step=0.01)
    top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    max_length = st.sidebar.slider('max_length', min_value=32, max_value=512, value=512, step=8)
    st.markdown('📖 Learn how to build this app in this [blog](https://replicate.com/meta/meta-llama-3-70b-instruct)!')

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# Function for generating LLaMA3 response
def generate_llama3_response(prompt_input):
    string_dialogue = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"
    response = replicate.run(
        llm, 
        input={
            "prompt": f"{string_dialogue} {prompt_input} Assistant: ",
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_length,
            "presence_penalty": 1.15,
            "frequency_penalty": 0.2
        }
    )
    return response

# User-provided prompt
if prompt := st.chat_input(disabled=not replicate_api):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            response = generate_llama3_response(prompt)
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)

# Function to convert messages to HTML
def messages_to_html(messages):
    html = "<html><body>"
    for message in messages:
        role = "User" if message["role"] == "user" else "Assistant"
        html += f"<p><strong>{role}:</strong> {message['content']}</p>"
    html += "</body></html>"
    return html

# Function to generate PDF report
def generate_pdf(description):
    if description is None:
        return None

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
        c = canvas.Canvas(tmpfile.name)
        c.setFont("Helvetica", 12)
        text_lines = description.split('\n')  # Ensure description is not None before splitting
        y = 750
        for line in text_lines:
            c.drawString(100, y, line)
            y -= 20
        c.save()
        tmpfile.close()
        return tmpfile.name

# Button to download chat history as PDF
if st.sidebar.button('Download Chat History as PDF'):
    html_content = messages_to_html(st.session_state.messages)
    pdf_filename = generate_pdf(html_content)
    if pdf_filename:
        with open(pdf_filename, "rb") as pdf_file:
            st.sidebar.download_button(
                label="Download PDF",
                data=pdf_file,
                file_name="chat_history.pdf",
                mime="application/pdf"
            )
