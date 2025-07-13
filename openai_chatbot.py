import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Simple Q&A Chatbot With OPENAI"

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user's queries."),
        ("user", "Question:{question}")
    ]
)

def generate_response(question, api_key, engine, temperature, max_tokens):
    openai.api_key = api_key
    llm = ChatOpenAI(model=engine)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({'question': question})
    return answer

st.set_page_config(page_title="Enhanced Q&A Chatbot", page_icon="🤖", layout="wide")
st.title("Enhanced Q&A Chatbot with OpenAI")

st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password", help="Enter your OpenAI API key here to enable the chatbot.")
engine = st.sidebar.selectbox("Select OpenAI Model", ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo","gpt-4-turbo-preview"], help="Choose the model you want to interact with.")
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, help="Adjust the creativity of responses.")
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150, help="Set the maximum length of the response.")

st.write("Go ahead and ask any question to the AI chatbot:")
user_input = st.text_input("You:")

if user_input and api_key:
    response = generate_response(user_input, api_key, engine, temperature, max_tokens)
    st.markdown(f"**Response:** {response}", unsafe_allow_html=True)
elif user_input:
    st.warning("Please enter the OpenAI API Key in the sidebar to get a response.")
else:
    st.write("Please enter a question to start the conversation.")
