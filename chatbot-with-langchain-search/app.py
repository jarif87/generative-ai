import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv

# Load API keys from environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Initialize Search Tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

search = DuckDuckGoSearchRun(name="Search")

# --------------------------- Streamlit UI Setup ---------------------------

# Page Configuration
st.set_page_config(page_title="🔎 LangChain Search Assistant", page_icon="🔍", layout="wide")

# Custom Styling
st.markdown(
    """
    <style>
        body { background-color: #f0f2f6; }
        .stChatMessage { border-radius: 12px; padding: 10px; margin-bottom: 10px; }
        .stChatMessage-user { background-color: #4a90e2; color: white; }
        .stChatMessage-assistant { background-color: #f8f9fa; color: black; }
        .stSpinner { font-size: 18px; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True
)

# Title & Description
st.title("🔎 LangChain - Chat with Search")
st.markdown(
    "This chatbot can **search the web, retrieve articles from Arxiv, Wikipedia**, and more.\n\n"
    "💡 Try **asking about recent discoveries, technical concepts, or general knowledge!**"
)

# --------------------------- Chat Memory ---------------------------

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi! I'm a chatbot that can search the web. How can I help you?"}
    ]

# Display previous messages
for msg in st.session_state.messages:
    role = "🧑‍💻 User" if msg["role"] == "user" else "🤖 Assistant"
    st.chat_message(msg["role"]).markdown(f"**{role}**: {msg['content']}")

# --------------------------- Chat Input & Processing ---------------------------

# User Input
if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(f"**🧑‍💻 User**: {prompt}")

    # Initialize LLM
    llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.3-70b-versatile", streaming=True)
    tools = [search, arxiv, wiki]

    search_agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True
    )

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        with st.spinner("🔍 Searching..."):
            response = search_agent.run(st.session_state.messages, callbacks=[st_cb])

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.markdown(f"**🤖 Assistant**: {response}")