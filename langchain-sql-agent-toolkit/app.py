import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy import create_engine
import sqlite3
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Streamlit UI
st.set_page_config(page_title="LangChain: Chat with SQL DB", page_icon="🦜")
st.title("🦜 LangChain: Chat with SQL DB")

# Database selection
LOCALDB = "USE_LOCALDB"
MYSQL = "USE_MYSQL"
radio_opt = ["Use SQLite3 Database - student.db", "Connect to MySQL Database"]
selected_opt = st.sidebar.radio("Choose the DB you want to chat with", options=radio_opt)

if radio_opt.index(selected_opt) == 1:
    db_uri = MYSQL
    mysql_host = st.sidebar.text_input("MySQL Host")
    mysql_user = st.sidebar.text_input("MySQL User")
    mysql_password = st.sidebar.text_input("MySQL Password", type="password")
    mysql_db = st.sidebar.text_input("MySQL Database")
else:
    db_uri = LOCALDB

if not api_key:
    st.info("Please add the Groq API key in the .env file.")
    st.stop()

# LLM model setup
llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)

# Database configuration
def configure_db(db_uri, mysql_host=None, mysql_user=None, mysql_password=None, mysql_db=None):
    if db_uri == LOCALDB:
        dbfilepath = (Path(__file__).parent / "student.db").absolute()
        creator = lambda: sqlite3.connect(f"file:{dbfilepath}?mode=ro", uri=True)
        return SQLDatabase(create_engine("sqlite:///", creator=creator))
    elif db_uri == MYSQL:
        if not (mysql_host and mysql_user and mysql_password and mysql_db):
            st.error("Please provide all MySQL connection details.")
            st.stop()
        return SQLDatabase(create_engine(f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}"))

# Initialize database connection
db = configure_db(db_uri, mysql_host, mysql_user, mysql_password, mysql_db) if db_uri == MYSQL else configure_db(db_uri)

# Initialize SQL toolkit and agent
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)

# Chat session state initialization
if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# User input handling
user_query = st.chat_input("Ask anything from the database")
if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)
    
    with st.chat_message("assistant"):
        streamlit_callback = StreamlitCallbackHandler(st.container())
        try:
            response = agent.run(user_query, callbacks=[streamlit_callback])
        except Exception as e:
            response = f"An error occurred: {str(e)}"
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
