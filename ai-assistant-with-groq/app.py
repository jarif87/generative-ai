import streamlit as st
import requests
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure page settings
st.set_page_config(
    page_title="AI Chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
        .stApp {
            background-color: #f5f5f5;
        }
        .main {
            padding: 2rem;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 20px;
            padding: 0.5rem 2rem;
            border: none;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }
        .stButton>button:hover {
            background-color: #45a049;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        div.stTextInput>div>div>input {
            border-radius: 15px;
            padding: 1rem;
        }
        .chat-message {
            padding: 1.5rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .user-message {
            background-color: #E3F2FD;
        }
        .bot-message {
            background-color: white;
        }
        .sidebar .stButton>button {
            width: 100%;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'processing' not in st.session_state:
    st.session_state.processing = False

## Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Enhanced Q&A Chatbot With GroQ"
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

def generate_response(question, temperature, max_tokens):
    api_url = "https://api.groq.com/openai/v1/chat/completions"
    
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        return "Error: GroQ API key not found in environment variables."
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Include full conversation history for context
    messages = [{"role": "system", "content": "You are a helpful AI assistant."}]
    for msg in st.session_state.chat_history:
        messages.append({
            "role": "user" if msg["role"] == "user" else "assistant",
            "content": msg["content"]
        })
    messages.append({"role": "user", "content": question})
    
    payload = {
        "model": "mixtral-8x7b-32768",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    try:
        response = requests.post(api_url, json=payload, headers=headers)
        response.raise_for_status()
        
        result = response.json()
        answer = result.get("choices", [{}])[0].get("message", {}).get("content", 
                "Sorry, I couldn't generate an answer.")
        return answer
    
    except requests.exceptions.RequestException as e:
        return f"Error: Failed to communicate with GroQ API: {str(e)}"
    except ValueError as e:
        return f"Error: Failed to parse API response: {str(e)}"

def handle_user_input():
    if st.session_state.user_input and not st.session_state.processing:
        user_message = st.session_state.user_input
        st.session_state.processing = True
        
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_message})
        
        # Generate and add assistant response
        with st.spinner('Thinking...'):
            response = generate_response(user_message, st.session_state.temperature, st.session_state.max_tokens)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Clear input and reset processing flag
        st.session_state.user_input = ""
        st.session_state.processing = False
        st.experimental_rerun()

# Sidebar configuration
with st.sidebar:
    st.markdown("## Model Settings")
    st.markdown("---")
    
    st.session_state.model = st.selectbox(
        "Select Model",
        ["deepseek-r1-distill-llama-70b", "llama-3.3-70b-versatile", "llama3-8b-8192","llama-guard-3-8b"],
        help="Choose the AI model for your conversation"
    )
    
    st.session_state.temperature = st.slider(
        "Creativity Level",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        help="Higher values make the output more creative but less focused"
    )
    
    st.session_state.max_tokens = st.slider(
        "Maximum Response Length",
        min_value=50,
        max_value=500,
        value=150,
        help="Maximum number of tokens in the response"
    )
    
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.experimental_rerun()

# Main chat interface
st.title("🤖 AI Chat Assistant")
st.markdown("### Your Personal AI Assistant")

# Chat display area
chat_container = st.container()
with chat_container:
    for message in st.session_state.chat_history:
        role = message["role"]
        content = message["content"]
        
        if role == "user":
            st.markdown(f"""
                <div class="chat-message user-message">
                    <b>You:</b> {content}
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="chat-message bot-message">
                    <b>Assistant:</b> {content}
                </div>
            """, unsafe_allow_html=True)

# User input area
st.text_input(
    "Type your message here...",
    key="user_input",
    placeholder="Ask me anything...",
    on_change=handle_user_input,
    label_visibility="collapsed"
)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <small>Powered by GroQ AI • Built with Streamlit</small>
    </div>
    """,
    unsafe_allow_html=True
)