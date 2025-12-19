import streamlit as st
import os
import time
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv
from langchain.prompts.chat import ChatPromptTemplate

# Page configuration
st.set_page_config(
    page_title="Research Paper Q&A",
    page_icon="📚",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        background-color: #f5f7f9;
    }
    .main {
        padding: 2rem;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        border: none;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        margin: 1rem auto;
        display: block;
        width: auto;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .stTextInput>div>div>input {
        border-radius: 0.5rem;
        border: 2px solid #e0e0e0;
        padding: 0.75rem;
    }
    .stExpander {
        background-color: white;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    h1 {
        color: #2c3e50;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 2rem;
        text-align: center;
    }
    .response-container {
        background-color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin: 1rem 0;
    }
    .metrics-container {
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin: 1rem 0;
    }
    .search-section {
        max-width: 800px;
        margin: 0 auto;
    }
    </style>
""", unsafe_allow_html=True)

# Load environment variables
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize LLM and prompt
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
prompt = ChatPromptTemplate.from_template("""
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    </context>
    Question:{input}
""")

# Function to create vector embeddings and load documents
def create_vector_embedding():
    with st.spinner("Creating vector embeddings... Please wait."):
        if "vectors" not in st.session_state:
            st.session_state.embeddings = OpenAIEmbeddings()
            st.session_state.loader = PyPDFDirectoryLoader("research_papers")
            st.session_state.docs = st.session_state.loader.load()
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=10000, 
                chunk_overlap=200
            )
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(
                st.session_state.docs[:50]
            )
            st.session_state.vectors = FAISS.from_documents(
                st.session_state.final_documents, 
                st.session_state.embeddings
            )
        st.success("✅ Vector Database is ready!")

# App header
st.title("📚 Research Paper Q&A Assistant")

# App description
st.markdown("""
    <div style='text-align: center; color: #666; margin-bottom: 2rem;'>
        Ask questions about your research papers and get accurate answers powered by Groq and Llama3
    </div>
""", unsafe_allow_html=True)

# Search section with centered layout
with st.container():
    # Center-aligned container for search and button
    st.markdown('<div class="search-section">', unsafe_allow_html=True)
    
    # User input
    user_prompt = st.text_input(
        "What would you like to know about the research papers?",
        placeholder="Enter your question here...",
        key="user_input"
    )
    
    # Centered initialize button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("📥 Initialize Document Database"):
            create_vector_embedding()
            
    st.markdown('</div>', unsafe_allow_html=True)

# Process query and display results
if user_prompt and "vectors" in st.session_state:
    with st.spinner("🤔 Analyzing your question..."):
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        start = time.process_time()
        response = retrieval_chain.invoke({'input': user_prompt})
        end = time.process_time()
        
        # Display response in a container
        st.markdown("<div class='response-container'>", unsafe_allow_html=True)
        st.markdown("### 📝 Answer")
        st.write(response['answer'])
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Display metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Response Time", f"{(end - start):.2f} seconds")
        with col2:
            st.metric("Documents Analyzed", len(response['context']))
        
        # Show document similarity search in an expandable section
        with st.expander("📑 Related Document Excerpts"):
            for i, doc in enumerate(response['context']):
                st.markdown(f"""
                    <div style='background-color: #f8f9fa; padding: 1rem; 
                    border-radius: 0.5rem; margin-bottom: 1rem;'>
                        <strong>Document {i + 1}</strong><br>
                        {doc.page_content}
                    </div>
                """, unsafe_allow_html=True)
elif user_prompt and "vectors" not in st.session_state:
    st.warning("⚠️ Please initialize the document database first!")

# Footer
st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem 0; margin-top: 2rem; 
    border-top: 1px solid #eee;'>
        Powered by Groq and Llama3 🚀
    </div>
""", unsafe_allow_html=True)