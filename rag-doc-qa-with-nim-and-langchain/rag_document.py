import streamlit as st
import os
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()
os.environ['NVIDIA_API_KEY'] = os.getenv("NVIDIA_API_KEY")

# Initialize Streamlit app
st.set_page_config(page_title="Nvidia NIM Demo", layout="wide")
st.title("\U0001F916 Nvidia NIM AI Chatbot")
st.markdown("""<style>
    .stButton button {background-color: #4CAF50; color: white; padding: 10px; border-radius: 5px;}
    .stTextInput>div>div>input {border-radius: 5px;}
    .stExpander {background-color: #f9f9f9; padding: 10px; border-radius: 5px;}
</style>""", unsafe_allow_html=True)

# Sidebar for embedding documents
with st.sidebar:
    st.header("📂 Document Processing")
    if st.button("Embed Documents \U0001F4DA"):
        with st.spinner("Processing documents..."):
            if "vectors" not in st.session_state:
                st.session_state.embeddings = NVIDIAEmbeddings()
                loader = PyPDFDirectoryLoader("./pdf")
                docs = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
                final_documents = text_splitter.split_documents(docs[:30])
                st.session_state.vectors = FAISS.from_documents(final_documents, st.session_state.embeddings)
            st.success("✅ Vector Store DB is ready!")

# Load LLM
llm = ChatNVIDIA(model="meta/llama-3.2-1b-instruct")

# Prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Question: {input}
    """
)

# User input
query = st.text_input("🔍 Ask a question from documents")

if st.button("Search Query \U0001F50E"):
    if query:
        if "vectors" in st.session_state:
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            
            with st.spinner("Searching for the best response..."):
                start = time.process_time()
                response = retrieval_chain.invoke({'input': query})
                elapsed_time = time.process_time() - start
            
            st.success(f"✅ Response generated in {elapsed_time:.2f} seconds")
            st.write(response['answer'])

            # Display relevant documents
            with st.expander("📄 Document Similarity Search"):
                for i, doc in enumerate(response.get("context", [])):
                    st.markdown(f"**Chunk {i+1}:**")
                    st.info(doc.page_content)
        else:
            st.warning("⚠️ Please embed documents first!")
