import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Streamlit App Configuration
st.set_page_config(page_title="LangChain: Summarize Text", page_icon="🦜", layout="centered")

# Custom Styling
st.markdown(
    """
    <style>
        .stTextInput>div>div>input {
            font-size: 18px;
            padding: 10px;
            border-radius: 10px;
            border: 2px solid #4CAF50;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 18px;
            padding: 10px 20px;
            border-radius: 10px;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .stAlert {
            font-size: 16px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# App Title
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader("Generate concise summaries effortlessly!")

# API Key
hf_api_key = os.getenv("HF_TOKEN")

# Input for URL
generic_url = st.text_input("Enter a YouTube or Website URL", placeholder="Paste the URL here...")

# Hugging Face Model
repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=300, temperature=0.7, token=hf_api_key)

# Prompt Template
prompt_template = """
Provide a summary of the following content in 300 words:

Content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

# Summarization Button
if st.button("Summarize Content", help="Click to summarize the provided URL"):
    if not hf_api_key or not generic_url.strip():
        st.error("⚠️ Please provide a valid Hugging Face API key and a URL.")
    elif not validators.url(generic_url):
        st.error("❌ Invalid URL! Please enter a valid YouTube or website URL.")
    else:
        try:
            with st.spinner("⏳ Fetching and summarizing content..."):
                # Load Content
                if "youtube.com" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) "
                                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"}
                    )
                docs = loader.load()

                # Summarization Chain
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run(docs)

                # Display Summary
                st.success("✅ Summary Generated Successfully!")
                st.markdown(f"**Summary:**\n\n{output_summary}")

        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")
