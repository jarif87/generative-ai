import os
import logging
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Configure logging
logging.basicConfig(level=logging.DEBUG)

def load_vector_store():
    # Ensure the directory exists
    persist_directory = "./chroma_db"
    if not os.path.exists(persist_directory):
        logging.error(f"The directory '{persist_directory}' does not exist. Please run the ingestion script.")
        st.error(f"The directory '{persist_directory}' does not exist. Please run the ingestion script.")
        return None

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    return vector_store

def load_llm():
    checkpoint = "LaMini-T5-738M"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    pipe = pipeline(
        'text2text-generation',
        model=model,
        tokenizer=tokenizer,
        max_length=1024,  # Increase max_length to allow longer output
        do_sample=True,
        temperature=0.3,  # Adjust temperature to control randomness
        top_p=0.9         # Adjust top_p to control diversity of the output
    )
    return HuggingFacePipeline(pipeline=pipe)

def process_answer(question):
    try:
        vector_store = load_vector_store()
        if vector_store is None:
            return "Vector store not found. Please run the ingestion script.", {}

        llm = load_llm()
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(),
            return_source_documents=True
        )
        result = qa.invoke(question)
        answer = result['result']
        return answer, result
    except Exception as e:
        logging.error(f"An error occurred while processing the answer: {e}")
        st.error(f"An error occurred while processing the answer: {e}")
        return "An error occurred while processing your request.", {}

def main():
    st.title("Search Your PDF 📚📝")
    with st.expander("About the App"):
        st.markdown(
            """
            This is a Generative AI-powered Question and Answering app that responds to questions about your PDF File.
            """
        )
    question = st.text_area("Enter your Question")
    if st.button("Ask"):
        st.info("Your Question: " + question)
        st.info("Your Answer")
        try:
            answer, metadata = process_answer(question)
            st.write(answer)
            st.write(metadata)
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    main()

