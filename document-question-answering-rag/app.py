import os
import json
import boto3
import streamlit as st
from langchain_community.embeddings import BedrockEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.vectorstores import FAISS
from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from langchain_community.chat_models import BedrockChat


bedrock=boto3.client("bedrock-runtime",region_name="us-east-1")
bedrock_embedding=BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0",client=bedrock)

# data ingestion

def data_ingestion():
    loader=PyPDFDirectoryLoader("pdf")
    documents=loader.load()
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=1000)
    docs=text_splitter.split_documents(documents)
    return docs

# vector embedding and vector store

def vector_store(docs):
    vectorstore_faiss=FAISS.from_documents(docs,bedrock_embedding)
    vectorstore_faiss.save_local("faiss_index")
    
# create Claude Model

def get_claude_model():
    llm=BedrockChat(model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",client=bedrock,model_kwargs={"max_tokens":512})
    return llm

# create LLma Model

def get_llama_model():
    llm=Bedrock(model_id="meta.llama3-8b-instruct-v1:0",client=bedrock,model_kwargs={"max_gen_len":512})
    return llm


prompt_template = """

Human: Use the following pieces of context to provide a 
concise answer to the question at the end but usse atleast summarize with 
250 words with detailed explaantions. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context

Question: {question}

Assistant:"""


PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

def get_response_llm(llm,vectorstore_faiss,query):
    qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore_faiss.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    ),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)
    answer=qa({"query":query})
    return answer['result']

def main():
    st.set_page_config("Chat PDF")
    
    st.header("Chat with PDF using AWS Bedrock💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    with st.sidebar:
        st.title("Update Or Create Vector Store:")
        
        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                vector_store(docs)
                st.success("Done")

    if st.button("Claude3 Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embedding,allow_dangerous_deserialization=True)
            llm=get_claude_model()
            
            #faiss_index = get_vector_store(docs)
            st.write(get_response_llm(llm,faiss_index,user_question))
            st.success("Done")

    if st.button("Llama3 Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embedding,allow_dangerous_deserialization=True)
            llm=get_llama_model()
            
            #faiss_index = get_vector_store(docs)
            st.write(get_response_llm(llm,faiss_index,user_question))
            st.success("Done")

if __name__ == "__main__":
    main()


