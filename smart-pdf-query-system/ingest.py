import os
import logging
from langchain.document_loaders import PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_vector_store():
    # Ensure the directory exists
    persist_directory = "./chroma_db"
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)
        logger.info(f"Created directory '{persist_directory}'")

    documents = []
    docs_dir = "docs"
    if not os.path.exists(docs_dir):
        logger.error(f"The directory '{docs_dir}' does not exist.")
        return
    
    for root, dirs, files in os.walk(docs_dir):
        for file in files:
            if file.endswith(".pdf"):
                file_path = os.path.join(root, file)
                logger.info(f"Loading document: {file_path}")
                try:
                    loader = PDFMinerLoader(file_path)
                    loaded_docs = loader.load()
                    if loaded_docs:
                        logger.info(f"Loaded {len(loaded_docs)} documents from {file_path}")
                        documents.extend(loaded_docs)
                    else:
                        logger.warning(f"No documents loaded from {file_path}")
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")

    if not documents:
        logger.error("No documents were loaded. Check the 'docs' directory and file paths.")
        return

    logger.info(f"Loaded {len(documents)} documents.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    logger.info(f"Created {len(texts)} text chunks.")

    if not texts:
        logger.error("No text chunks created. Check the text splitting process.")
        return

    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        logger.info("Embeddings initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize embeddings: {e}")
        return

    try:
        vector_store = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)
        vector_store.persist()
        logger.info(f"Created Chroma vector store with {len(texts)} vectors.")
    except Exception as e:
        logger.error(f"Failed to create Chroma vector store: {e}")

if __name__ == "__main__":
    create_vector_store()

