import os
from config import (
    DATA_DIR, CHROMA_DB_DIR, CHUNK_SIZE, CHUNK_OVERLAP, 
    GDRIVE_CREDENTIALS_PATH, GDRIVE_TOKEN_PATH, EMBEDDING_MODEL_NAME
)
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_google_community import GoogleDriveLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

def ingest_local_files():
    """Loads and splits local files from the data directory."""
    print(f"Loading local files from {DATA_DIR}...")
    documents = []
    for root, _, files in os.walk(DATA_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                if file.lower().endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                elif file.lower().endswith(('.txt', '.md', '.csv')):
                    loader = TextLoader(file_path, autodetect_encoding=True)
                else:
                    print(f"Skipping unsupported file type: {file}")
                    continue
                documents.extend(loader.load())
            except Exception as e:
                print(f"Error loading {file}: {e}")
    return documents

def ingest_gdrive(folder_id=None, document_ids=None):
    """Loads files from Google Drive."""
    if not os.path.exists(GDRIVE_CREDENTIALS_PATH):
        print("Google Drive credentials.json not found. Skipping Google Drive ingestion.")
        return []
        
    print("Loading files from Google Drive...")
    loader = GoogleDriveLoader(
        folder_id=folder_id,
        document_ids=document_ids,
        credentials_path=GDRIVE_CREDENTIALS_PATH,
        token_path=GDRIVE_TOKEN_PATH,
        recursive=False
    )
    documents = loader.load()
    return documents

def chunk_and_store(documents):
    if not documents:
        print("No documents to store.")
        return
        
    print(f"Splitting {len(documents)} documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(documents)
    
    print(f"Storing {len(chunks)} chunks in ChromaDB at {CHROMA_DB_DIR}...")
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR
    )
    print("Ingestion complete.")
    return vectorstore

def main(use_local=True, gdrive_folder_id=None):
    docs = []
    if use_local:
        docs.extend(ingest_local_files())
        
    if gdrive_folder_id:
        docs.extend(ingest_gdrive(folder_id=gdrive_folder_id))
        
    chunk_and_store(docs)

if __name__ == "__main__":
    # Example usage for CLI ingestion:
    main(use_local=True)
