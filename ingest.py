import hashlib
import os
from config import (
    DATA_DIR, CHROMA_DB_DIR, CHUNK_SIZE, CHUNK_OVERLAP,
    GDRIVE_CREDENTIALS_PATH, GDRIVE_TOKEN_PATH
)
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_google_community import GoogleDriveLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from vectorstore import get_vectorstore

# Max IDs per Chroma get/add call, to stay under Chroma's batch size limit
BATCH_SIZE = 1000

def ingest_local_files():
    """Loads local files from the data directory.

    Returns a tuple of (documents, errors) where errors is a list of
    human-readable messages for files that failed to load.
    """
    print(f"Loading local files from {DATA_DIR}...")
    documents = []
    errors = []
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
                errors.append(f"{file}: {e}")
    return documents, errors

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

def chunk_id(chunk):
    """Stable ID for a chunk so re-ingesting the same content is a no-op."""
    key = "|".join([
        str(chunk.metadata.get("source", "")),
        str(chunk.metadata.get("page", "")),
        chunk.page_content,
    ])
    return hashlib.sha256(key.encode("utf-8")).hexdigest()

def chunk_and_store(documents):
    """Splits documents and stores chunks not already in ChromaDB.

    Returns the number of newly added chunks.
    """
    if not documents:
        print("No documents to store.")
        return 0


    print(f"Splitting {len(documents)} documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(documents)

    # Dedupe within this batch, then skip chunks already in the store
    unique = {chunk_id(c): c for c in chunks}
    ids = list(unique)
    vectorstore = get_vectorstore()
    existing = set()
    for i in range(0, len(ids), BATCH_SIZE):
        existing.update(vectorstore.get(ids=ids[i:i + BATCH_SIZE], include=[])["ids"])
    new_ids = [i for i in ids if i not in existing]

    print(f"Storing {len(new_ids)} new chunks in ChromaDB at {CHROMA_DB_DIR} "
          f"({len(existing)} already indexed)...")
    for i in range(0, len(new_ids), BATCH_SIZE):
        batch = new_ids[i:i + BATCH_SIZE]
        vectorstore.add_documents([unique[id_] for id_ in batch], ids=batch)
    print("Ingestion complete.")
    return len(new_ids)

def main(use_local=True, gdrive_folder_id=None):
    docs = []
    if use_local:
        local_docs, _ = ingest_local_files()
        docs.extend(local_docs)
        
    if gdrive_folder_id:
        docs.extend(ingest_gdrive(folder_id=gdrive_folder_id))
        
    chunk_and_store(docs)

if __name__ == "__main__":
    # Example usage for CLI ingestion:
    main(use_local=True)
