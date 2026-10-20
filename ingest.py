import hashlib
import os

from langchain_community.document_loaders import CSVLoader, PyPDFLoader, TextLoader
from langchain_google_community import GoogleDriveLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHROMA_DB_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_DIR,
    GDRIVE_CREDENTIALS_PATH,
    GDRIVE_SCOPES,
    GDRIVE_TOKEN_PATH,
)
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
                elif file.lower().endswith('.csv'):
                    # One document per row, as "column: value" lines
                    loader = CSVLoader(file_path, autodetect_encoding=True)
                elif file.lower().endswith(('.txt', '.md')):
                    loader = TextLoader(file_path, autodetect_encoding=True)
                else:
                    print(f"Skipping unsupported file type: {file}")
                    continue
                documents.extend(loader.load())
            except Exception as e:
                print(f"Error loading {file}: {e}")
                errors.append(f"{file}: {e}")
    return documents, errors

def get_gdrive_credentials():
    """Returns OAuth credentials for Google Drive, opening a browser
    consent window when there is no valid token.json."""
    from google.auth.exceptions import RefreshError
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    creds = None
    if os.path.exists(GDRIVE_TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(GDRIVE_TOKEN_PATH)
        # A token issued for other scopes (e.g. drive.file) can't read the folder
        if not creds.has_scopes(GDRIVE_SCOPES):
            creds = None
    if creds and creds.valid:
        return creds

    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
        except RefreshError:
            creds = None
    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file(GDRIVE_CREDENTIALS_PATH, GDRIVE_SCOPES)
        creds = flow.run_local_server(port=0)
    with open(GDRIVE_TOKEN_PATH, "w") as token:
        token.write(creds.to_json())
    return creds

def ingest_gdrive(folder_id=None, document_ids=None):
    """Loads files from Google Drive."""
    if not os.path.exists(GDRIVE_CREDENTIALS_PATH):
        print("Google Drive credentials.json not found. Skipping Google Drive ingestion.")
        return []
        
    print("Loading files from Google Drive...")
    # Passing credentials directly bypasses the loader's own auth logic, which
    # otherwise needs GOOGLE_APPLICATION_CREDENTIALS set to use the OAuth flow
    loader = GoogleDriveLoader(
        folder_id=folder_id,
        document_ids=document_ids,
        credentials=get_gdrive_credentials(),
        credentials_path=GDRIVE_CREDENTIALS_PATH,
        token_path=GDRIVE_TOKEN_PATH,
        scopes=GDRIVE_SCOPES,
        recursive=False
    )
    documents = loader.load()
    return documents

def chunk_id(chunk):
    """Stable ID for a chunk so re-ingesting the same content is a no-op."""
    key = "|".join([
        str(chunk.metadata.get("source", "")),
        str(chunk.metadata.get("page", "")),
        str(chunk.metadata.get("row", "")),
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
