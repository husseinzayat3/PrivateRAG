import csv
import os
import shutil
from typing import NamedTuple

from langchain_core.documents import Document
from langchain_google_community import GoogleDriveLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from config import (
    CHROMA_DB_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_DIR,
    GDRIVE_CREDENTIALS_PATH,
    GDRIVE_SCOPES,
    GDRIVE_TOKEN_PATH,
    SAMPLES_DIR,
)
from vectorstore import BATCH_SIZE, chunk_id, delete_chunks, get_vectorstore, ids_for_sources, mark_changed

# Tried in order when reading text files; latin-1 accepts any byte sequence
TEXT_ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")


class IngestResult(NamedTuple):
    added: int  # new chunks embedded and stored
    removed: int  # outdated chunks of changed documents


def read_text(path):
    """Reads a text file, trying common encodings."""
    with open(path, "rb") as f:
        data = f.read()
    for encoding in TEXT_ENCODINGS:
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue

def load_text(path):
    return [Document(page_content=read_text(path), metadata={"source": path})]

def load_pdf(path_or_file, source=None):
    """One document per page; `page` is 0-based, `page_label` is the printed page number."""
    reader = PdfReader(path_or_file)
    labels = reader.page_labels
    meta = {"source": source} if source else {}
    return [
        Document(
            page_content=page.extract_text(),
            metadata={**meta, "page": i, "page_label": labels[i], "total_pages": len(reader.pages)},
        )
        for i, page in enumerate(reader.pages)
    ]

class DrivePdfLoader:
    """Reads PDFs downloaded by GoogleDriveLoader, which then sets their source and title."""

    def __init__(self, file):
        self.file = file

    def load(self):
        return load_pdf(self.file)

def load_csv(path):
    """One document per row, as "column: value" lines."""
    reader = csv.DictReader(read_text(path).splitlines())
    docs = []
    for i, row in enumerate(reader):
        lines = []
        for column, value in row.items():
            # Cells beyond the header row end up in a list under the None column
            if isinstance(value, list):
                value = ",".join(v.strip() for v in value)
            lines.append(f"{column.strip() if column is not None else column}: "
                         f"{value.strip() if isinstance(value, str) else value}")
        docs.append(Document(page_content="\n".join(lines), metadata={"source": path, "row": i}))
    return docs

LOADERS = {".pdf": lambda path: load_pdf(path, source=path), ".csv": load_csv, ".txt": load_text, ".md": load_text}

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
            loader = LOADERS.get(os.path.splitext(file)[1].lower())
            if loader is None:
                if file != ".gitkeep":
                    print(f"Skipping unsupported file type: {file}")
                continue
            try:
                documents.extend(loader(file_path))
            except Exception as e:
                print(f"Error loading {file}: {e}")
                errors.append(f"{file}: {e}")
    return documents, errors

def copy_samples():
    """Copies the sample documents into the data directory.

    Returns the number of files copied; files already there are left alone.
    """
    copied = 0
    for name in sorted(os.listdir(SAMPLES_DIR)):
        target = os.path.join(DATA_DIR, name)
        # samples/README.md describes the samples and isn't one of them
        if name == "README.md" or not name.endswith(".md") or os.path.exists(target):
            continue
        shutil.copyfile(os.path.join(SAMPLES_DIR, name), target)
        copied += 1
    return copied

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
        # Google Docs, Sheets and Slides are exported as text; other files (only
        # PDFs, given file_types) go through file_loader_cls
        file_types=["document", "sheet", "presentation", "pdf"] if folder_id else None,
        file_loader_cls=DrivePdfLoader,
        recursive=False
    )
    documents = loader.load()
    return documents

def chunk_and_store(documents):
    """Splits documents and stores chunks not already in ChromaDB.

    Documents are always loaded whole, so any chunk of the same source that isn't
    in the new set is from an older version of the file and is removed.
    """
    if not documents:
        print("No documents to store.")
        return IngestResult(0, 0)

    print(f"Splitting {len(documents)} documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(documents)

    # Dedupe within this batch, then compare with what's indexed for the same sources
    unique = {chunk_id(c): c for c in chunks}
    existing = set(ids_for_sources({str(d.metadata.get("source", "")) for d in documents}))
    new_ids = [i for i in unique if i not in existing]
    outdated = existing - unique.keys()

    print(f"Storing {len(new_ids)} new chunks in ChromaDB at {CHROMA_DB_DIR}, removing "
          f"{len(outdated)} outdated ({len(existing) - len(outdated)} unchanged)...")
    delete_chunks(outdated)
    vectorstore = get_vectorstore()
    for i in range(0, len(new_ids), BATCH_SIZE):
        batch = new_ids[i:i + BATCH_SIZE]
        vectorstore.add_documents([unique[id_] for id_ in batch], ids=batch)
    if new_ids:
        mark_changed()
    print("Ingestion complete.")
    return IngestResult(len(new_ids), len(outdated))

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
