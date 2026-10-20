from functools import cache
from config import CHROMA_DB_DIR, EMBEDDING_MODEL_NAME
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Cached so the embedding model and Chroma client are loaded once per process
# and shared between ingestion and retrieval.
@cache
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

@cache
def get_vectorstore():
    return Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=get_embeddings()
    )
