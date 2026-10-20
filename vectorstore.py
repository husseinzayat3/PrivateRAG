import re
from functools import cache

from langchain_chroma import Chroma

from config import CHROMA_DB_DIR, EMBEDDING_MODEL_NAME, EMBEDDING_PROVIDER, OLLAMA_BASE_URL


# Cached so the embedding model and Chroma client are loaded once per process
# and shared between ingestion and retrieval.
@cache
def get_embeddings():
    if EMBEDDING_PROVIDER == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model=EMBEDDING_MODEL_NAME, base_url=OLLAMA_BASE_URL)
    if EMBEDDING_PROVIDER == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            encode_kwargs={"normalize_embeddings": True}
        )
    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {EMBEDDING_PROVIDER!r}")

def collection_name():
    """One collection per embedding model, since vectors from different
    models can't be mixed. Switching models starts a fresh (empty) index."""
    name = re.sub(r"[^a-zA-Z0-9._-]", "-", f"{EMBEDDING_PROVIDER}-{EMBEDDING_MODEL_NAME}")
    return name.strip("-._")[:512]

@cache
def get_vectorstore():
    return Chroma(
        collection_name=collection_name(),
        persist_directory=CHROMA_DB_DIR,
        embedding_function=get_embeddings(),
        collection_metadata={
            "embedding_provider": EMBEDDING_PROVIDER,
            "embedding_model": EMBEDDING_MODEL_NAME,
        }
    )
