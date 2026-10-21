import hashlib
import os
import re
from collections import Counter
from functools import cache

from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import CHROMA_DB_DIR, EMBEDDING_MODEL_NAME, EMBEDDING_PROVIDER, OLLAMA_BASE_URL

# Max IDs per Chroma get/add/delete call, to stay under Chroma's batch size limit
BATCH_SIZE = 1000

# Incremented whenever this process changes the index, so in-memory caches
# built from it (like the BM25 keyword index) know to rebuild
_version = 0


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
        # Chroma sends anonymous usage events by default
        client_settings=Settings(anonymized_telemetry=False),
        embedding_function=get_embeddings(),
        collection_metadata={
            "embedding_provider": EMBEDDING_PROVIDER,
            "embedding_model": EMBEDDING_MODEL_NAME,
        }
    )

def chunk_id(chunk):
    """Stable ID for a chunk so re-ingesting the same content is a no-op."""
    key = "|".join([
        str(chunk.metadata.get("source", "")),
        str(chunk.metadata.get("page", "")),
        str(chunk.metadata.get("row", "")),
        chunk.page_content,
    ])
    return hashlib.sha256(key.encode("utf-8")).hexdigest()

def document_name(metadata):
    """Display name of a document: its title (Google Drive) or file name."""
    return metadata.get("title") or os.path.basename(str(metadata.get("source", "unknown")))

def mark_changed():
    global _version
    _version += 1

def index_version():
    """Changes whenever the index changes: in this process, or (by chunk count) in another."""
    return _version, get_vectorstore()._collection.count()

def _get_all(include):
    """Pages through the whole collection; yields (id, document, metadata)."""
    store = get_vectorstore()
    offset = 0
    while True:
        batch = store.get(include=include, limit=BATCH_SIZE, offset=offset)
        ids = batch["ids"]
        documents = batch.get("documents") or [None] * len(ids)
        metadatas = batch.get("metadatas") or [None] * len(ids)
        yield from zip(ids, documents, metadatas, strict=True)
        if len(ids) < BATCH_SIZE:
            return
        offset += BATCH_SIZE

def all_chunks():
    """Every chunk in the index, as Documents with their IDs."""
    return [
        Document(id=id_, page_content=text, metadata=meta or {})
        for id_, text, meta in _get_all(["documents", "metadatas"])
    ]

def ids_for_sources(sources):
    """IDs of all chunks that came from the given sources (file paths or Drive URLs)."""
    sources = list(sources)
    store = get_vectorstore()
    ids = []
    for i in range(0, len(sources), BATCH_SIZE):
        batch = sources[i:i + BATCH_SIZE]
        where = {"source": batch[0]} if len(batch) == 1 else {"source": {"$in": batch}}
        ids.extend(store.get(where=where, include=[])["ids"])
    return ids

def delete_chunks(ids):
    ids = list(ids)
    store = get_vectorstore()
    for i in range(0, len(ids), BATCH_SIZE):
        store.delete(ids=ids[i:i + BATCH_SIZE])
    if ids:
        mark_changed()

def list_documents():
    """The indexed documents, as [{"source", "name", "chunks"}] sorted by name."""
    chunks = Counter()
    names = {}
    for _, _, meta in _get_all(["metadatas"]):
        source = str((meta or {}).get("source", "unknown"))
        chunks[source] += 1
        names.setdefault(source, document_name(meta or {}))
    return sorted(
        ({"source": s, "name": names[s], "chunks": n} for s, n in chunks.items()),
        key=lambda d: (d["name"].lower(), d["source"]),
    )

def delete_documents(sources):
    """Removes every chunk of the given sources from the index. Returns the number removed."""
    ids = ids_for_sources(sources)
    delete_chunks(ids)
    return len(ids)

def reset_index():
    """Deletes everything in the current collection."""
    get_vectorstore().reset_collection()
    mark_changed()
