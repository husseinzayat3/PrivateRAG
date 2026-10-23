import os

from dotenv import load_dotenv

# Release version (semver). Bump it together with CHANGELOG.md; the release
# workflow checks that a pushed tag v<VERSION> matches.
VERSION = "0.1.0"

# Suppress HuggingFace tokenizers warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Settings can be overridden with environment variables or a .env file
# in the project root (see .env.example)
load_dotenv(os.path.join(BASE_DIR, ".env"))

# Opt out of usage telemetry from HuggingFace Hub (Chroma's and Streamlit's are
# disabled in vectorstore.py and .streamlit/config.toml)
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def _path(env_var, default):
    """A path from the environment; relative paths are resolved against the project root."""
    return os.path.join(BASE_DIR, os.path.expanduser(os.getenv(env_var) or default))


# Paths
DATA_DIR = _path("DATA_DIR", "data")
CHROMA_DB_DIR = _path("CHROMA_DB_DIR", "chroma_db")
CHAT_DB_PATH = _path("CHAT_DB_PATH", "chat_history.db")
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.dirname(CHAT_DB_PATH), exist_ok=True)

# Chunking Config
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# Embeddings Config
# "huggingface" runs the model in-process via sentence-transformers;
# "ollama" uses an Ollama embedding model (e.g. nomic-embed-text).
# Each provider/model pair gets its own Chroma collection, so changing
# these requires re-ingesting your documents.
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "huggingface")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")

# Ollama LLM Config (Make sure Ollama is running locally with this model)
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "llama3.1")  # `ollama pull llama3.1`
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
# Context window in tokens. Ollama silently truncates prompts longer than this,
# so it must fit the retrieved passages plus the chat history.
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
# How long Ollama keeps the model in memory after a request; reloading it
# adds several seconds before the first token
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")

# Retrieval Config
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "6"))
# "similarity" (vectors only), "mmr" (vectors, preferring diverse passages) or
# "hybrid" (BM25 keyword search fused with vectors; best for exact names and IDs)
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "hybrid")
# Rerank the retrieved passages with a cross-encoder: better ordering, but a model
# download on first use and some extra time per question
USE_RERANKER = os.getenv("USE_RERANKER", "false").lower() in ("1", "true", "yes")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2")
# Number of previous chat messages used to understand follow-up questions
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "6"))

# Google Drive Config
# Requires an OAuth credentials.json file (in the project root by default)
GDRIVE_CREDENTIALS_PATH = _path("GDRIVE_CREDENTIALS_PATH", "credentials.json")
GDRIVE_TOKEN_PATH = _path("GDRIVE_TOKEN_PATH", "token.json")
GDRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
