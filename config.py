import os

from dotenv import load_dotenv

# Suppress HuggingFace tokenizers warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")

# Settings can be overridden with environment variables or a .env file
# in the project root (see .env.example)
load_dotenv(os.path.join(BASE_DIR, ".env"))

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)

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

# Retrieval Config
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "6"))
# Number of previous chat messages used to understand follow-up questions
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "6"))

# Google Drive Config
# Requires a credentials.json file in the root directory for OAuth
GDRIVE_CREDENTIALS_PATH = os.path.join(BASE_DIR, "credentials.json")
GDRIVE_TOKEN_PATH = os.path.join(BASE_DIR, "token.json")
GDRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
