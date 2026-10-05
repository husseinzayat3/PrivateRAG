import os

# Suppress HuggingFace tokenizers warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)

# Chunking Config
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Models Config
# Using a lightweight local HuggingFace embedding model
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" 
# Ollama LLM Config (Make sure Ollama is running locally with this model)
OLLAMA_MODEL_NAME = "llama3" # Make sure to run `ollama run llama3` locally
OLLAMA_BASE_URL = "http://localhost:11434"

# Google Drive Config
# Requires a credentials.json file in the root directory for OAuth
GDRIVE_CREDENTIALS_PATH = os.path.join(BASE_DIR, "credentials.json")
GDRIVE_TOKEN_PATH = os.path.join(BASE_DIR, "token.json")
