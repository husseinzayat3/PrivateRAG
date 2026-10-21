# Troubleshooting & FAQ

- [Setup](#setup)
  - ["Error connecting to LLM or retrieving context"](#error-connecting-to-llm-or-retrieving-context)
  - ["model not found" / the model isn't pulled](#model-not-found--the-model-isnt-pulled)
  - [Docker: the app waits forever on first start](#docker-the-app-waits-forever-on-first-start)
  - [Port 8501 is already in use](#port-8501-is-already-in-use)
- [Performance](#performance)
  - [The first answer takes a long time to start](#the-first-answer-takes-a-long-time-to-start)
  - [Which hardware do I need? (Apple Silicon, NVIDIA, CPU only)](#which-hardware-do-i-need)
- [Answers and the index](#answers-and-the-index)
  - [The answer says there isn't enough information](#the-answer-says-there-isnt-enough-information)
  - [I changed the embedding model and now nothing is found](#i-changed-the-embedding-model-and-now-nothing-is-found)
  - [I edited or deleted a file, but answers still use the old version](#i-edited-or-deleted-a-file-but-answers-still-use-the-old-version)
  - [Which retrieval mode should I use?](#which-retrieval-mode-should-i-use)
- [Google Drive](#google-drive)
  - [Google says "Google hasn't verified this app"](#google-says-google-hasnt-verified-this-app)
  - [Using Google Drive with Docker](#using-google-drive-with-docker)
- [Windows](#windows)
- [Privacy: what goes over the network?](#privacy-what-goes-over-the-network)

## Setup

### "Error connecting to LLM or retrieving context"

The app can't reach Ollama. Check that it's running:

```bash
curl http://localhost:11434/api/version
```

- If that fails, start Ollama: open the Ollama app (macOS/Windows) or run `ollama serve` (Linux).
- If Ollama runs on another machine or port, set `OLLAMA_BASE_URL` in `.env`.
- With Docker Compose, the app talks to the `ollama` container, so `OLLAMA_BASE_URL` in `.env`
  is ignored. Check that container with `docker compose logs ollama`.

### "model not found" / the model isn't pulled

Pull the model named in `OLLAMA_MODEL_NAME` (default `llama3.1`), or the one typed in the
sidebar:

```bash
ollama pull llama3.1
ollama list            # shows the models you have
```

With Docker Compose, the model in `OLLAMA_MODEL_NAME` is pulled automatically on first boot. To
add another model later:

```bash
docker compose exec ollama ollama pull qwen2.5:7b
```

### Docker: the app waits forever on first start

On first start, `ollama-init` downloads the chat model (about 4.9 GB for `llama3.1`), and the
app only starts once that's done. Follow the download with:

```bash
docker compose logs -f ollama-init
```

The first question also downloads the embedding model (about 130 MB). Both are kept in Docker
volumes, so later starts are fast and work offline.

### Port 8501 is already in use

Run on another port: `streamlit run app.py --server.port 8502`. With Docker, change the port
mapping in `docker-compose.yml` to `"8502:8501"`.

## Performance

### The first answer takes a long time to start

Most of the wait before the first word is the model **reading the retrieved passages**, not
writing the answer. To make it faster:

- Lower **k** (passages to retrieve) in the sidebar, e.g. from 6 to 3.
- Lower `CHUNK_SIZE` (e.g. to 600) and re-ingest.
- Use a smaller model, e.g. `llama3.2:3b` or `qwen2.5:3b`.
- Keep the model loaded: `OLLAMA_KEEP_ALIVE` (default `30m`) controls how long Ollama keeps it
  in memory between questions. Reloading it adds several seconds.

The first question after starting the app can also be slower while the models load. The app
preloads them in the background at startup.

### Which hardware do I need?

| Hardware | What to expect |
|---|---|
| **Apple Silicon (M1 or later)** | Runs well with native Ollama, which uses the GPU through Metal. Docker on macOS **can't** use the GPU, so for speed run Ollama natively (see below). |
| **NVIDIA GPU** | Fast. Native Ollama uses CUDA automatically. With Docker, install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) and start with `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up`. |
| **CPU only** | Works, but expect 10–60 seconds before an answer starts with `llama3.1` (8B). Use a 1–3B model (`llama3.2:3b`, `qwen2.5:1.5b`) and a lower k. |

As a rule of thumb, an 8B model needs about 8 GB of free RAM (or VRAM).

**Docker app with native Ollama** (e.g. on a Mac): start only the app container and point it at
Ollama on the host. Edit `OLLAMA_BASE_URL` under `app` in `docker-compose.yml` to
`http://host.docker.internal:11434`, then run:

```bash
docker compose up --no-deps app
```

## Answers and the index

### The answer says there isn't enough information

Open **Sources** under the answer to see which passages were retrieved.

- **No sources at all:** nothing is indexed yet. Click **Ingest Documents**, and check the
  messages it shows for files that failed to load.
- **Sources are unrelated:** rephrase using words that appear in your documents, or raise k.
  Make sure the retrieval mode is **Hybrid**, which also matches exact words, and try turning
  on **Rerank passages**.
- **A scanned PDF:** PDFs that are images of text contain no text to extract. OCR isn't
  supported yet.
- **For follow-ups**, "Searched for:" in the sources panel shows how your question was
  rewritten. If the rewrite is wrong, ask a standalone question.

### I changed the embedding model and now nothing is found

Each embedding model gets its own index, because vectors from different models can't be mixed.
After changing `EMBEDDING_PROVIDER` or `EMBEDDING_MODEL_NAME`, click **Ingest Documents** again
to index your files with the new model. The old index is kept, so switching back works without
re-ingesting.

### I edited or deleted a file, but answers still use the old version

- **Edited files:** click **Ingest Documents** again. Each file is compared with what's indexed,
  and its outdated chunks are replaced.
- **Deleted files:** open **Manage index** in the sidebar. Files that no longer exist are marked
  *Missing from disk*, and **Remove missing documents** removes them from the index.
- **Starting over:** **Manage index → Reset index** empties the index. Your files and chat
  history are kept.

Indexes created before v0.1.0 may still contain duplicate chunks from edits made back then. A
one-time **Reset index** followed by **Ingest Documents** cleans them up.

### Which retrieval mode should I use?

Set it in the sidebar, or with `RETRIEVAL_MODE` in `.env`.

| Mode | Best for |
|---|---|
| **Hybrid** (default) | Most documents. Combines keyword search (BM25) with vector search, so exact names, IDs and error codes are found as well as paraphrases. |
| **Vector similarity** | Questions phrased very differently from the documents' wording. |
| **MMR** | Repetitive documents, where similarity search returns several near-identical passages. |

**Rerank passages** fetches a larger pool of candidates and reorders them with a cross-encoder
model, which judges relevance more accurately than either search. It downloads the model
(about 90 MB) on first use and adds roughly 0.2–1 s per question on a CPU. For non-English
documents, set `RERANKER_MODEL=BAAI/bge-reranker-base` (about 1.1 GB).

## Google Drive

### Google says "Google hasn't verified this app"

That's expected. You created the OAuth app yourself, and it's in testing mode. Click
**Advanced → Go to (your app name) (unsafe)** to continue. It's your own app, asking for
read-only access to your own Drive.

If you see **"Access blocked"** instead, add your Google account as a **test user** on the
OAuth consent screen (see the README).

### Using Google Drive with Docker

The Google sign-in opens a browser from the app, which doesn't work inside a container.
Sign in once outside Docker to create `token.json`:

```bash
pip install -r requirements.txt
python -c "import ingest; ingest.get_gdrive_credentials()"
```

Then move `credentials.json` and `token.json` into a `secrets/` folder and add this to the
`app` service in `docker-compose.yml`:

```yaml
    volumes:
      - ./secrets:/app/secrets
    environment:
      GDRIVE_CREDENTIALS_PATH: /app/secrets/credentials.json
      GDRIVE_TOKEN_PATH: /app/secrets/token.json
```

## Windows

- Activate the virtual environment with `.venv\Scripts\activate` (PowerShell:
  `.venv\Scripts\Activate.ps1`).
- Paths in `.env` can use forward slashes: `DATA_DIR=C:/Users/me/Documents/rag`.
- If `pip install` fails building a package, install the
  [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/), or
  use Docker Desktop and Docker Compose instead.

## Privacy: what goes over the network?

Your documents, questions and answers stay on your machine. The only outgoing connections are:

- **Ollama**, at `OLLAMA_BASE_URL` (local by default).
- **Hugging Face**, once, to download the embedding model (and the reranker, if you turn it on).
- **Google's APIs**, only when you ingest from Google Drive.
- **Ollama's model registry**, when pulling a model.

Usage telemetry from Streamlit, Chroma and Hugging Face Hub is turned off in
`.streamlit/config.toml`, `vectorstore.py` and `config.py`.
