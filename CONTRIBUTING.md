# Contributing to PrivateRAG

Thanks for your interest! Bug reports, documentation fixes and pull requests are all welcome.
Issues labelled [`good first issue`](https://github.com/husseinzayat3/PrivateRAG/labels/good%20first%20issue)
are a good place to start.

PrivateRAG aims to stay **small and readable**: a reference RAG app people can understand in
an afternoon and fork. Changes that keep it that way are the easiest to merge. For larger
features, please open an issue first to discuss the approach.

## Development setup

You need Python 3.10+ and, to run the app, [Ollama](https://ollama.com/).

```bash
git clone https://github.com/husseinzayat3/PrivateRAG.git
cd PrivateRAG
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt
```

Run the app with `streamlit run app.py`. Code reloading is off by default (see
`.streamlit/config.toml`); to reload on save while developing, run
`streamlit run app.py --server.fileWatcherType auto`. In that mode Streamlit logs harmless
"No module named 'torchvision'" errors after the embedding model loads.

## Checks

Both run in CI on every pull request, on Python 3.10–3.14. CI also builds the Docker image
and checks that the app starts in it.

```bash
ruff check .        # lint (add --fix to fix import order etc.)
pytest              # tests
```

The tests use fake embeddings and a temporary ChromaDB, so they don't need Ollama, a GPU or a
model download, and they run in seconds. `tests/test_app.py` runs the real Streamlit script
with Streamlit's `AppTest`, with the LLM faked. Please add tests for new behaviour where
practical. See `tests/conftest.py` for the `data_dir` and `store` fixtures.

New to the code? [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) explains how it fits together
and where to make common changes.

## Code layout

| File | What it does |
|---|---|
| `app.py` | Streamlit UI: chats, uploads, ingestion, settings |
| `ingest.py` | Loading local and Google Drive files, chunking, deduplicated storage |
| `rag.py` | Prompts, follow-up question rewriting, streaming answers |
| `retrieval.py` | Search modes: vector similarity, MMR, hybrid BM25 + vectors, reranking |
| `vectorstore.py` | Embedding model, Chroma collection, listing and deleting indexed documents |
| `chat_store.py` | Chat history in SQLite |
| `config.py` | Settings from environment variables / `.env` |

## Pull requests

1. Fork the repo and create a branch from `main`.
2. Make your change, with tests and docs as needed.
3. Add a line under **Unreleased** in [CHANGELOG.md](CHANGELOG.md) if users will notice the change.
4. Make sure `ruff check .` and `pytest` pass, then open a pull request and fill in the template.

Keep pull requests focused: one fix or feature per PR is much easier to review.

## Releasing (maintainers)

1. Update `VERSION` in `config.py`, and in `CHANGELOG.md` rename **Unreleased** to the new
   version with today's date (`## [0.2.0] - 2026-12-01`), plus a new empty **Unreleased**
   section and an updated compare link at the bottom.
2. Commit, then tag and push: `git tag -a v0.2.0 -m "v0.2.0" && git push origin v0.2.0`.
3. The [release workflow](.github/workflows/release.yml) checks the version and changelog,
   publishes the Docker image to `ghcr.io/husseinzayat3/privaterag` (amd64 and arm64), and
   creates the GitHub Release with the changelog section as its notes.

## Reporting bugs and security issues

Open an [issue](https://github.com/husseinzayat3/PrivateRAG/issues/new/choose) for bugs and
feature ideas. Please report security vulnerabilities privately, as described in
[SECURITY.md](SECURITY.md).

By contributing, you agree that your contributions are licensed under the [MIT License](LICENSE)
and that you'll follow the [Code of Conduct](CODE_OF_CONDUCT.md).
