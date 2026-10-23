# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - Unreleased

First public release.

### Added

- Hybrid retrieval: BM25 keyword search fused with vector search (reciprocal rank fusion),
  now the default. `RETRIEVAL_MODE` and a sidebar setting switch between hybrid, MMR and plain
  vector similarity.
- Optional cross-encoder reranker (`USE_RERANKER`, `RERANKER_MODEL`, or the sidebar toggle).
- **Manage index** dialog: list indexed documents with chunk counts, remove documents
  (optionally deleting the files), remove documents whose files no longer exist, and reset the index.
- Re-ingesting an edited file replaces its outdated chunks instead of keeping them next to
  the new ones.
- Chat with local `.pdf`, `.txt`, `.md` and `.csv` files and Google Drive folders, using a local
  LLM through Ollama and local embeddings (Hugging Face `bge-small-en-v1.5` or any Ollama
  embedding model).
- Cited answers (`[1]`, `[2]`, …) with a sources panel showing the retrieved passages, file and page.
- Follow-up questions rewritten into standalone questions using the chat history.
- Streaming answers, with the models preloaded at startup.
- Saved chat history (SQLite) with a sidebar list; each chat has its own URL.
- Deduplicated ingestion: re-ingesting unchanged files adds nothing.
- Settings through environment variables or a `.env` file, including the storage paths
  `DATA_DIR`, `CHROMA_DB_DIR`, `CHAT_DB_PATH`, `GDRIVE_CREDENTIALS_PATH` and `GDRIVE_TOKEN_PATH`.
- One-command start with Docker Compose (app + Ollama, model pulled on first boot),
  with an optional NVIDIA GPU override. The app image is published to
  `ghcr.io/husseinzayat3/privaterag` for amd64 and arm64 with each release.
- Sample documents and an **Add sample documents** button for trying the app without your own files.
- The version is shown at the bottom of the sidebar.
- Documentation: architecture guide, Google Drive setup guide, troubleshooting guide,
  contributing guide, code of conduct, security policy, and issue/PR templates.
- CI with ruff and pytest on Python 3.10–3.14, app smoke tests, and a Docker build check;
  a release workflow; Dependabot updates.

### Changed

- Local files are loaded with `pypdf` and the standard library instead of
  `langchain-community`, which is being retired. `langchain-community` and `unstructured` are
  no longer dependencies. Existing indexes stay valid.

### Fixed

- PDFs in Google Drive folders failed to load, because the Drive loader needed `PyPDF2`,
  which wasn't installed. They are now read with `pypdf`.

### Security

- Usage telemetry from Streamlit, Chroma and Hugging Face Hub is turned off.
- Uploads are limited to supported file types and 50 MB per file.

[Unreleased]: https://github.com/husseinzayat3/PrivateRAG/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/husseinzayat3/PrivateRAG/releases/tag/v0.1.0
