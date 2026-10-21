# Security Policy

## Supported versions

Security fixes are made for the latest release only.

## Reporting a vulnerability

Please **don't open a public issue** for security problems. Report them privately through
[GitHub's private vulnerability reporting](https://github.com/husseinzayat3/PrivateRAG/security/advisories/new)
instead.

Include what you found, how to reproduce it, and what an attacker could do with it. You can
expect an acknowledgement within 7 days. Once a fix is released, you'll be credited in the
release notes unless you'd rather not be.

## Scope and design notes

PrivateRAG is designed to run on your own machine, for a single user.

- **No authentication.** The web UI has no login. Don't expose port 8501 to the internet or
  an untrusted network without putting authentication in front of it (for example a reverse
  proxy with single sign-on).
- **Where data is stored.** Everything stays on your machine:

  | What | Default location |
  |---|---|
  | Your documents (uploaded or copied in) | `data/` |
  | The vector index (text chunks and embeddings) | `chroma_db/` |
  | Chat history | `chat_history.db` |
  | Google Drive OAuth client and token | `credentials.json`, `token.json` |

  With Docker Compose, these are in `./data` and the `storage` volume. `token.json` grants
  read access to your Google Drive, so treat it like a password.
- **Network access.** The app talks to Ollama (`OLLAMA_BASE_URL`), to Hugging Face to
  download the embedding and reranker models the first time, and to Google's APIs only when you ingest from
  Drive. Usage telemetry from Streamlit, Chroma and Hugging Face Hub is turned off.
