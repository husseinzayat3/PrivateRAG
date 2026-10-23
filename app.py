import os
import threading
import warnings

# Suppress HuggingFace and Google API warnings before anything else loads
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=FutureWarning, module="google.*")

import streamlit as st

import chat_store
from config import (
    DATA_DIR,
    GDRIVE_CREDENTIALS_PATH,
    LLM_TEMPERATURE,
    MAX_HISTORY_MESSAGES,
    OLLAMA_MODEL_NAME,
    RETRIEVAL_K,
    RETRIEVAL_MODE,
    USE_RERANKER,
    VERSION,
)
from ingest import LOADERS, chunk_and_store, copy_samples, ingest_gdrive, ingest_local_files
from rag import retrieve, source_label, stream_answer, warm_up
from retrieval import MODES
from vectorstore import delete_documents, list_documents, reset_index

st.set_page_config(page_title="PrivateRAG", page_icon="📚", layout="wide")

st.title("📚 PrivateRAG")
st.caption("Chat with your documents, privately. Answers come from your files, with citations.")


@st.cache_resource(show_spinner=False)
def start_warm_up(model, use_reranker):
    """Preloads the models in the background, once per process and settings."""
    threading.Thread(target=warm_up, args=(model, use_reranker), daemon=True).start()


def is_url(source):
    return source.startswith(("http://", "https://"))


def show_after_rerun(message):
    """Keeps a message to show once after st.rerun(), which would otherwise clear it."""
    st.session_state.flash = message
    st.rerun()


@st.dialog("Indexed documents", width="large")
def manage_index():
    docs = list_documents()
    if not docs:
        st.info("The index is empty. Add documents and click **Ingest Documents**.")
        return

    data_prefix = os.path.join(DATA_DIR, "")
    missing = [d["source"] for d in docs if not is_url(d["source"]) and not os.path.exists(d["source"])]
    st.caption(f"{len(docs)} documents, {sum(d['chunks'] for d in docs)} chunks. "
               "Ingesting again picks up edited files and replaces their old chunks.")
    st.dataframe(
        [{
            "Document": d["name"],
            "Chunks": d["chunks"],
            "Location": "Google Drive" if is_url(d["source"])
            else os.path.relpath(d["source"], DATA_DIR) if d["source"].startswith(data_prefix) else d["source"],
            "Status": "Missing from disk" if d["source"] in missing else "Indexed",
        } for d in docs],
        hide_index=True, width="stretch",
    )

    labels = {d["source"]: f"{d['name']} ({d['chunks']} chunks)" for d in docs}
    selected = st.multiselect("Remove documents from the index", options=list(labels), format_func=labels.get)
    delete_files = st.checkbox(
        "Also delete the files from the data folder",
        help="Otherwise local files are indexed again the next time you click Ingest Documents.",
    )
    if st.button("Remove selected", disabled=not selected):
        removed = delete_documents(selected)
        deleted = 0
        if delete_files:
            for source in selected:
                if source.startswith(data_prefix) and os.path.isfile(source):
                    os.remove(source)
                    deleted += 1
        show_after_rerun(f"Removed {len(selected)} documents ({removed} chunks) from the index"
                         + (f" and deleted {deleted} files." if delete_files else "."))

    if missing:
        st.divider()
        st.write(f"{len(missing)} indexed files no longer exist on disk.")
        if st.button(f"Remove {len(missing)} missing documents"):
            removed = delete_documents(missing)
            show_after_rerun(f"Removed {len(missing)} missing documents ({removed} chunks) from the index.")

    st.divider()
    confirm = st.checkbox(f"I want to remove all {len(docs)} documents from the index",
                          help="Your files and chat history are kept.")
    if st.button("Reset index", type="primary", disabled=not confirm):
        reset_index()
        show_after_rerun("The index was reset. Click **Ingest Documents** to index your files again.")


# The open chat lives in the URL (?chat=<id>), so a page refresh keeps it
chat_id = st.query_params.get("chat")
if chat_id and not chat_store.chat_exists(chat_id):
    st.query_params.clear()
    chat_id = None


def open_chat(new_chat_id):
    st.query_params.clear()
    if new_chat_id:
        st.query_params["chat"] = new_chat_id


def delete_chat(old_chat_id):
    chat_store.delete_chat(old_chat_id)
    st.query_params.clear()


# Sidebar for chats, configuration and ingestion
with st.sidebar:
    st.header("Chats")
    st.button("➕ New chat", on_click=open_chat, args=(None,), width="stretch")
    chats = chat_store.list_chats()
    # Scroll only once the list gets long; short lists take their natural height
    with st.container(height=360 if len(chats) > 7 else "content", border=False):
        for chat in chats:
            title = chat["title"]
            st.button(
                # One line per chat; the full title shows on hover
                title if len(title) <= 32 else title[:31].rstrip() + "…",
                help=title, key=f"chat-{chat['id']}", on_click=open_chat, args=(chat["id"],),
                type="primary" if chat["id"] == chat_id else "secondary", width="stretch",
            )
    if chat_id:
        st.button("🗑️ Delete this chat", on_click=delete_chat, args=(chat_id,), width="stretch")

    st.header("Ingestion Settings")

    st.subheader("Local Files")
    # The size limit is server.maxUploadSize in .streamlit/config.toml
    uploaded_files = st.file_uploader(
        "Upload files to local directory", accept_multiple_files=True,
        type=[ext.lstrip(".") for ext in LOADERS],
    )
    # The uploader keeps its files across reruns, so only save ones not saved yet
    if "saved_uploads" not in st.session_state:
        st.session_state.saved_uploads = set()
    new_uploads = [
        uf for uf in uploaded_files or []
        if (uf.name, uf.size) not in st.session_state.saved_uploads
    ]
    for uf in new_uploads:
        file_path = os.path.join(DATA_DIR, os.path.basename(uf.name))
        with open(file_path, "wb") as f:
            f.write(uf.getbuffer())
        st.session_state.saved_uploads.add((uf.name, uf.size))
    if new_uploads:
        st.success(f"Saved {len(new_uploads)} files to {DATA_DIR}")
    if st.button("Add sample documents", help="Copies three short example documents into the data folder."):
        copied = copy_samples()
        st.success(f"Added {copied} sample documents. Click **Ingest Documents** to index them."
                   if copied else "The sample documents are already in the data folder.")

    st.subheader("Google Drive")
    has_gdrive_creds = os.path.exists(GDRIVE_CREDENTIALS_PATH)
    if not has_gdrive_creds:
        st.warning("Google Drive credentials.json not found in project root. Drive ingestion disabled.")
        gdrive_folder_id = ""
    else:
        st.success("Google Drive credentials found!")
        gdrive_folder_id = st.text_input("Enter Google Drive Folder ID:")

    if st.button("Ingest Documents"):
        with st.spinner("Ingesting documents..."):
            docs = []
            # Local docs
            try:
                local_docs, local_errors = ingest_local_files()
                docs.extend(local_docs)
                st.write(f"Loaded {len(local_docs)} local documents.")
                for err in local_errors:
                    st.warning(f"Failed to load {err}")
            except Exception as e:
                st.error(f"Error loading local files: {e}")

            # GDrive docs
            if has_gdrive_creds and gdrive_folder_id:
                try:
                    gdrive_docs = ingest_gdrive(folder_id=gdrive_folder_id)
                    docs.extend(gdrive_docs)
                    st.write(f"Loaded {len(gdrive_docs)} Google Drive documents.")
                except Exception as e:
                    st.error(f"Error loading GDrive files: {e}")

            if docs:
                try:
                    result = chunk_and_store(docs)
                    st.success(f"Ingestion complete! Added {result.added} new chunks"
                               + (f" and removed {result.removed} outdated ones." if result.removed else "."))
                except Exception as e:
                    st.error(f"Error storing documents in ChromaDB: {e}")
            else:
                st.warning("No documents found to ingest.")

    if st.button("🗂️ Manage index", width="stretch"):
        manage_index()
    if flash := st.session_state.pop("flash", None):
        st.success(flash)

    st.header("Chat Settings")
    llm_model = st.text_input("Ollama model", value=OLLAMA_MODEL_NAME,
                              help="Any model you've pulled with `ollama pull`.")
    temperature = st.slider("Temperature", 0.0, 1.0, LLM_TEMPERATURE, 0.05)
    k = st.slider("Passages to retrieve (k)", 1, 20, RETRIEVAL_K,
                  help="More passages give the model more context, but take longer to read.")
    retrieval_mode = st.selectbox(
        "Retrieval mode", options=list(MODES), format_func=MODES.get,
        index=list(MODES).index(RETRIEVAL_MODE) if RETRIEVAL_MODE in MODES else 0,
        help="Hybrid also matches exact words (names, IDs, error codes) that vector search can miss. "
             "MMR avoids returning several near-identical passages.",
    )
    use_reranker = st.toggle(
        "Rerank passages", value=USE_RERANKER,
        help="Reorders the retrieved passages with a cross-encoder model for better relevance. "
             "Downloads the model on first use and adds a little time per question.",
    )

    st.divider()
    st.caption(f"PrivateRAG v{VERSION} · [GitHub](https://github.com/husseinzayat3/PrivateRAG) · "
               "[Troubleshooting](https://github.com/husseinzayat3/PrivateRAG/blob/main/docs/TROUBLESHOOTING.md)")

start_warm_up(llm_model, use_reranker)


def render_sources(sources, search_query=None):
    """Shows the retrieved passages behind an answer, numbered to match its citations."""
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        if search_query:
            st.caption(f"Searched for: {search_query}")
        for i, src in enumerate(sources, start=1):
            st.markdown(f"**[{i}] {src['label']}**")
            st.text(src["content"])


def clear_on_first_chunk(stream, placeholder):
    """Passes the stream through, removing the status placeholder once text arrives."""
    for i, chunk in enumerate(stream):
        if i == 0:
            placeholder.empty()
        yield chunk


st.header("Chat with your Documents")

messages = chat_store.get_messages(chat_id) if chat_id else []

# Display the saved messages of the open chat
for message in messages:
    with st.chat_message(message["role"]):
        if message["is_error"]:
            st.error(message["content"])
        else:
            st.markdown(message["content"])
        render_sources(message["sources"], message["search_query"])

# React to user input
if prompt := st.chat_input("Ask a question about your documents:"):
    # Failed answers are kept for display but not fed back to the model
    history = [m for m in messages if not m["is_error"]]
    history = history[-MAX_HISTORY_MESSAGES:] if MAX_HISTORY_MESSAGES else []

    is_new_chat = chat_id is None
    if is_new_chat:
        chat_id = chat_store.create_chat(chat_store.make_title(prompt))
        st.query_params["chat"] = chat_id
    chat_store.add_message(chat_id, "user", prompt)
    st.chat_message("user").markdown(prompt)

    with st.chat_message("assistant"):
        status = st.empty()
        is_error = False
        try:
            status.caption("🔎 Searching documents…")
            search_query, docs = retrieve(prompt, history, model=llm_model, k=k,
                                          mode=retrieval_mode, use_reranker=use_reranker)
            status.caption("💭 Thinking…")
            response = st.write_stream(clear_on_first_chunk(
                stream_answer(prompt, docs, history, model=llm_model, temperature=temperature), status
            ))
        except Exception as e:
            is_error = True
            response = (f"Error connecting to LLM or retrieving context. Make sure Ollama is running "
                        f"and the model is pulled (`ollama pull {llm_model}`). Details: {e}")
            st.error(response)
            docs, search_query = [], None
        status.empty()
        sources = [{"label": source_label(d), "content": d.page_content} for d in docs]
        shown_query = search_query if search_query and search_query != prompt else None
        render_sources(sources, shown_query)

    chat_store.add_message(chat_id, "assistant", response, sources, shown_query, is_error)
    if is_new_chat:
        # Rerun so the new chat shows up in the sidebar list
        st.rerun()
