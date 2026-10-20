import os
import warnings

# Suppress HuggingFace and Google API warnings before anything else loads
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=FutureWarning, module="google.*")

import streamlit as st

from config import (
    DATA_DIR,
    GDRIVE_CREDENTIALS_PATH,
    LLM_TEMPERATURE,
    MAX_HISTORY_MESSAGES,
    OLLAMA_MODEL_NAME,
    RETRIEVAL_K,
)
from ingest import chunk_and_store, ingest_gdrive, ingest_local_files
from rag import retrieve, source_label, stream_answer

st.set_page_config(page_title="RAG System", page_icon="🤖", layout="wide")

st.title("Generic RAG System (Local + Google Drive)")

# Sidebar for configuration and ingestion
with st.sidebar:
    st.header("1. Ingestion Settings")
    
    st.subheader("Local Files")
    uploaded_files = st.file_uploader("Upload files to local directory", accept_multiple_files=True)
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
                    added = chunk_and_store(docs)
                    st.success(f"Ingestion complete! Added {added} new chunks to ChromaDB.")
                except Exception as e:
                    st.error(f"Error storing documents in ChromaDB: {e}")
            else:
                st.warning("No documents found to ingest.")


with st.sidebar:
    st.header("2. Chat Settings")
    llm_model = st.text_input("Ollama model", value=OLLAMA_MODEL_NAME,
                              help="Any model you've pulled with `ollama pull`.")
    temperature = st.slider("Temperature", 0.0, 1.0, LLM_TEMPERATURE, 0.05)
    k = st.slider("Passages to retrieve (k)", 1, 20, RETRIEVAL_K)
    if st.button("Clear chat"):
        st.session_state.messages = []


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


st.header("3. Chat with your Documents")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        render_sources(message.get("sources"), message.get("search_query"))

# React to user input
if prompt := st.chat_input("Ask a question about your documents:"):
    history = st.session_state.messages[-MAX_HISTORY_MESSAGES:] if MAX_HISTORY_MESSAGES else []
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching documents..."):
                search_query, docs = retrieve(prompt, history, model=llm_model, k=k)
            response = st.write_stream(
                stream_answer(prompt, docs, history, model=llm_model, temperature=temperature)
            )
        except Exception as e:
            response = (f"Error connecting to LLM or retrieving context. Make sure Ollama is running "
                        f"and the model is pulled (`ollama pull {llm_model}`). Details: {e}")
            st.error(response)
            docs, search_query = [], None
        sources = [{"label": source_label(d), "content": d.page_content} for d in docs]
        shown_query = search_query if search_query and search_query != prompt else None
        render_sources(sources, shown_query)
    # Add assistant response to chat history
    st.session_state.messages.append({
        "role": "assistant", "content": response,
        "sources": sources, "search_query": shown_query,
    })
