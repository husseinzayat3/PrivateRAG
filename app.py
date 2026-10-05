import streamlit as st
import os
from ingest import ingest_local_files, ingest_gdrive, chunk_and_store
from rag import answer_question
from config import DATA_DIR, GDRIVE_CREDENTIALS_PATH

st.set_page_config(page_title="RAG System", page_icon="🤖", layout="wide")

st.title("Generic RAG System (Local + Google Drive)")

# Sidebar for configuration and ingestion
with st.sidebar:
    st.header("1. Ingestion Settings")
    
    st.subheader("Local Files")
    uploaded_files = st.file_uploader("Upload files to local directory", accept_multiple_files=True)
    if uploaded_files:
        for uf in uploaded_files:
            file_path = os.path.join(DATA_DIR, uf.name)
            with open(file_path, "wb") as f:
                f.write(uf.getbuffer())
        st.success(f"Saved {len(uploaded_files)} files to {DATA_DIR}")
        
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
                local_docs = ingest_local_files()
                docs.extend(local_docs)
                st.write(f"Loaded {len(local_docs)} local documents.")
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
                chunk_and_store(docs)
                st.success("Ingestion complete! ChromaDB updated.")
            else:
                st.warning("No documents found to ingest.")


st.header("2. Chat with your Documents")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask a question about your documents:"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Thinking..."):
        response = answer_question(prompt)
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
