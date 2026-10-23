"""Smoke tests that run the Streamlit app script, with the LLM and retrieval faked."""
import pytest
from langchain_core.documents import Document
from streamlit.testing.v1 import AppTest

import chat_store
import config
import rag


@pytest.fixture
def app(data_dir, store, tmp_path, monkeypatch):
    """The app wired to a temporary data folder, index and chat database, without Ollama."""
    monkeypatch.setattr(config, "DATA_DIR", str(data_dir))
    monkeypatch.setattr(chat_store, "CHAT_DB_PATH", str(tmp_path / "chats.db"))
    monkeypatch.setattr(rag, "warm_up", lambda *args: None)

    def fake_retrieve(question, history=(), **kwargs):
        return question, [Document("Error E7 means the boiler did not respond.", metadata={"source": "manual.md"})]

    def fake_stream_answer(question, docs, history=(), **kwargs):
        yield from ["E7 means ", "the boiler did not respond [1]."]

    monkeypatch.setattr(rag, "retrieve", fake_retrieve)
    monkeypatch.setattr(rag, "stream_answer", fake_stream_answer)
    return AppTest.from_file("../app.py", default_timeout=30)


def click(at, label):
    next(b for b in at.button if b.label == label).click().run()


def test_app_renders_without_errors(app):
    app.run()

    assert not app.exception
    assert app.title[0].value == "📚 PrivateRAG"
    assert [s.label for s in app.selectbox] == ["Retrieval mode"]
    assert any(f"v{config.VERSION}" in c.value for c in app.caption)


def test_add_samples_and_ingest(app, store):
    app.run()

    click(app, "Add sample documents")
    click(app, "Ingest Documents")

    assert not app.exception
    assert any(s.value.startswith("Ingestion complete! Added") for s in app.success)
    assert len(store.get(include=[])["ids"]) > 0


def test_manage_index_lists_documents(app):
    app.run()
    click(app, "Add sample documents")
    click(app, "Ingest Documents")

    click(app, "🗂️ Manage index")

    assert not app.exception
    assert len(app.dataframe[0].value) == 3


def test_asking_a_question_saves_a_chat_with_sources(app):
    app.run()

    app.chat_input[0].set_value("What does E7 mean?").run()

    assert not app.exception
    [chat] = chat_store.list_chats()
    assert chat["title"] == "What does E7 mean?"
    messages = chat_store.get_messages(chat["id"])
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[1]["content"] == "E7 means the boiler did not respond [1]."
    assert messages[1]["sources"][0]["label"] == "manual.md"
