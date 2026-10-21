import time

import pytest

import chat_store


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / "chats.db")


def test_new_chat_starts_empty(db):
    chat_id = chat_store.create_chat("First", db_path=db)

    assert chat_store.chat_exists(chat_id, db_path=db)
    assert chat_store.get_messages(chat_id, db_path=db) == []


def test_messages_round_trip_in_order(db):
    chat_id = chat_store.create_chat("Chat", db_path=db)
    sources = [{"label": "a.pdf, p. 1", "content": "passage"}]

    chat_store.add_message(chat_id, "user", "question", db_path=db)
    chat_store.add_message(chat_id, "assistant", "answer [1]", sources, "rewritten", db_path=db)

    assert chat_store.get_messages(chat_id, db_path=db) == [
        {"role": "user", "content": "question", "sources": [], "search_query": None, "is_error": False},
        {"role": "assistant", "content": "answer [1]", "sources": sources, "search_query": "rewritten",
         "is_error": False},
    ]


def test_error_flag_is_kept(db):
    chat_id = chat_store.create_chat("Chat", db_path=db)

    chat_store.add_message(chat_id, "assistant", "Ollama is down", is_error=True, db_path=db)

    assert chat_store.get_messages(chat_id, db_path=db)[0]["is_error"] is True


def test_chats_are_listed_by_latest_activity(db):
    older = chat_store.create_chat("Older", db_path=db)
    time.sleep(0.01)
    newer = chat_store.create_chat("Newer", db_path=db)
    assert [c["id"] for c in chat_store.list_chats(db_path=db)] == [newer, older]

    time.sleep(0.01)
    chat_store.add_message(older, "user", "bump", db_path=db)

    assert [c["title"] for c in chat_store.list_chats(db_path=db)] == ["Older", "Newer"]


def test_messages_are_scoped_to_their_chat(db):
    a = chat_store.create_chat("A", db_path=db)
    b = chat_store.create_chat("B", db_path=db)

    chat_store.add_message(a, "user", "only in A", db_path=db)

    assert chat_store.get_messages(b, db_path=db) == []


def test_delete_chat_removes_it_and_its_messages(db):
    chat_id = chat_store.create_chat("Doomed", db_path=db)
    chat_store.add_message(chat_id, "user", "hi", db_path=db)

    chat_store.delete_chat(chat_id, db_path=db)

    assert not chat_store.chat_exists(chat_id, db_path=db)
    assert chat_store.list_chats(db_path=db) == []
    assert chat_store.get_messages(chat_id, db_path=db) == []


def test_make_title_collapses_whitespace_and_truncates():
    assert chat_store.make_title("  What is\n  RAG?  ") == "What is RAG?"

    title = chat_store.make_title("word " * 50)
    assert len(title) <= chat_store.TITLE_LENGTH
    assert title.endswith("…")
