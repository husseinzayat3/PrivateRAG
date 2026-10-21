"""Local, persistent chat history backed by SQLite."""
import json
import sqlite3
import time
import uuid
from contextlib import closing

from config import CHAT_DB_PATH

TITLE_LENGTH = 60

SCHEMA = """
CREATE TABLE IF NOT EXISTS chats (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    sources TEXT,
    search_query TEXT,
    is_error INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id, id);
"""


def _connect(db_path=None):
    # A short-lived connection per call keeps this safe across Streamlit's threads
    conn = sqlite3.connect(db_path or CHAT_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


def make_title(question):
    """Short chat title from the first question."""
    title = " ".join(question.split())
    return title if len(title) <= TITLE_LENGTH else title[:TITLE_LENGTH - 1].rstrip() + "…"


def create_chat(title, db_path=None):
    chat_id = uuid.uuid4().hex
    now = time.time()
    with closing(_connect(db_path)) as conn, conn:
        conn.execute(
            "INSERT INTO chats (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (chat_id, title, now, now),
        )
    return chat_id


def list_chats(db_path=None):
    """All chats, most recently active first."""
    with closing(_connect(db_path)) as conn:
        rows = conn.execute("SELECT id, title, updated_at FROM chats ORDER BY updated_at DESC").fetchall()
    return [dict(r) for r in rows]


def chat_exists(chat_id, db_path=None):
    with closing(_connect(db_path)) as conn:
        return conn.execute("SELECT 1 FROM chats WHERE id = ?", (chat_id,)).fetchone() is not None


def get_messages(chat_id, db_path=None):
    """Messages of a chat in order, as {"role", "content", ...} dicts."""
    with closing(_connect(db_path)) as conn:
        rows = conn.execute(
            "SELECT role, content, sources, search_query, is_error FROM messages WHERE chat_id = ? ORDER BY id",
            (chat_id,),
        ).fetchall()
    return [
        {
            "role": r["role"],
            "content": r["content"],
            "sources": json.loads(r["sources"]) if r["sources"] else [],
            "search_query": r["search_query"],
            "is_error": bool(r["is_error"]),
        }
        for r in rows
    ]


def add_message(chat_id, role, content, sources=None, search_query=None, is_error=False, db_path=None):
    now = time.time()
    with closing(_connect(db_path)) as conn, conn:
        conn.execute(
            "INSERT INTO messages (chat_id, role, content, sources, search_query, is_error, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (chat_id, role, content, json.dumps(sources) if sources else None, search_query, int(is_error), now),
        )
        conn.execute("UPDATE chats SET updated_at = ? WHERE id = ?", (now, chat_id))


def delete_chat(chat_id, db_path=None):
    with closing(_connect(db_path)) as conn, conn:
        conn.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
