import importlib
import os

import config


def test_paths_come_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "docs"))
    monkeypatch.setenv("CHROMA_DB_DIR", str(tmp_path / "index"))
    monkeypatch.setenv("CHAT_DB_PATH", str(tmp_path / "state" / "chats.db"))
    try:
        cfg = importlib.reload(config)
        assert cfg.DATA_DIR == str(tmp_path / "docs")
        assert cfg.CHROMA_DB_DIR == str(tmp_path / "index")
        assert cfg.CHAT_DB_PATH == str(tmp_path / "state" / "chats.db")
        # Directories the app writes into are created up front
        assert (tmp_path / "docs").is_dir()
        assert (tmp_path / "state").is_dir()
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_relative_paths_resolve_against_project_root(monkeypatch):
    monkeypatch.setenv("CHROMA_DB_DIR", "index")
    try:
        cfg = importlib.reload(config)
        assert cfg.CHROMA_DB_DIR == os.path.join(cfg.BASE_DIR, "index")
    finally:
        monkeypatch.undo()
        importlib.reload(config)
