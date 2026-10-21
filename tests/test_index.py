from langchain_core.documents import Document

from ingest import chunk_and_store
from vectorstore import all_chunks, delete_documents, list_documents, reset_index


def doc(text, source, **meta):
    return Document(page_content=text, metadata={"source": source, **meta})


def test_list_documents_counts_chunks_per_source(store):
    chunk_and_store([
        doc("page one", "/data/report.pdf", page=0),
        doc("page two", "/data/report.pdf", page=1),
        doc("notes", "/data/notes.txt"),
        doc("drive doc", "https://docs.google.com/document/d/abc/edit", title="Roadmap"),
    ])

    assert list_documents() == [
        {"source": "/data/notes.txt", "name": "notes.txt", "chunks": 1},
        {"source": "/data/report.pdf", "name": "report.pdf", "chunks": 2},
        {"source": "https://docs.google.com/document/d/abc/edit", "name": "Roadmap", "chunks": 1},
    ]


def test_list_documents_on_empty_index(store):
    assert list_documents() == []


def test_delete_documents_removes_only_their_chunks(store):
    chunk_and_store([doc("a1", "a.txt"), doc("b1", "b.txt"), doc("c1", "c.txt")])

    removed = delete_documents(["a.txt", "c.txt"])

    assert removed == 2
    assert [d["source"] for d in list_documents()] == ["b.txt"]


def test_delete_unknown_document_is_a_no_op(store):
    chunk_and_store([doc("a1", "a.txt")])

    assert delete_documents(["missing.txt"]) == 0
    assert len(list_documents()) == 1


def test_reset_index_removes_everything(store):
    chunk_and_store([doc("a1", "a.txt"), doc("b1", "b.txt")])

    reset_index()

    assert list_documents() == []
    # The index is still usable afterwards
    chunk_and_store([doc("c1", "c.txt")])
    assert [d["source"] for d in list_documents()] == ["c.txt"]


def test_all_chunks_returns_documents_with_ids(store):
    chunk_and_store([doc("a1", "a.txt"), doc("b1", "b.txt")])

    chunks = all_chunks()

    assert sorted(c.page_content for c in chunks) == ["a1", "b1"]
    assert all(c.id and c.metadata["source"] for c in chunks)
