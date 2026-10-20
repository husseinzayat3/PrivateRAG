from langchain_core.documents import Document

import ingest
from ingest import chunk_and_store, chunk_id


def stored_ids(store):
    return set(store.get(include=[])["ids"])


def long_doc(source="doc.txt", word="alpha"):
    # Several chunks' worth of text, with spaces so the splitter can break it
    text = " ".join(f"{word}{i}" for i in range(ingest.CHUNK_SIZE))
    return Document(page_content=text, metadata={"source": source})


# --- ID generation ---

def test_chunk_id_is_stable_sha256():
    doc = Document(page_content="hello", metadata={"source": "a.txt", "page": 1})

    assert chunk_id(doc) == chunk_id(Document(page_content="hello", metadata={"source": "a.txt", "page": 1}))
    assert len(chunk_id(doc)) == 64
    int(chunk_id(doc), 16)  # hex


def test_chunk_id_depends_on_source_page_row_and_content():
    base = Document(page_content="hello", metadata={"source": "a.txt", "page": 1, "row": 0})
    variants = [
        Document(page_content="hello", metadata={"source": "b.txt", "page": 1, "row": 0}),
        Document(page_content="hello", metadata={"source": "a.txt", "page": 2, "row": 0}),
        Document(page_content="hello", metadata={"source": "a.txt", "page": 1, "row": 1}),
        Document(page_content="world", metadata={"source": "a.txt", "page": 1, "row": 0}),
    ]

    ids = {chunk_id(d) for d in [base, *variants]}

    assert len(ids) == 5


def test_chunk_id_ignores_unrelated_metadata():
    a = Document(page_content="hello", metadata={"source": "a.txt", "author": "x"})
    b = Document(page_content="hello", metadata={"source": "a.txt", "author": "y"})

    assert chunk_id(a) == chunk_id(b)


# --- Chunking and storing ---

def test_splits_long_documents_into_bounded_chunks(store):
    added = chunk_and_store([long_doc()])

    assert added > 1
    contents = store.get(include=["documents"])["documents"]
    assert len(contents) == added
    assert all(len(c) <= ingest.CHUNK_SIZE for c in contents)


def test_reingesting_same_documents_adds_nothing(store):
    first = chunk_and_store([long_doc()])
    ids_after_first = stored_ids(store)

    second = chunk_and_store([long_doc()])

    assert first > 0
    assert second == 0
    assert stored_ids(store) == ids_after_first


def test_duplicate_documents_in_one_batch_are_stored_once(store):
    doc = Document(page_content="short doc", metadata={"source": "a.txt"})

    added = chunk_and_store([doc, doc])

    assert added == 1
    assert len(stored_ids(store)) == 1


def test_only_new_content_is_added(store):
    chunk_and_store([Document(page_content="original", metadata={"source": "a.txt"})])

    added = chunk_and_store([
        Document(page_content="original", metadata={"source": "a.txt"}),
        Document(page_content="new doc", metadata={"source": "b.txt"}),
    ])

    assert added == 1
    assert len(stored_ids(store)) == 2


def test_stored_ids_match_chunk_ids(store):
    doc = Document(page_content="short doc", metadata={"source": "a.txt"})

    chunk_and_store([doc])

    assert stored_ids(store) == {chunk_id(doc)}


def test_empty_input_stores_nothing(store):
    assert chunk_and_store([]) == 0
    assert stored_ids(store) == set()
