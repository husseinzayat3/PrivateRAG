from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

from rag import format_docs, source_label, to_messages
from vectorstore import collection_name


def test_source_label_for_pdf_page():
    assert source_label(Document("x", metadata={"source": "/data/report.pdf", "page": 2})) == "report.pdf, p. 3"


def test_source_label_prefers_page_label_and_title():
    doc = Document("x", metadata={"source": "/data/r.pdf", "title": "Annual Report", "page": 4, "page_label": "iv"})

    assert source_label(doc) == "Annual Report, p. iv"


def test_source_label_for_csv_row():
    assert source_label(Document("x", metadata={"source": "/data/team.csv", "row": 0})) == "team.csv, row 1"


def test_source_label_for_plain_file():
    assert source_label(Document("x", metadata={"source": "/data/notes.txt"})) == "notes.txt"


def test_format_docs_numbers_passages_with_sources():
    docs = [
        Document("First passage", metadata={"source": "a.txt"}),
        Document("Second passage", metadata={"source": "b.pdf", "page": 0}),
    ]

    assert format_docs(docs) == "[1] (Source: a.txt)\nFirst passage\n\n[2] (Source: b.pdf, p. 1)\nSecond passage"


def test_to_messages_maps_roles():
    messages = to_messages([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}])

    assert [type(m) for m in messages] == [HumanMessage, AIMessage]
    assert [m.content for m in messages] == ["hi", "hello"]


def test_collection_name_is_valid_for_chroma():
    name = collection_name()

    assert 3 <= len(name) <= 512
    assert name[0].isalnum() and name[-1].isalnum()
    assert all(c.isalnum() or c in "._-" for c in name)
