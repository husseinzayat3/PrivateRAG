import pytest
from langchain_chroma import Chroma
from langchain_core.embeddings import DeterministicFakeEmbedding

import ingest


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """An empty data directory that ingest_local_files() reads from."""
    path = tmp_path / "data"
    path.mkdir()
    monkeypatch.setattr(ingest, "DATA_DIR", str(path))
    return path


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A throwaway Chroma store with fake embeddings, so tests don't download a model."""
    vectorstore = Chroma(
        collection_name="test",
        persist_directory=str(tmp_path / "chroma"),
        embedding_function=DeterministicFakeEmbedding(size=32),
    )
    monkeypatch.setattr(ingest, "get_vectorstore", lambda: vectorstore)
    return vectorstore


def make_pdf(path, pages):
    """Writes a minimal PDF with one line of text per page."""
    objects = ["<< /Type /Catalog /Pages 2 0 R >>"]
    page_ids = [3 + 2 * i for i in range(len(pages))]
    kids = " ".join(f"{i} 0 R" for i in page_ids)
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>")
    font_id = 3 + 2 * len(pages)
    for i, text in enumerate(pages):
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET"
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {page_ids[i] + 1} 0 R "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> >>"
        )
        objects.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
    objects.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out = b"%PDF-1.4\n"
    offsets = []
    for num, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{num} 0 obj\n{body}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets).encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(out)
