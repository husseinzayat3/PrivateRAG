import io

from conftest import make_pdf

import ingest
from ingest import DrivePdfLoader, copy_samples, ingest_local_files


def test_loads_text_and_markdown(data_dir):
    (data_dir / "notes.txt").write_text("Plain text notes")
    (data_dir / "readme.md").write_text("# Markdown doc")

    docs, errors = ingest_local_files()

    assert errors == []
    assert sorted(d.page_content for d in docs) == ["# Markdown doc", "Plain text notes"]


def test_loads_pdf_one_document_per_page(data_dir):
    make_pdf(data_dir / "report.pdf", ["First page", "Second page"])

    docs, errors = ingest_local_files()

    assert errors == []
    assert [d.metadata["page"] for d in docs] == [0, 1]
    assert [d.metadata["page_label"] for d in docs] == ["1", "2"]
    assert docs[1].metadata["source"] == str(data_dir / "report.pdf")
    assert "Second page" in docs[1].page_content


def test_loads_csv_one_document_per_row(data_dir):
    (data_dir / "team.csv").write_text("name,city\nAlice,Berlin\nBob,Lisbon\n")

    docs, errors = ingest_local_files()

    assert errors == []
    assert [d.page_content for d in docs] == ["name: Alice\ncity: Berlin", "name: Bob\ncity: Lisbon"]
    assert [d.metadata["row"] for d in docs] == [0, 1]


def test_csv_keeps_extra_cells(data_dir):
    (data_dir / "odd.csv").write_text("name,city\nAlice,Berlin,Germany,EU\n")

    docs, _ = ingest_local_files()

    assert docs[0].page_content == "name: Alice\ncity: Berlin\nNone: Germany,EU"


def test_reads_non_utf8_text(data_dir):
    (data_dir / "legacy.txt").write_bytes("Café crème, 5 €".encode("cp1252"))
    (data_dir / "bom.txt").write_bytes("\ufeffWith BOM".encode("utf-8"))

    docs, errors = ingest_local_files()

    assert errors == []
    assert sorted(d.page_content for d in docs) == ["Café crème, 5 €", "With BOM"]


def test_loads_files_in_subdirectories(data_dir):
    (data_dir / "nested").mkdir()
    (data_dir / "nested" / "deep.txt").write_text("Nested file")

    docs, _ = ingest_local_files()

    assert [d.page_content for d in docs] == ["Nested file"]


def test_skips_unsupported_file_types(data_dir):
    (data_dir / "image.png").write_bytes(b"\x89PNG")

    docs, errors = ingest_local_files()

    assert docs == []
    assert errors == []


def test_reports_files_that_fail_to_load(data_dir):
    (data_dir / "broken.pdf").write_bytes(b"this is not a pdf")
    (data_dir / "ok.txt").write_text("Still loaded")

    docs, errors = ingest_local_files()

    assert [d.page_content for d in docs] == ["Still loaded"]
    assert len(errors) == 1
    assert errors[0].startswith("broken.pdf:")


def test_copy_samples_skips_readme_and_existing_files(data_dir, monkeypatch, tmp_path):
    samples = tmp_path / "samples"
    samples.mkdir()
    (samples / "README.md").write_text("about the samples")
    (samples / "a.md").write_text("sample a")
    (samples / "b.md").write_text("sample b")
    (data_dir / "b.md").write_text("my own edited copy")
    monkeypatch.setattr(ingest, "SAMPLES_DIR", str(samples))

    assert copy_samples() == 1
    assert sorted(p.name for p in data_dir.iterdir()) == ["a.md", "b.md"]
    assert (data_dir / "b.md").read_text() == "my own edited copy"
    assert copy_samples() == 0


def test_bundled_samples_load(data_dir):
    assert copy_samples() == 3
    docs, errors = ingest_local_files()
    assert errors == []
    assert len(docs) == 3


def test_drive_pdf_loader_reads_downloaded_bytes(tmp_path):
    make_pdf(tmp_path / "a.pdf", ["Hello from Drive"])

    docs = DrivePdfLoader(file=io.BytesIO((tmp_path / "a.pdf").read_bytes())).load()

    assert "Hello from Drive" in docs[0].page_content
    # GoogleDriveLoader adds the Drive URL as source and the file name as title
    assert docs[0].metadata == {"page": 0, "page_label": "1", "total_pages": 1}
