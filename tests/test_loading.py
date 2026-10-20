from conftest import make_pdf

from ingest import ingest_local_files


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
    assert "Second page" in docs[1].page_content


def test_loads_csv_one_document_per_row(data_dir):
    (data_dir / "team.csv").write_text("name,city\nAlice,Berlin\nBob,Lisbon\n")

    docs, errors = ingest_local_files()

    assert errors == []
    assert [d.page_content for d in docs] == ["name: Alice\ncity: Berlin", "name: Bob\ncity: Lisbon"]
    assert [d.metadata["row"] for d in docs] == [0, 1]


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
