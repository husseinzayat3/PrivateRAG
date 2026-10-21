import pytest
from langchain_core.documents import Document

import retrieval
from ingest import chunk_and_store
from retrieval import BM25, reciprocal_rank_fusion, search, tokenize


def doc(text, source="a.txt", id=None):
    return Document(page_content=text, metadata={"source": source}, id=id)


MANUAL = [
    doc("Error E1 means there is no power on the C wire.", "manual.md#1"),
    doc("Error E7 means the heating system did not respond within 15 minutes.", "manual.md#2"),
    doc("Clean the display with a dry cloth. Do not use spray cleaners.", "manual.md#3"),
    doc("The thermostat learns your schedule over the first ten days.", "manual.md#4"),
]


def test_tokenize_keeps_ids_and_numbers():
    assert tokenize("Error E7: INV-2041 failed!") == ["error", "e7", "inv", "2041", "failed"]


def test_bm25_ranks_exact_term_first():
    results = BM25(MANUAL).search("what does E7 mean", k=2)

    assert results[0].metadata["source"] == "manual.md#2"


def test_bm25_ignores_unknown_terms_and_empty_corpus():
    assert BM25(MANUAL).search("zebra", k=3) == []
    assert BM25([]).search("anything", k=3) == []


def test_bm25_prefers_rarer_terms():
    # "error" appears in two passages, "cloth" in one
    results = BM25(MANUAL).search("error cloth", k=4)

    assert results[0].metadata["source"] == "manual.md#3"


def test_rrf_rewards_documents_ranked_high_in_both_lists():
    a, b, c = doc("a", id="a"), doc("b", id="b"), doc("c", id="c")

    fused = reciprocal_rank_fusion([a, b, c], [b, c, a])

    assert [d.id for d in fused] == ["b", "a", "c"]


def test_rrf_identifies_documents_without_ids_by_content():
    fused = reciprocal_rank_fusion([doc("same")], [doc("same"), doc("other")])

    assert [d.page_content for d in fused] == ["same", "other"]


@pytest.mark.parametrize("mode", ["similarity", "mmr", "hybrid"])
def test_search_modes_return_k_distinct_chunks(store, mode):
    chunk_and_store(MANUAL)

    results = search("error code", k=3, mode=mode)

    assert len(results) == 3
    assert len({d.page_content for d in results}) == 3


def test_hybrid_search_finds_exact_terms(store):
    # The fake embeddings are random, so only the keyword side can find this
    chunk_and_store(MANUAL)

    results = search("E7", k=1, mode="hybrid")

    assert "E7" in results[0].page_content


def test_hybrid_search_sees_newly_ingested_chunks(store):
    chunk_and_store(MANUAL[:2])
    search("E7", k=1, mode="hybrid")

    chunk_and_store([doc("Firmware updates install between 02:00 and 04:00.", "manual.md#5")])

    assert "Firmware" in search("firmware", k=1, mode="hybrid")[0].page_content


def test_search_on_empty_index_returns_nothing(store):
    assert search("anything", k=3, mode="hybrid") == []


def test_unknown_mode_is_rejected(store):
    with pytest.raises(ValueError, match="Unknown retrieval mode"):
        search("anything", k=3, mode="fuzzy")


def test_reranker_reorders_and_trims(store, monkeypatch):
    chunk_and_store(MANUAL)

    class FakeCrossEncoder:
        def predict(self, pairs):
            # Scores passages about cleaning highest
            return [1.0 if "Clean" in text else 0.0 for _, text in pairs]

    monkeypatch.setattr(retrieval, "get_reranker", lambda: FakeCrossEncoder())

    results = search("how do I clean it", k=2, mode="similarity", use_reranker=True)

    assert len(results) == 2
    assert results[0].page_content.startswith("Clean the display")
