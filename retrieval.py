"""Search strategies over the index: vector similarity, MMR, hybrid (BM25 keywords +
vectors), each optionally followed by a cross-encoder reranker."""
import math
import re
from collections import Counter, defaultdict
from functools import cache

import vectorstore
from config import RERANKER_MODEL

MODES = {
    "similarity": "Vector similarity",
    "mmr": "Vector + diversity (MMR)",
    "hybrid": "Hybrid (keywords + vectors)",
}
# Passages each search fetches before fusing or reranking down to k
MIN_CANDIDATES = 20
# Standard constant for reciprocal rank fusion; dampens the weight of top ranks
RRF_K = 60
# MMR trade-off: 1 is pure relevance, 0 is pure diversity
MMR_LAMBDA = 0.5


def tokenize(text):
    """Lowercased words and numbers, so IDs like "E7" or "INV-2041" stay searchable."""
    return re.findall(r"\w+", text.lower())


class BM25:
    """Okapi BM25 keyword search over a fixed list of documents."""

    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = docs
        self.k1, self.b = k1, b
        # Inverted index: term -> [(doc index, term frequency)]
        self.postings = defaultdict(list)
        self.doc_lens = []
        for i, doc in enumerate(docs):
            terms = Counter(tokenize(doc.page_content))
            self.doc_lens.append(sum(terms.values()))
            for term, freq in terms.items():
                self.postings[term].append((i, freq))
        self.avg_len = sum(self.doc_lens) / len(docs) if docs else 0
        n = len(docs)
        self.idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()}

    def search(self, query, k):
        scores = defaultdict(float)
        for term in set(tokenize(query)):
            for i, freq in self.postings.get(term, ()):
                norm = self.k1 * (1 - self.b + self.b * self.doc_lens[i] / self.avg_len)
                scores[i] += self.idf[term] * freq * (self.k1 + 1) / (freq + norm)
        best = sorted(scores, key=scores.get, reverse=True)[:k]
        return [self.docs[i] for i in best]


_bm25 = (None, None)  # (index version it was built for, BM25)

def get_bm25():
    """BM25 index over all chunks, rebuilt whenever the index changes."""
    global _bm25
    version = vectorstore.index_version()
    if _bm25[0] != version:
        _bm25 = (version, BM25(vectorstore.all_chunks()))
    return _bm25[1]


def doc_key(doc):
    return doc.id or vectorstore.chunk_id(doc)

def reciprocal_rank_fusion(*rankings, k=RRF_K):
    """Merges ranked lists: each document scores 1 / (k + rank) in every list it appears in."""
    scores = defaultdict(float)
    docs = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            key = doc_key(doc)
            scores[key] += 1 / (k + rank)
            docs.setdefault(key, doc)
    return [docs[key] for key in sorted(scores, key=scores.get, reverse=True)]


@cache
def get_reranker(model=RERANKER_MODEL):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(model)

def rerank(query, docs, k):
    """Reorders docs by a cross-encoder's relevance score and keeps the top k."""
    if not docs:
        return []
    scores = get_reranker().predict([(query, doc.page_content) for doc in docs])
    order = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
    return [docs[i] for i in order[:k]]


def search(query, k, mode="hybrid", use_reranker=False):
    """The k most relevant chunks for the query."""
    if mode not in MODES:
        raise ValueError(f"Unknown retrieval mode: {mode!r} (expected one of {', '.join(MODES)})")
    store = vectorstore.get_vectorstore()
    if store._collection.count() == 0:
        return []
    candidates = max(MIN_CANDIDATES, 4 * k)
    # With a reranker, retrieve a larger pool for it to choose from
    n = candidates if use_reranker else k

    if mode == "similarity":
        docs = store.similarity_search(query, k=n)
    elif mode == "mmr":
        docs = store.max_marginal_relevance_search(
            query, k=n, fetch_k=max(candidates, 2 * n), lambda_mult=MMR_LAMBDA
        )
    else:
        docs = reciprocal_rank_fusion(
            store.similarity_search(query, k=candidates),
            get_bm25().search(query, candidates),
        )[:n]

    return rerank(query, docs, k) if use_reranker else docs
