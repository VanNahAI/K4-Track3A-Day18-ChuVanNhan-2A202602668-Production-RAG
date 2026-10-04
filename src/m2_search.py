from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os, sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K, RRF_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def _validate_top_k(top_k: int) -> None:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
        raise ValueError("top_k must be a non-negative integer")


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    from underthesea import word_tokenize

    return word_tokenize(text, format="text").replace("_", " ")


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        from rank_bm25 import BM25Okapi

        tokens = [segment_vietnamese(c["text"].lower()).split() for c in chunks]
        bm25 = BM25Okapi(tokens) if any(tokens) else None
        self.documents = list(chunks)
        self.corpus_tokens = tokens
        self.bm25 = bm25

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        _validate_top_k(top_k)
        if self.bm25 is None or top_k == 0 or not query.strip():
            return []
        scores = self.bm25.get_scores(segment_vietnamese(query.lower()).split())
        indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [SearchResult(text=self.documents[i]["text"], score=float(scores[i]),
                             metadata=dict(self.documents[i].get("metadata", {})), method="bm25")
                for i in indices if scores[i] > 0][:top_k]


class DenseSearch:
    def __init__(self):
        from qdrant_client import QdrantClient
        try:
            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=2)
            self.client.get_collections()
        except Exception:
            self.client = QdrantClient(":memory:")
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(EMBEDDING_MODEL)
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        from qdrant_client.models import Distance, VectorParams, PointStruct

        points = []
        if chunks:
            vectors = self._get_encoder().encode([c["text"] for c in chunks], show_progress_bar=True)
            if len(vectors) != len(chunks) or any(len(v) != EMBEDDING_DIM for v in vectors):
                raise ValueError(f"Encoder must return one {EMBEDDING_DIM}-dimensional vector per chunk")
            points = [PointStruct(id=i, vector=v.tolist(),
                                  payload={**c.get("metadata", {}), "text": c["text"]})
                      for i, (c, v) in enumerate(zip(chunks, vectors))]
        if self.client.collection_exists(collection):
            self.client.delete_collection(collection)
        self.client.create_collection(collection, vectors_config=VectorParams(
            size=EMBEDDING_DIM, distance=Distance.COSINE))
        if points:
            self.client.upsert(collection, points=points, wait=True)

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        _validate_top_k(top_k)
        if top_k == 0 or not query.strip() or not self.client.collection_exists(collection):
            return []
        query_vector = self._get_encoder().encode(query).tolist()
        if len(query_vector) != EMBEDDING_DIM:
            raise ValueError(f"Query vector must have {EMBEDDING_DIM} dimensions")
        response = self.client.query_points(collection, query=query_vector,
                                            limit=top_k, with_payload=True)
        return [SearchResult(text=point.payload["text"], score=float(point.score),
                             metadata=dict(point.payload), method="dense")
                for point in response.points if point.payload and "text" in point.payload]


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = RRF_K,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank + 1)."""
    _validate_top_k(top_k)
    if isinstance(k, bool) or not isinstance(k, int) or k < 0:
        raise ValueError("k must be a non-negative integer")
    fused = {}
    for results in results_list:
        seen = set()
        for rank, result in enumerate(results):
            if result.text in seen:
                continue
            seen.add(result.text)
            if result.text not in fused:
                fused[result.text] = SearchResult(result.text, 0.0, dict(result.metadata), "hybrid")
            fused[result.text].score += 1.0 / (k + rank + 1)
    return sorted(fused.values(), key=lambda result: result.score, reverse=True)[:top_k]


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
