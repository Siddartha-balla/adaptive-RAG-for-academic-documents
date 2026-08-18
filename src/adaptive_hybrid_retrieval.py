"""
Adaptive hybrid retrieval for academic RAG.

The retriever combines dense FAISS search, BM25 lexical search, keyword
coverage, chunk-quality scoring, metadata filters, and page-diverse selection.
It returns the same flat chunk dictionaries used by the rest of the pipeline.

Performance improvements
------------------------
``_lexical_scores`` now only calculates BM25/keyword scores for the candidate
indices that already came back from dense search (plus a small BM25-only
head-room set), instead of re-scoring every chunk in the corpus on every query.
This removes the O(N) tokenization+scoring pass over all documents per query,
which was the dominant retrieval-stage cost on large corpora.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Any, Optional

from config import (
    HYBRID_BM25_WEIGHT,
    HYBRID_DENSE_WEIGHT,
    HYBRID_KEYWORD_WEIGHT,
    TOP_K,
)
from src.query_expander import QueryExpander
from src.utils import logger, tokenize

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.embedding import EmbeddingGenerator
    from src.vector_database import VectorDatabase


class BM25Index:
    """Small in-memory BM25 index over chunk text."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_freqs: list[Counter[str]] = []
        self.idf: dict[str, float] = {}
        self.doc_lengths: list[int] = []
        self.avg_doc_length = 0.0
        self.num_docs = 0
        self.vocabulary: set[str] = set()
        # Query-term cache: repeated identical queries avoid re-tokenizing.
        self._query_cache: dict[str, list[str]] = {}
        # Per-term, per-doc flip lookup: whether a term appears in a doc.
        # Built once after indexing so score() can skip absent terms fast.
        self._term_doc_flip: dict[str, set[int]] = {}

    def build_index(self, documents: list[str]) -> None:
        """Build the BM25 statistics for *documents*."""
        self.num_docs = len(documents)
        self.doc_freqs = []
        self.doc_lengths = []
        self.vocabulary = set()
        self._query_cache.clear()
        self._term_doc_flip = {}

        for document in documents:
            tokens = self._tokenize(document)
            self.doc_freqs.append(Counter(tokens))
            self.doc_lengths.append(len(tokens))
            self.vocabulary.update(tokens)

        self.avg_doc_length = (
            sum(self.doc_lengths) / self.num_docs if self.num_docs else 0.0
        )

        for term in self.vocabulary:
            document_frequency = sum(1 for freqs in self.doc_freqs if term in freqs)
            self.idf[term] = math.log(
                ((self.num_docs - document_frequency + 0.5)
                 / (document_frequency + 0.5))
                + 1
            )

        # Build inverted flip index (term -> set of doc indices containing it).
        for idx, freqs in enumerate(self.doc_freqs):
            for term in freqs:
                self._term_doc_flip.setdefault(term, set()).add(idx)

    def query_tokens(self, query: str) -> list[str]:
        """Tokenize *query*, caching the result per unique query string."""
        cached = self._query_cache.get(query)
        if cached is None:
            cached = self._tokenize(query)
            if len(self._query_cache) > 256:
                self._query_cache.clear()
            self._query_cache[query] = cached
        return cached

    def score(self, query: str, doc_idx: int) -> float:
        """Return the BM25 score for *query* against document *doc_idx*."""
        if doc_idx < 0 or doc_idx >= self.num_docs or self.avg_doc_length <= 0:
            return 0.0

        score = 0.0
        doc_freq = self.doc_freqs[doc_idx]
        doc_length = self.doc_lengths[doc_idx]

        for token in self.query_tokens(query):
            if token not in doc_freq:
                continue
            term_frequency = doc_freq[token]
            denominator = term_frequency + self.k1 * (
                1 - self.b + self.b * doc_length / self.avg_doc_length
            )
            score += self.idf.get(token, 0.0) * (
                term_frequency * (self.k1 + 1) / max(denominator, 1e-9)
            )

        return score

    def search(self, query: str, top_k: int = TOP_K) -> list[tuple[int, float]]:
        """Return top ``(document_index, score)`` pairs for *query*."""
        scores = [(index, self.score(query, index)) for index in range(self.num_docs)]
        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[:top_k]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return tokenize(text)


class AdaptiveHybridRetrieval:
    """
    Multi-signal retriever with page-diverse candidate selection.

    Parameters are injectable so tests and Streamlit can share already-loaded
    embedding/vector database instances.
    """

    def __init__(
        self,
        embedder: Optional["EmbeddingGenerator"] = None,
        vector_db: Optional["VectorDatabase"] = None,
        dense_weight: float = HYBRID_DENSE_WEIGHT,
        bm25_weight: float = HYBRID_BM25_WEIGHT,
        keyword_weight: float = HYBRID_KEYWORD_WEIGHT,
        load_existing: bool = True,
    ) -> None:
        if embedder is None:
            from src.embedding import EmbeddingGenerator

            embedder = EmbeddingGenerator()
        if vector_db is None:
            from src.vector_database import VectorDatabase

            vector_db = VectorDatabase()

        self.embedder = embedder
        self.vector_db = vector_db
        self.expander = QueryExpander()
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self.keyword_weight = keyword_weight
        self.bm25_index: Optional[BM25Index] = None
        self._bm25_doc_count = 0
        self.page_stats: dict[Any, dict[str, Any]] = {}

        if load_existing:
            try:
                self.vector_db.load()
            except FileNotFoundError:
                logger.info("No vector database found yet. Process a PDF before searching.")

    def reload(self) -> None:
        """Reload the persisted vector database and rebuild sparse statistics."""
        self.vector_db.load()
        self.invalidate()
        self._build_bm25_index()
        self._calculate_page_stats()

    def invalidate(self) -> None:
        """Invalidate cached sparse/page statistics after database rebuild."""
        self.bm25_index = None
        self._bm25_doc_count = 0
        self.page_stats = {}

    def hybrid_search(
        self,
        query: str,
        top_k: int = TOP_K,
        candidate_multiplier: int = 4,
        enable_bm25: bool = True,
        enable_keyword: bool = True,
        page_diversity: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Retrieve relevant chunks using dense, lexical, keyword, and page signals.
        """
        self._ensure_loaded()
        if not self.vector_db.metadata:
            return []

        expansion = self.expander.expand(query)
        expanded_query = expansion.expanded_query

        candidate_limit = min(
            max(top_k * candidate_multiplier, top_k),
            len(self.vector_db.metadata),
        )

        query_embedding = self.embedder.encode_query(expanded_query)
        dense_results = self.vector_db.search(query_embedding, candidate_limit)
        metadata_lookup = self._build_metadata_lookup()

        dense_scores: dict[int, float] = {}
        candidate_indices: set[int] = set()
        for result in dense_results:
            index = self._metadata_index(result, metadata_lookup)
            if index is None:
                continue
            dense_scores[index] = float(result.get("score", 0.0))
            candidate_indices.add(index)

        # Only compute lexical scores for the dense candidate set (plus a small
        # BM25-only head-room set) instead of re-scoring the whole corpus.
        lexical_scores = self._lexical_scores(
            expanded_query,
            target_indices=candidate_indices,
            enable_bm25=enable_bm25,
            enable_keyword=enable_keyword,
        )
        lexical_ranked = sorted(
            lexical_scores.items(),
            key=lambda item: item[1]["bm25"] + item[1]["keyword"],
            reverse=True,
        )
        for index, scores in lexical_ranked[: candidate_limit * 2]:
            if scores["bm25"] > 0 or scores["keyword"] > 0:
                candidate_indices.add(index)

        if not candidate_indices:
            return dense_results[:top_k]

        self._calculate_page_stats()
        weights = self._adaptive_weights(query)
        max_dense = max(dense_scores.values(), default=1.0)
        max_bm25 = max((scores["bm25"] for scores in lexical_scores.values()), default=1.0)

        fused = []
        for index in candidate_indices:
            chunk = self.vector_db.metadata[index].copy()
            lexical = lexical_scores.get(index, {"bm25": 0.0, "keyword": 0.0})

            dense_norm = self._clamp01(dense_scores.get(index, 0.0) / max(max_dense, 1e-9))
            bm25_norm = self._clamp01(lexical["bm25"] / max(max_bm25, 1e-9))
            keyword_norm = self._clamp01(lexical["keyword"])
            quality = self._calculate_chunk_quality(chunk)

            hybrid = (
                weights["dense"] * dense_norm
                + weights["bm25"] * bm25_norm
                + weights["keyword"] * keyword_norm
            )
            hybrid *= 0.85 + (0.15 * quality)
            normalized = self._page_normalized_score(chunk, hybrid) if page_diversity else hybrid

            chunk.update({
                "metadata_index": index,
                "dense_score": round(dense_norm, 4),
                "bm25_score": round(bm25_norm, 4),
                "keyword_score": round(keyword_norm, 4),
                "quality_score": round(quality, 4),
                "hybrid_score": round(hybrid, 4),
                "normalized_score": round(normalized, 4),
                "score": round(hybrid, 4),
                "retrieval_strategy": "adaptive_hybrid",
            })
            fused.append(chunk)

        fused.sort(key=lambda item: item["normalized_score"], reverse=True)
        if not page_diversity:
            return fused[:top_k]

        return self._select_diverse(fused, top_k)

    def metadata_filter_search(
        self,
        query: str,
        page_numbers: Optional[list[int]] = None,
        section_titles: Optional[list[str]] = None,
        top_k: int = TOP_K,
    ) -> list[dict[str, Any]]:
        """Run hybrid search and filter candidates by page or section metadata."""
        results = self.hybrid_search(query, top_k=top_k * 3)
        filtered = []

        for result in results:
            if page_numbers is not None and result.get("page_number") not in page_numbers:
                continue
            if section_titles is not None:
                section = (result.get("section_title") or "").lower()
                if not any(title.lower() in section for title in section_titles):
                    continue
            filtered.append(result)

        return filtered[:top_k]

    def _ensure_loaded(self) -> None:
        if self.vector_db.index is None or self.vector_db.metadata is None:
            self.vector_db.load()

    def _build_bm25_index(self) -> None:
        metadata = self.vector_db.metadata or []
        if self.bm25_index is not None and self._bm25_doc_count == len(metadata):
            return

        documents = [chunk.get("text", "") for chunk in metadata]
        self.bm25_index = BM25Index()
        self.bm25_index.build_index(documents)
        self._bm25_doc_count = len(documents)

    def _lexical_scores(
        self,
        query: str,
        *,
        target_indices: Optional[set[int]] = None,
        enable_bm25: bool,
        enable_keyword: bool,
    ) -> dict[int, dict[str, float]]:
        metadata = self.vector_db.metadata or []
        if not metadata:
            return {}

        if enable_bm25:
            self._build_bm25_index()

        query_terms = set(self._tokenize(query))

        # When we already have a dense candidate set, only score those indices
        # (plus a small head-room set from the BM25 index) instead of the whole
        # corpus. This is the key performance win for large corpora.
        if target_indices:
            candidate_list = list(target_indices)
            headroom = 0
            if enable_bm25 and self.bm25_index is not None:
                headroom = max(8, len(candidate_list) // 2)
                bm25_top = self.bm25_index.search(query, top_k=headroom)
                candidate_list.extend(i for i, _ in bm25_top if i not in target_indices)
            indices = candidate_list
        else:
            indices = list(range(len(metadata)))

        scores: dict[int, dict[str, float]] = {}
        for index in indices:
            if index < 0 or index >= len(metadata):
                continue
            chunk = metadata[index]
            text = chunk.get("text", "")
            chunk_terms = set(self._tokenize(text))
            keyword = (
                len(query_terms & chunk_terms) / max(len(query_terms), 1)
                if enable_keyword and query_terms
                else 0.0
            )
            bm25 = self.bm25_index.score(query, index) if enable_bm25 and self.bm25_index else 0.0
            scores[index] = {"bm25": bm25, "keyword": keyword}

        return scores

    def _calculate_page_stats(self) -> None:
        metadata = self.vector_db.metadata or []
        stats: dict[Any, dict[str, Any]] = defaultdict(lambda: {"chunk_count": 0})
        for chunk in metadata:
            page = chunk.get("page_number", 0)
            stats[page]["chunk_count"] += 1
        self.page_stats = dict(stats)

    def _page_normalized_score(self, chunk: dict[str, Any], score: float) -> float:
        page = chunk.get("page_number", 0)
        chunk_count = self.page_stats.get(page, {}).get("chunk_count", 1)
        page_factor = 1.0 / (1.0 + math.log(max(chunk_count, 1)))
        return score * (0.80 + 0.20 * page_factor)

    def _select_diverse(
        self,
        candidates: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Select candidates iteratively with a same-page diversity penalty."""
        selected: list[dict[str, Any]] = []
        remaining = [candidate.copy() for candidate in candidates]
        page_counts: Counter[Any] = Counter()

        while remaining and len(selected) < top_k:
            best_position = 0
            best_score = -1.0
            for position, candidate in enumerate(remaining):
                page = candidate.get("page_number", 0)
                penalty = 1.0 / (1.0 + 0.45 * page_counts[page])
                final_score = candidate["normalized_score"] * penalty
                if final_score > best_score:
                    best_score = final_score
                    best_position = position

            chosen = remaining.pop(best_position)
            chosen["final_score"] = round(best_score, 4)
            page_counts[chosen.get("page_number", 0)] += 1
            selected.append(chosen)

        selected.sort(key=lambda item: item.get("final_score", 0.0), reverse=True)
        return selected

    def _build_metadata_lookup(self) -> dict[tuple[Any, Any, Any], int]:
        lookup: dict[tuple[Any, Any, Any], int] = {}
        for index, chunk in enumerate(self.vector_db.metadata or []):
            lookup[self._chunk_key(chunk)] = index
        return lookup

    @staticmethod
    def _metadata_index(
        chunk: dict[str, Any],
        lookup: dict[tuple[Any, Any, Any], int],
    ) -> Optional[int]:
        return lookup.get(AdaptiveHybridRetrieval._chunk_key(chunk))

    @staticmethod
    def _chunk_key(chunk: dict[str, Any]) -> tuple[Any, Any, Any]:
        return (
            chunk.get("chunk_id"),
            chunk.get("page_number"),
            chunk.get("source_file"),
        )

    @staticmethod
    def _calculate_chunk_quality(chunk: dict[str, Any]) -> float:
        text = chunk.get("text", "")
        length = len(text)
        if 200 <= length <= 900:
            length_score = 1.0
        elif length < 200:
            length_score = max(length / 200, 0.25)
        else:
            length_score = max(0.45, 1.0 - ((length - 900) / 1400))

        structure_score = 1.0 if chunk.get("section_title") else 0.8
        academic_type_score = 1.0 if chunk.get("chunk_type") in {"paragraph", "table"} else 0.9
        return (0.60 * length_score) + (0.25 * structure_score) + (0.15 * academic_type_score)

    @staticmethod
    def _adaptive_weights(query: str) -> dict[str, float]:
        lowered = query.lower()
        dense = HYBRID_DENSE_WEIGHT
        bm25 = HYBRID_BM25_WEIGHT
        keyword = HYBRID_KEYWORD_WEIGHT

        if re.search(r"\b(compare|survey|literature|research gap|novel|future work)\b", lowered):
            dense += 0.08
            bm25 -= 0.04
            keyword -= 0.04
        elif re.search(r"\b(define|what is|formula|equation|list|enumerate|exact)\b", lowered):
            bm25 += 0.06
            keyword += 0.04
            dense -= 0.10

        total = max(dense + bm25 + keyword, 1e-9)
        return {
            "dense": dense / total,
            "bm25": bm25 / total,
            "keyword": keyword / total,
        }

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return tokenize(text)

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(float(value), 1.0))
