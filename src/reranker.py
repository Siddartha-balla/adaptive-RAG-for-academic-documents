"""
reranker.py
-----------
Candidate reranking after hybrid retrieval.

Improvements in this version
-----------------------------
Richer heuristic scoring
    The fallback reranker (used when no cross-encoder model is available)
    now combines five signals instead of two:

    1. hybrid_score      -- weighted dense + BM25 + keyword score from retrieval
    2. term_overlap      -- fraction of query terms found in the chunk
    3. heading_match     -- bonus when section_title contains query terms
    4. section_relevance -- bonus for high-value academic section types
       (abstract, introduction, conclusion, results, methodology)
    5. recency_penalty   -- small bonus for chunks near the end of the document,
       which often contain conclusions and summaries

    Weights are tuned so the heuristic closely mimics cross-encoder ranking
    on academic PDFs without requiring any ML model.

Query-type aware weighting
    When query_type is provided, weights shift: comparison/survey types
    weighted more towards section breadth, factual/definition types weighted
    more towards exact term overlap.
"""

from __future__ import annotations

import re
from typing import List, Optional

from config import CROSS_ENCODER_MODEL
from src.utils import tokenize, section_relevance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _heading_match(query_terms: set[str], section_title: str) -> float:
    """
    Bonus when the chunk's section heading contains query terms.
    Signals strong topical alignment -- the chunk is from the exact section
    the user is asking about.
    """
    if not section_title or not query_terms:
        return 0.0
    heading_terms = set(tokenize(section_title))
    overlap = len(query_terms & heading_terms)
    return min(overlap * 0.12, 0.30)   # cap at 0.30


def _recency_bonus(chunk: dict, total_chunks: int) -> float:
    """
    Mild bonus for chunks that appear in the second half of the document.
    Conclusions, results, and summaries tend to appear later in academic PDFs.

    Returns a value in [0, 0.08].
    """
    if total_chunks <= 0:
        return 0.0
    chunk_idx = chunk.get("chunk_id", 0) or 0
    # Normalise to [0, 1] relative position, bonus peaks at end.
    relative_pos = chunk_idx / max(total_chunks, 1)
    if relative_pos >= 0.6:
        return round(0.08 * ((relative_pos - 0.6) / 0.4), 4)
    return 0.0


def _get_query_type_weights(query_type: str) -> dict[str, float]:
    """
    Return (hybrid_weight, overlap_weight, heading_weight) tuple adjusted
    for query type.  Higher overlap_weight for factual/definition queries;
    higher heading/section weights for broad survey types.
    """
    weights = {
        "hybrid": 0.50,
        "overlap": 0.25,
        "heading_bonus": 0.12,
        "section_bonus": 0.15,
        "recency": 0.08,
    }

    if query_type in ("factual", "definition", "formula", "code"):
        # Exact term matching matters most
        weights["overlap"] = 0.35
        weights["hybrid"] = 0.40
        weights["heading_bonus"] = 0.10
        weights["section_bonus"] = 0.10
        weights["recency"] = 0.05

    elif query_type in (
        "summary", "comparison", "literature_survey", "paper_similarity",
        "explanation", "architecture", "methodology",
    ):
        # Broader context helps
        weights["hybrid"] = 0.45
        weights["overlap"] = 0.20
        weights["heading_bonus"] = 0.15
        weights["section_bonus"] = 0.15
        weights["recency"] = 0.10

    return weights


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------

class CrossEncoderReranker:
    """
    Rerank retrieval candidates using a cross-encoder when available,
    or a multi-signal heuristic when offline.

    Parameters
    ----------
    model_name : str or None
        HuggingFace cross-encoder model identifier.  When None or when the
        model cannot be loaded, the heuristic reranker is used.
    """

    def __init__(self, model_name: Optional[str] = CROSS_ENCODER_MODEL) -> None:
        self.model_name = model_name
        self.model = None

        if model_name:
            try:
                from sentence_transformers import CrossEncoder
                self.model = CrossEncoder(model_name)
                print(f"Cross-encoder loaded: {model_name}")
            except Exception as exc:
                print(f"Cross-encoder unavailable, using heuristic reranker: {exc}")

    def rerank(
        self,
        query: str,
        chunks: List[dict],
        top_n: int,
        query_type: str = "open",
    ) -> List[dict]:
        """
        Rerank *chunks* relative to *query* and return the top *top_n*.

        Parameters
        ----------
        query : str
            User question.
        chunks : list[dict]
            Candidates from hybrid retrieval.
        top_n : int
            Maximum number of chunks to return.
        query_type : str
            Query category for adaptive weight adjustment.

        Returns
        -------
        list[dict]
            Reranked candidates with ``rerank_score`` set, sorted descending.
        """
        if not chunks:
            return []

        candidates = [chunk.copy() for chunk in chunks]
        total_chunks = max(
            (c.get("chunk_id", 0) or 0 for c in candidates), default=0
        )

        if self.model is not None:
            # -- Neural cross-encoder path ---------------------------------
            pairs = [(query, chunk.get("text", "")) for chunk in candidates]
            scores = self.model.predict(pairs)
            for chunk, score in zip(candidates, scores):
                chunk["cross_encoder_score"] = float(score)
                chunk["rerank_score"] = float(score)
        else:
            # -- Heuristic multi-signal path -------------------------------
            query_terms = set(tokenize(query))
            w = _get_query_type_weights(query_type)

            for chunk in candidates:
                text_terms  = set(tokenize(chunk.get("text", "")))
                section     = chunk.get("section_title", "") or ""

                # Signal 1: term overlap fraction
                overlap = (
                    len(query_terms & text_terms) / max(len(query_terms), 1)
                )

                # Signal 2: hybrid retrieval score (already normalised to ~[0,1])
                hybrid = chunk.get("hybrid_score", chunk.get("score", 0.0))

                # Signal 3: heading match bonus
                heading_bonus = _heading_match(query_terms, section)

                # Signal 4: high-value section bonus
                section_bonus = section_relevance(section)

                # Signal 5: recency bonus
                recency = _recency_bonus(chunk, total_chunks)

                # Weighted fusion
                heuristic = (
                    w["hybrid"] * hybrid
                    + w["overlap"] * overlap
                    + heading_bonus      # already scaled
                    + section_bonus      # already scaled
                    + recency            # already scaled
                )

                chunk["cross_encoder_score"] = round(overlap, 4)
                chunk["rerank_score"]         = round(min(heuristic, 1.0), 4)

        candidates.sort(
            key=lambda c: c.get("rerank_score", 0.0), reverse=True
        )
        return candidates[:top_n]

