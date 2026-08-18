"""
confidence.py
-------------
Multi-factor confidence estimator for Adaptive RAG.

Improvements in this version
-----------------------------
Very High level
    Scores >= 0.85 are now labelled "Very High" instead of being
    lumped together with "High" (>= 0.72).

keyword_coverage component
    Fraction of non-trivial query terms that appear in the selected
    context; measures topical relevance of retrieved evidence.

retrieval_consistency component
    1 - coefficient_of_variation of the similarity scores; measures
    how stable (agreement across) the retrieved evidence is.

question parameter
    Optional; enables keyword_coverage computation. Fully backward
    compatible - callers that do not pass ``question`` get the previous
    5-component confidence without keyword_coverage.

Component weights (updated)
    retrieval_similarity  0.26
    citation_coverage     0.18
    chunk_agreement       0.16
    cross_encoder         0.15
    verification          0.13
    keyword_coverage      0.07   (0.0 when question not provided)
    retrieval_consistency 0.05
"""

from __future__ import annotations

import re
from typing import Optional

from src.utils import extract_terms


class ConfidenceEstimator:
    """
    Calculates a multi-factor confidence score for a RAG answer.

    Parameters are injected per-call via :meth:`calculate`.
    """

    # Confidence level thresholds (inclusive lower bounds).
    _LEVEL_VERY_HIGH = 0.85
    _LEVEL_HIGH      = 0.72
    _LEVEL_MEDIUM    = 0.52

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate(
        self,
        retrieved_chunks: list[dict],
        citations: Optional[dict] = None,
        verification: Optional[dict] = None,
        question: str = "",
    ) -> dict:
        """
        Compute a normalized confidence score from multiple evidence signals.

        Parameters
        ----------
        retrieved_chunks : list[dict]
            Selected chunks sent to the LLM (post-BSCO).
        citations : dict, optional
            Output of :class:`CitationGenerator.generate`.
        verification : dict, optional
            Output of :class:`SelfVerifier.verify`.
        question : str, optional
            Original user question (enables keyword_coverage component).

        Returns
        -------
        dict
            Keys: ``score``, ``level``, ``components``, ``explanation``.
        """
        if not retrieved_chunks:
            return {
                "score": 0.0,
                "level": "Low",
                "components": self._empty_components(),
                "explanation": "No retrieved evidence was available.",
            }

        scores = [c["score"] for c in retrieved_chunks if "score" in c]
        if not scores:
            return {
                "score": 0.0,
                "level": "Low",
                "components": self._empty_components(),
                "explanation": "Retrieved chunks did not include usable scores.",
            }

        # -- Component computation ------------------------------------------
        retrieval_sim  = self._average(scores)
        citation_cov   = self._citation_coverage(retrieved_chunks, citations)
        chunk_agree    = self._chunk_agreement(retrieved_chunks)
        rerank_score   = self._average([
            c.get("cross_encoder_score", c.get("rerank_score", 0.0))
            for c in retrieved_chunks
        ])
        verif_score    = (verification or {}).get("score", 0.0)
        kw_coverage    = self._keyword_coverage(retrieved_chunks, question) if question else 0.0
        consistency    = self._retrieval_consistency(scores)

        # -- Weighted fusion ------------------------------------------------
        # Weights sum to 1.0 when question is provided; when not, the
        # keyword_coverage weight is redistributed proportionally.
        if question:
            composite = (
                0.26 * retrieval_sim
                + 0.18 * citation_cov
                + 0.16 * chunk_agree
                + 0.15 * rerank_score
                + 0.13 * verif_score
                + 0.07 * kw_coverage
                + 0.05 * consistency
            )
        else:
            # Normalise without keyword_coverage (weights sum to 0.93 -> rescale).
            composite = (
                0.30 * retrieval_sim
                + 0.20 * citation_cov
                + 0.18 * chunk_agree
                + 0.17 * rerank_score
                + 0.15 * verif_score
            )

        score = round(min(max(composite, 0.0), 1.0), 4)
        level = self._level(score)

        return {
            "score": score,
            "level": level,
            "components": {
                "retrieval_similarity":   round(retrieval_sim, 4),
                "citation_coverage":      round(citation_cov, 4),
                "chunk_agreement":        round(chunk_agree, 4),
                "cross_encoder":          round(rerank_score, 4),
                "verification":           round(verif_score, 4),
                "keyword_coverage":       round(kw_coverage, 4),
                "retrieval_consistency":  round(consistency, 4),
            },
            "explanation": (
                "Confidence combines retrieval similarity, citation coverage, "
                "agreement across selected chunks, reranking score, "
                "post-generation verification, keyword coverage, and "
                "retrieval consistency."
            ),
        }

    # ------------------------------------------------------------------
    # Individual components
    # ------------------------------------------------------------------

    def _citation_coverage(
        self,
        chunks: list[dict],
        citations: Optional[dict],
    ) -> float:
        """Fraction of selected chunks that appear on a cited page."""
        pages = set((citations or {}).get("pages", []))
        if not chunks:
            return 0.0
        cited = [c for c in chunks if c.get("page_number") in pages]
        return len(cited) / len(chunks)

    def _chunk_agreement(self, chunks: list[dict]) -> float:
        """
        Measure vocabulary overlap across selected chunks.

        High agreement means all chunks discuss the same topic, which
        increases confidence that the retrieved context is coherent.
        """
        if len(chunks) <= 1:
            return 1.0

        term_sets = [
            {w.lower() for w in chunk.get("text", "").split() if len(w) > 3}
            for chunk in chunks
        ]

        overlaps: list[float] = []
        for i, terms in enumerate(term_sets):
            others = set().union(*(term_sets[:i] + term_sets[i + 1:]))
            overlaps.append(len(terms & others) / max(len(terms), 1))

        return self._average(overlaps)

    def _keyword_coverage(
        self,
        chunks: list[dict],
        question: str,
    ) -> float:
        """
        Fraction of meaningful query terms that appear in the selected context.

        A high score means the retrieved chunks actually address the question's
        key topics; a low score indicates possible retrieval mismatch.
        """
        query_terms = extract_terms(question)
        if not query_terms:
            return 1.0

        context = " ".join(c.get("text", "") for c in chunks)
        context_terms = set(re.findall(r"[a-zA-Z0-9]+", context.lower()))
        return len(query_terms & context_terms) / len(query_terms)

    @staticmethod
    def _retrieval_consistency(scores: list[float]) -> float:
        """
        Return 1 - coefficient_of_variation, clamped to [0, 1].

        Measures how uniformly relevant the retrieved chunks are.
        A CV close to 0 means all chunks have similar scores (consistent);
        a high CV means scores are spread out (one or two outliers).
        """
        if len(scores) <= 1:
            return 1.0
        mean = sum(scores) / len(scores)
        if mean < 1e-9:
            return 0.0
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)
        std = variance ** 0.5
        cv = std / mean
        return round(max(0.0, min(1.0 - cv, 1.0)), 4)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _level(self, score: float) -> str:
        """Map a numeric score to a human-readable confidence level."""
        if score >= self._LEVEL_VERY_HIGH:
            return "Very High"
        if score >= self._LEVEL_HIGH:
            return "High"
        if score >= self._LEVEL_MEDIUM:
            return "Medium"
        return "Low"

    @staticmethod
    def _average(values: list[float]) -> float:
        """Mean of *values*, ignoring None; returns 0.0 for empty list."""
        clean = [v for v in values if v is not None]
        return sum(clean) / len(clean) if clean else 0.0

    @staticmethod
    def _empty_components() -> dict:
        return {
            "retrieval_similarity":  0.0,
            "citation_coverage":     0.0,
            "chunk_agreement":       0.0,
            "cross_encoder":         0.0,
            "verification":          0.0,
            "keyword_coverage":      0.0,
            "retrieval_consistency": 0.0,
        }

