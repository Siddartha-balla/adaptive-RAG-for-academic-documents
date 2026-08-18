"""
citations.py
------------
Rich citation generation for retrieved document chunks.

Improvements in this version
-----------------------------
Per-chunk detail
    Each citation now records page number, chunk index, section heading,
    similarity score, matched keyword set, and a highlighted evidence snippet
    (the highest-scoring sentence from the chunk text).

Multiple citations
    ``generate()`` returns a ``chunks`` list with one entry per selected
    chunk so the UI can render individual citation cards, not just page numbers.

Backward compatibility
    All original keys (``pages``, ``citation_text``, ``total_pages``) are
    preserved so existing code does not break.
"""

from __future__ import annotations

import re
from typing import Optional

from src.utils import extract_terms


class CitationGenerator:
    """
    Generates rich, per-chunk citations from retrieved document chunks.

    The ``generate`` method is the primary API; all internal helpers are
    private and may change without notice.
    """

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        chunks: list[dict],
        query: Optional[str] = None,
    ) -> dict:
        """
        Build a complete citation dictionary from *chunks*.

        Parameters
        ----------
        chunks : list[dict]
            Selected chunks from the RAG pipeline (post-BSCO).
        query : str, optional
            The original user question, used to compute matched keywords.

        Returns
        -------
        dict
            Keys:
            - ``pages``          — sorted list of unique page numbers (legacy)
            - ``citation_text``  — human-readable page list string (legacy)
            - ``total_pages``    — number of unique pages cited (legacy)
            - ``chunks``         — list of per-chunk citation dicts (new)
            - ``total_chunks``   — total number of cited chunks (new)
        """
        pages = self._unique_pages(chunks)
        chunk_citations = [
            self._build_chunk_citation(chunk, idx, query)
            for idx, chunk in enumerate(chunks, start=1)
        ]

        return {
            # ── Legacy keys (backward compatible) ──────────────────────
            "pages": pages,
            "citation_text": self._format_page_list(pages),
            "total_pages": len(pages),
            # ── Rich per-chunk detail ──────────────────────────────────
            "chunks": chunk_citations,
            "total_chunks": len(chunk_citations),
        }

    def get_page_numbers(self, chunks: list[dict]) -> list[int]:
        """Return a sorted list of unique page numbers (legacy helper)."""
        return self._unique_pages(chunks)

    def format_citations(self, chunks: list[dict]) -> str:
        """Return citations as ``'Page 10, Page 12, …'`` (legacy helper)."""
        return self._format_page_list(self._unique_pages(chunks))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _unique_pages(chunks: list[dict]) -> list[int]:
        """Sorted unique page numbers from *chunks*."""
        return sorted({
            chunk["page_number"]
            for chunk in chunks
            if "page_number" in chunk
        })

    @staticmethod
    def _format_page_list(pages: list[int]) -> str:
        """Format page list as ``'Page 10, Page 12'`` or a not-found message."""
        if not pages:
            return "No citations available."
        return ", ".join(f"Page {p}" for p in pages)

    # ------------------------------------------------------------------
    # Per-chunk citation builder
    # ------------------------------------------------------------------

    def _build_chunk_citation(
        self,
        chunk: dict,
        rank: int,
        query: Optional[str] = None,
    ) -> dict:
        """
        Build a rich citation dictionary for a single *chunk*.

        Parameters
        ----------
        chunk : dict
            A single retrieved / selected chunk.
        rank : int
            1-based rank within the selected context (for display).
        query : str, optional
            Original user question used to find matched keywords.

        Returns
        -------
        dict
            Rich citation with the following keys:
            ``rank``, ``page``, ``chunk_id``, ``section``, ``score``,
            ``rerank_score``, ``matched_keywords``, ``evidence``,
            ``source_file``, ``chunk_type``.
        """
        text = chunk.get("text", "")
        score = chunk.get("score", 0.0)
        rerank_score = chunk.get("rerank_score") or chunk.get("cross_encoder_score")
        section = chunk.get("section_title") or "Unknown section"
        source = chunk.get("source_file", "Uploaded PDF")
        chunk_type = chunk.get("chunk_type", "paragraph")

        # ── Matched keywords ─────────────────────────────────────────
        matched_kw = self._matched_keywords(text, query) if query else []

        # ── Evidence snippet ──────────────────────────────────────────
        # Pick the sentence from the chunk that contains the most
        # query term matches (or just the first sentence if no query).
        evidence = self._best_evidence_sentence(text, query)

        return {
            "rank": rank,
            "page": chunk.get("page_number"),
            "chunk_id": chunk.get("chunk_id"),
            "section": section,
            "score": round(score, 4),
            "rerank_score": round(rerank_score, 4) if rerank_score is not None else None,
            "matched_keywords": matched_kw,
            "evidence": evidence,
            "source_file": source,
            "chunk_type": chunk_type,
        }

    # ------------------------------------------------------------------
    # Keyword matching — uses shared extract_terms from utils
    # ------------------------------------------------------------------

    def _matched_keywords(self, chunk_text: str, query: str) -> list[str]:
        """
        Return query terms that appear in *chunk_text*, sorted alphabetically.
        """
        query_terms = extract_terms(query)
        chunk_terms = extract_terms(chunk_text)
        matched = sorted(query_terms & chunk_terms)
        return matched[:10]  # cap at 10 for display

    # ------------------------------------------------------------------
    # Evidence sentence extraction
    # ------------------------------------------------------------------

    def _best_evidence_sentence(
        self,
        text: str,
        query: Optional[str] = None,
        max_length: int = 250,
    ) -> str:
        """
        Extract the most query-relevant sentence from *text*.

        Strategy:
        1. Split text into sentences.
        2. Score each sentence by how many query terms it contains.
        3. Return the highest-scoring sentence, truncated to *max_length*.
        4. Fall back to the first sentence if no query is given.

        Parameters
        ----------
        text : str
            Chunk text.
        query : str, optional
            Original user question.
        max_length : int
            Maximum character length of the returned snippet.
        """
        if not text.strip():
            return ""

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if not sentences:
            return text[:max_length].strip()

        if not query:
            best = sentences[0]
            return best[:max_length].strip()

        query_terms = extract_terms(query)
        best_sentence = sentences[0]
        best_score = -1

        for sentence in sentences:
            sentence_terms = extract_terms(sentence)
            score = len(query_terms & sentence_terms)
            if score > best_score:
                best_score = score
                best_sentence = sentence

        # Truncate cleanly at a word boundary.
        if len(best_sentence) > max_length:
            best_sentence = best_sentence[:max_length].rsplit(" ", 1)[0] + "…"

        return best_sentence
