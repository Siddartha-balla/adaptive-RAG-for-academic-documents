"""
Post-generation grounding verification.

Improvements in this version (merged from research version)
------------------------------------------------------------
Hallucination detection
    verify() now computes a hallucination_score in [0, 1] that reflects the
    fraction of answer statements that lack any supporting evidence in the
    retrieved context.  Scores above 0.5 trigger a warning flag.

Statement extraction (advanced)
    The new _extract_statements() method parses the answer into atomic
    verifiable statements, enabling per-statement support tracking.
"""

from __future__ import annotations

import re
from typing import Any, List

from src.utils import extract_terms


class SelfVerifier:
    """Checks whether answer statements are supported by retrieved context."""

    INSUFFICIENT_MESSAGE = (
        "The uploaded academic document does not contain enough information "
        "to answer this question."
    )

    def verify(self, answer: str, chunks: List[dict]) -> dict:
        """
        Verify answer against retrieved chunks.

        Returns a dict with:
        - answer: grounded answer text (unsupported statements removed)
        - score: proportion of supported statements in [0, 1]
        - unsupported_statements: list of unsupported statement texts
        - supported_statements: count of supported statements
        - hallucination_score: proportion of statements without evidence
        - hallucination_flag: True when hallucination_score > 0.5
        """
        # Build context terms from chunks
        context = " ".join(chunk.get("text", "") for chunk in chunks)
        context_terms = extract_terms(context)

        if not answer.strip() or not context_terms:
            return {
                "answer": self.INSUFFICIENT_MESSAGE,
                "score": 0.0,
                "unsupported_statements": [],
                "supported_statements": 0,
                "hallucination_score": 1.0,
                "hallucination_flag": True,
            }

        statements = self._extract_statements(answer)

        if not statements:
            return {
                "answer": answer,
                "score": 1.0,
                "unsupported_statements": [],
                "supported_statements": 0,
                "hallucination_score": 0.0,
                "hallucination_flag": False,
            }

        kept = []
        unsupported = []

        for statement in statements:
            terms = extract_terms(statement)

            if not terms or statement.lower().startswith(("answer:", "supporting pages:", "confidence:")):
                kept.append(statement)
                continue

            coverage = len(terms & context_terms) / max(len(terms), 1)

            if coverage >= 0.34:
                kept.append(statement)
            else:
                unsupported.append(statement)

        factual_count = max(len(statements), 1)
        score = 1.0 - (len(unsupported) / factual_count)
        score = max(0.0, min(score, 1.0))

        grounded_answer = " ".join(kept).strip()

        if not grounded_answer:
            grounded_answer = self.INSUFFICIENT_MESSAGE

        hallucination_score = len(unsupported) / max(len(statements), 1)

        return {
            "answer": grounded_answer,
            "score": round(score, 4),
            "unsupported_statements": unsupported,
            "supported_statements": len(kept),
            "hallucination_score": round(hallucination_score, 4),
            "hallucination_flag": hallucination_score > 0.5,
        }

    def _extract_statements(self, text: str) -> list[str]:
        """
        Parse *text* into atomic verifiable statements.

        Splits on sentence boundaries and filters out format labels.
        """
        # Split on sentence-ending punctuation
        raw = re.split(r"(?<=[.!?])\s+", text)
        statements = []
        for s in raw:
            s = s.strip()
            if not s:
                continue
            # Skip format-only lines
            if s.lower().startswith(("answer:", "supporting pages:", "confidence:")):
                continue
            if len(s) < 10:
                continue
            statements.append(s)
        return statements
