"""
Research Gap Detector
---------------------
Automatically identifies research gaps, limitations, and open problems
from academic document chunks.

This module analyzes retrieved chunks to surface:
- Limitations explicitly stated in the papers
- Missing comparisons or unexplored aspects
- Future work directions and open problems
- Methodological gaps (e.g., small datasets, limited evaluation)
- Contradictions or tensions between papers
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set

LIMITATION_PATTERNS = [
    r"\b(limitation|drawback|weakness|shortcoming|disadvantage)\b",
    r"\b(limited to|restricted|confined|scope is limited)\b",
    r"\b(not considered|not evaluated|not tested|not explored)\b",
    r"\b(does not address|fails to|unable to|cannot handle)\b",
    r"\b(lack of|lacks|missing|absence of)\b",
    r"\b(small dataset|small sample|limited data)\b",
    r"\b(computational cost|high complexity|expensive)\b",
    r"\b(future work|future research|further investigation)\b",
    r"\b(open problem|open question|challenge|unresolved)\b",
    r"\b(assumption|simplif|caveat|trade.off)\b",
    r"\b(need for|requires further|needs improvement)\b",
]

GAP_PATTERNS = [
    r"\b(research gap|gap in|not studied|unexplored)\b",
    r"\b(little work|limited research|few studies)\b",
    r"\b(not yet|yet to be|remains to be)\b",
    r"\b(beyond the scope|outside the scope)\b",
    r"\b(potential improvement|possible extension)\b",
    r"\b(future direction|next step|open issue)\b",
    r"\b(insufficient|inadequate|suboptimal)\b",
]


class ResearchGapDetector:
    """
    Identifies research gaps, limitations, and open problems from
    academic document chunks.

    Uses pattern matching and ranking to surface the most significant gaps.
    """

    def __init__(self) -> None:
        self._limitation_patterns = [
            re.compile(p, re.IGNORECASE) for p in LIMITATION_PATTERNS
        ]
        self._gap_patterns = [
            re.compile(p, re.IGNORECASE) for p in GAP_PATTERNS
        ]

    def analyze(
        self,
        chunks: List[Dict[str, Any]],
        all_chunks: Optional[List[Dict[str, Any]]] = None,
        question: str = "",
    ) -> Dict[str, Any]:
        """
        Unified analysis method called by pipeline.py.

        Delegates to :meth:`detect_gaps` internally. When the retrieved
        chunks cover fewer than two distinct papers and the full corpus
        (``all_chunks``) is available, the analysis falls back to the full
        corpus so cross-paper gaps are not missed.

        Parameters
        ----------
        chunks : list[dict]
            Retrieved document chunks (post-retrieval).
        all_chunks : list[dict], optional
            All chunks from all uploaded papers (for cross-paper analysis).
        question : str
            Original user question.

        Returns
        -------
        dict
            Same structure as :meth:`detect_gaps`.
        """
        effective = chunks or []
        if all_chunks and self._paper_count(effective) < 2:
            effective = all_chunks
        return self.detect_gaps(chunks=effective, question=question)

    @staticmethod
    def _paper_count(chunks: List[Dict[str, Any]]) -> int:
        """Return the number of distinct source papers in *chunks*."""
        return len(dict.fromkeys(
            c.get("source_file", "Unknown") for c in chunks
        ))

    def suggest_future_work(
        self,
        chunks: List[Dict[str, Any]],
        question: str = "",
    ) -> Dict[str, Any]:
        """
        Suggest future work directions from document chunks.

        Called by pipeline.py when query_type is ``future_work``.

        Parameters
        ----------
        chunks : list[dict]
            Retrieved document chunks.
        question : str
            Original user question.

        Returns
        -------
        dict
            ``future_work`` — list of future work directions.
            ``gaps`` — related research gaps.
            ``future_work_score`` — significance score [0, 1].
            ``summary`` — text summary.
        """
        base = self.detect_gaps(chunks=chunks, question=question)
        return {
            "future_work": base.get("future_work", []),
            "gaps": base.get("gaps", []),
            "future_work_score": base.get("gap_score", 0.0),
            "summary": "Future work analysis: " + (
                base["future_work"][0]["text"][:200]
                if base.get("future_work")
                else "No explicit future work directions found in the retrieved context."
            ),
        }

    def detect_gaps(
        self,
        chunks: List[Dict[str, Any]],
        question: str = "",
    ) -> Dict[str, Any]:
        """
        Detect research gaps from a list of document chunks.

        Parameters
        ----------
        chunks : list[dict]
            Retrieved document chunks.
        question : str
            Original user question (used to focus gap detection).

        Returns
        -------
        dict
            ``gaps`` — list of identified gaps with details.
            ``limitations`` — list of limitations found.
            ``future_work`` — future work directions mentioned.
            ``gap_score`` — overall gap significance score [0, 1].
            ``summary`` — text summary of detected gaps.
        """
        if not chunks:
            return self._empty_result()

        all_text = " ".join(c.get("text", "") for c in chunks)

        # Extract limitation statements
        limitations = self._extract_limitations(chunks, all_text)

        # Extract gap statements  
        gaps = self._extract_gaps(chunks, all_text)

        # Extract future work directions
        future_work = self._extract_future_work(chunks, all_text)

        # Score gap significance
        gap_score = self._score_gaps(limitations, gaps, future_work, chunks)

        # Generate summary
        summary = self._generate_summary(gaps, limitations, future_work)

        return {
            "gaps": gaps,
            "limitations": limitations,
            "future_work": future_work,
            "gap_score": round(gap_score, 4),
            "total_gaps_found": len(gaps) + len(limitations),
            "summary": summary,
        }

    def _extract_limitations(
        self,
        chunks: List[Dict[str, Any]],
        all_text: str,
    ) -> List[Dict[str, Any]]:
        """Extract limitation statements from chunks."""
        limitations = []
        seen: Set[str] = set()

        for chunk in chunks:
            text = chunk.get("text", "")
            sentences = re.split(r"(?<=[.!?])\s+", text)

            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 20:
                    continue

                normalized = sentence.lower()
                for pattern in self._limitation_patterns:
                    if pattern.search(normalized):
                        key = sentence[:100].lower()
                        if key not in seen:
                            seen.add(key)
                            limitations.append({
                                "text": sentence,
                                "page": chunk.get("page_number"),
                                "source": chunk.get("source_file"),
                                "matched_pattern": pattern.pattern[:40],
                                "type": "limitation",
                            })
                        break

        return limitations

    def _extract_gaps(
        self,
        chunks: List[Dict[str, Any]],
        all_text: str,
    ) -> List[Dict[str, Any]]:
        """Extract research gap statements from chunks."""
        gaps = []
        seen: Set[str] = set()

        for chunk in chunks:
            text = chunk.get("text", "")
            sentences = re.split(r"(?<=[.!?])\s+", text)

            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 15:
                    continue

                normalized = sentence.lower()
                for pattern in self._gap_patterns:
                    if pattern.search(normalized):
                        key = sentence[:100].lower()
                        if key not in seen:
                            seen.add(key)
                            gaps.append({
                                "text": sentence,
                                "page": chunk.get("page_number"),
                                "source": chunk.get("source_file"),
                                "matched_pattern": pattern.pattern[:40],
                                "type": "research_gap",
                            })
                        break

        return gaps

    def _extract_future_work(
        self,
        chunks: List[Dict[str, Any]],
        all_text: str,
    ) -> List[Dict[str, Any]]:
        """Extract future work directions from chunks."""
        future_work = []
        seen: Set[str] = set()

        future_patterns = [
            r"\b(future work|future research|future direction|future scope)\b",
            r"\b(next step|further research|ongoing work)\b",
            r"\b(can be extended|could be explored|planned to)\b",
            r"\b(would be interesting|should investigate|worth exploring)\b",
        ]
        compiled = [re.compile(p, re.IGNORECASE) for p in future_patterns]

        for chunk in chunks:
            text = chunk.get("text", "")
            sentences = re.split(r"(?<=[.!?])\s+", text)

            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 15:
                    continue

                normalized = sentence.lower()
                for pattern in compiled:
                    if pattern.search(normalized):
                        key = sentence[:100].lower()
                        if key not in seen:
                            seen.add(key)
                            future_work.append({
                                "text": sentence,
                                "page": chunk.get("page_number"),
                                "source": chunk.get("source_file"),
                                "type": "future_work",
                            })
                        break

        return future_work

    def _score_gaps(
        self,
        limitations: List[Dict],
        gaps: List[Dict],
        future_work: List[Dict],
        chunks: List[Dict],
    ) -> float:
        """Score the overall gap significance [0, 1]."""
        total_items = len(limitations) + len(gaps) + len(future_work)
        if not chunks or total_items == 0:
            return 0.0

        # Base score from count of gap-related items
        chunk_count = max(len(chunks), 1)
        coverage = min(total_items / (chunk_count * 2), 1.0)

        # Score is coverage weighted by confidence
        avg_score = sum(
            c.get("score", 0.5) for c in chunks
        ) / len(chunks)

        return coverage * avg_score

    def _generate_summary(
        self,
        gaps: List[Dict],
        limitations: List[Dict],
        future_work: List[Dict],
    ) -> str:
        """Generate a summary of detected gaps."""
        parts = []

        if limitations:
            parts.append(
                f"**{len(limitations)} limitation(s)** identified: "
                + "; ".join(l["text"][:120] for l in limitations[:3])
            )

        if gaps:
            parts.append(
                f"**{len(gaps)} research gap(s)** identified: "
                + "; ".join(g["text"][:120] for g in gaps[:3])
            )

        if future_work:
            parts.append(
                f"**{len(future_work)} future work direction(s)** mentioned: "
                + "; ".join(f["text"][:120] for f in future_work[:3])
            )

        if not parts:
            return "No significant research gaps or limitations detected in the retrieved context."

        return "\n\n".join(parts)

    def _empty_result(self) -> Dict[str, Any]:
        return {
            "gaps": [],
            "limitations": [],
            "future_work": [],
            "gap_score": 0.0,
            "total_gaps_found": 0,
            "summary": "No document context available for gap analysis.",
        }

