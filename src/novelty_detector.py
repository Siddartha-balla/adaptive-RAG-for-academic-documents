"""
Novelty Detector
----------------
Identifies novel contributions, originality claims, and unique aspects
from academic document chunks.

Uses pattern matching and comparative analysis to detect:
- Explicit novelty/contribution claims
- Methodological novelty (new approaches, architectures)
- Empirical novelty (new datasets, benchmarks, results)
- Application novelty (new domains, use cases)
- Theoretical novelty (new frameworks, formalisms)
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, List, Set

NOVELTY_PATTERNS = [
    r"\b(novel|novelty|first|first-ever|first of its kind)\b",
    r"\b(contribution|contributions|original|originally)\b",
    r"\b(new approach|new method|new framework|new architecture)\b",
    r"\b(new dataset|new benchmark|new evaluation|new task)\b",
    r"\b(state.?of.?the.?art|sota|beyond.?the.?state.?of.?the.?art)\b",
    r"\b(to the best of our knowledge|as far as we know)\b",
    r"\b(unlike|different from|in contrast to|compared to existing)\b",
    r"\b(significant improvement|substantial improvement|remarkable)\b",
    r"\b(introduce[sd]? a novel|propose[sd]? a novel|present[sd]? a new)\b",
]

NOVELTY_TYPE_PATTERNS = {
    "methodological": [
        r"\b(method|approach|algorithm|technique|framework|architecture)\b",
        r"\b(design|pipeline|system|model|network)\b",
    ],
    "empirical": [
        r"\b(dataset|benchmark|corpus|experiment|evaluation)\b",
        r"\b(result|performance|accuracy|improvement)\b",
    ],
    "theoretical": [
        r"\b(theorem|lemma|proof|theory|theoretical|formalism)\b",
        r"\b(formulation|derivation|framework|foundation)\b",
    ],
    "application": [
        r"\b(application|domain|use case|task|scenario)\b",
        r"\b(deploy|real-world|practical|industry)\b",
    ],
}


class NoveltyDetector:
    """
    Detects and classifies novel contributions in academic document chunks.
    """

    def __init__(self) -> None:
        self._novelty_patterns = [
            re.compile(p, re.IGNORECASE) for p in NOVELTY_PATTERNS
        ]
        self._type_patterns = {
            category: [re.compile(p, re.IGNORECASE) for p in patterns]
            for category, patterns in NOVELTY_TYPE_PATTERNS.items()
        }

    def detect_novelty(
        self,
        chunks: List[Dict[str, Any]],
        all_chunks: Optional[List[Dict[str, Any]]] = None,
        question: str = "",
    ) -> Dict[str, Any]:
        """
        Detect novelty and original contributions from document chunks.

        Parameters
        ----------
        chunks : list[dict]
            Retrieved document chunks (post-BSCO).
        all_chunks : list[dict], optional
            All chunks from all uploaded papers (for deeper novelty analysis
            when retrieved chunks are insufficient).
        question : str
            User question (for focused analysis).

        Returns
        -------
        dict
            ``novelty_claims`` — list of detected novelty statements.
            ``novelty_types`` — categorized novelty (method, empirical, etc.).
            ``novelty_score`` — overall novelty significance [0, 1].
            ``summary`` — text summary of novelty findings.
            ``comparison_to_prior`` — how this differs from prior work.
        """
        if not chunks:
            return self._empty_result()

        # Use all_chunks if provided and retrieved chunks have minimal novelty
        if all_chunks and len(self._get_source_files(chunks)) < 2:
            chunks = all_chunks

        all_texts = [c.get("text", "") for c in chunks]
        all_text = " ".join(all_texts)

        # Extract novelty claims
        novelty_claims = self._extract_novelty_claims(chunks, all_text)

        # Classify novelty type
        novelty_types = self._classify_novelty(novelty_claims)

        # Extract comparison to prior work
        comparison = self._extract_comparison(chunks, all_text)

        # Score novelty significance
        novelty_score = self._score_novelty(
            novelty_claims, novelty_types, chunks
        )

        # Generate summary
        summary = self._generate_summary(novelty_claims, novelty_types)

        return {
            "novelty_claims": novelty_claims,
            "novelty_types": novelty_types,
            "comparison_to_prior": comparison,
            "novelty_score": round(novelty_score, 4),
            "total_claims": len(novelty_claims),
            "summary": summary,
        }

    def _extract_novelty_claims(
        self,
        chunks: List[Dict[str, Any]],
        all_text: str,
    ) -> List[Dict[str, Any]]:
        """Extract sentences containing novelty/contribution claims."""
        claims = []
        seen: Set[str] = set()

        for chunk in chunks:
            text = chunk.get("text", "")
            sentences = re.split(r"(?<=[.!?])\s+", text)

            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 15:
                    continue

                normalized = sentence.lower()
                for pattern in self._novelty_patterns:
                    if pattern.search(normalized):
                        key = sentence[:100].lower()
                        if key not in seen:
                            seen.add(key)
                            claims.append({
                                "text": sentence[:300].strip(),
                                "page": chunk.get("page_number"),
                                "source": chunk.get("source_file"),
                                "matched_indicator": pattern.pattern[:40],
                            })
                        break

        return claims

    def _classify_novelty(
        self,
        claims: List[Dict[str, Any]],
    ) -> Dict[str, List[str]]:
        """Classify novelty claims into categories."""
        types: Dict[str, List[str]] = defaultdict(list)

        for claim in claims:
            text = claim["text"].lower()
            for category, patterns in self._type_patterns.items():
                for pattern in patterns:
                    if pattern.search(text):
                        types[category].append(claim["text"][:150])
                        break

        # Ensure all categories exist
        for category in ["methodological", "empirical", "theoretical", "application"]:
            if category not in types:
                types[category] = []

        return dict(types)

    def _extract_comparison(
        self,
        chunks: List[Dict[str, Any]],
        all_text: str,
    ) -> List[str]:
        """Extract statements comparing to prior work."""
        comparisons = []
        seen: Set[str] = set()

        comp_patterns = [
            r"\b(compared to|compared with|in contrast|unlike|different from)\b",
            r"\b(superior|better than|outperform|exceed)\b",
            r"\b(prior work|previous work|existing method|related method)\b",
            r"\b(baseline|benchmark|standard approach|conventional)\b",
        ]
        compiled = [re.compile(p, re.IGNORECASE) for p in comp_patterns]

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
                            comparisons.append(sentence[:300].strip())
                        break

        return comparisons[:5]

    def _score_novelty(
        self,
        claims: List[Dict],
        types: Dict[str, List],
        chunks: List[Dict],
    ) -> float:
        """Score the overall novelty significance [0, 1]."""
        if not chunks:
            return 0.0

        # Factor 1: Number of novelty claims
        claim_density = min(len(claims) / max(len(chunks), 1), 1.0)

        # Factor 2: Breadth across novelty types
        active_types = sum(1 for items in types.values() if items)
        type_breadth = active_types / max(len(types), 1)

        # Factor 3: Average chunk confidence
        avg_confidence = sum(
            c.get("score", 0.5) for c in chunks
        ) / max(len(chunks), 1)

        # Weighted score
        score = 0.4 * claim_density + 0.3 * type_breadth + 0.3 * avg_confidence
        return min(score, 1.0)

    def _generate_summary(
        self,
        claims: List[Dict],
        types: Dict[str, List],
    ) -> str:
        """Generate a summary of novelty findings."""
        parts = []

        if claims:
            parts.append(
                f"**{len(claims)} novelty claim(s)** identified."
            )

            # Show top claims
            for i, claim in enumerate(claims[:3], 1):
                parts.append(f"{i}. {claim['text'][:150]}")

        # Show novelty type distribution
        active_types = {
            k: len(v) for k, v in types.items() if v
        }
        if active_types:
            type_str = ", ".join(
                f"{k} ({v})" for k, v in active_types.items()
            )
            parts.append(f"\n**Novelty distribution:** {type_str}")

        if not parts:
            return "No explicit novelty claims detected in the retrieved context."

        return "\n\n".join(parts)

    @staticmethod
    def _get_source_files(chunks: List[Dict[str, Any]]) -> List[str]:
        """Extract unique source file names from chunks."""
        return list(dict.fromkeys(
            c.get("source_file", "Unknown") for c in chunks
        ))

    def _empty_result(self) -> Dict[str, Any]:
        return {
            "novelty_claims": [],
            "novelty_types": {
                "methodological": [],
                "empirical": [],
                "theoretical": [],
                "application": [],
            },
            "comparison_to_prior": [],
            "novelty_score": 0.0,
            "total_claims": 0,
            "summary": "No document context available for novelty analysis.",
        }
