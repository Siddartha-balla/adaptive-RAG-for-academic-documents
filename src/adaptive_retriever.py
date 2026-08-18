"""
adaptive_retriever.py
----------------------
Adaptive retrieval policy for academic RAG.

Maps each query type produced by :class:`QueryClassifier` to a concrete
retrieval policy: how many candidates to fetch from the vector database,
what similarity threshold to enforce, and what the BSCO chunk budget is.

The policy also adjusts dynamically based on a live ``retrieval_signals``
snapshot (score spread, document density) so that sparse or dense corpora
are handled automatically.

Changes in this version
-----------------------
Added policies for 8 new query types:
    advantages, disadvantages, code, table, architecture,
    methodology, research_gap, future_work.
"""

from __future__ import annotations

from src.query_classifier import QueryClassifier


class AdaptiveRetriever:
    """
    Translates a user question into a retrieval policy dictionary.

    Policy keys
    -----------
    top_k : int
        Number of candidates to retrieve from FAISS hybrid search.
    threshold : float
        Minimum similarity score for a chunk to be selected.
    max_chunks : int
        Upper bound on BSCO-selected chunks sent to the LLM.
    min_chunks : int
        Lower bound — guarantees at least this many chunks reach the LLM.
    query_type : str
        Classified query type (propagated for downstream use).
    complexity : float
        Complexity score from the classifier (0-1).
    query_tokens : int
        Token count of the question.
    has_follow_up : bool
        Whether the question references a prior turn.
    """

    # Base policies per query type — tuned for academic documents.
    # Wider, softer thresholds for multi-faceted types; tighter for factual ones.
    _BASE_POLICIES: dict[str, dict] = {
        # ── Original types ─────────────────────────────────────────────
        "definition": {
            "top_k": 6,
            "threshold": 0.68,
            "max_chunks": 3,
            "min_chunks": 1,
        },
        "explanation": {
            "top_k": 10,
            "threshold": 0.62,
            "max_chunks": 5,
            "min_chunks": 2,
        },
        "comparison": {
            "top_k": 14,
            "threshold": 0.56,
            "max_chunks": 7,
            "min_chunks": 3,
        },
        "summary": {
            "top_k": 14,
            "threshold": 0.58,
            "max_chunks": 7,
            "min_chunks": 3,
        },
        "algorithm": {
            "top_k": 10,
            "threshold": 0.62,
            "max_chunks": 5,
            "min_chunks": 2,
        },
        "procedure": {
            "top_k": 10,
            "threshold": 0.60,
            "max_chunks": 6,
            "min_chunks": 2,
        },
        "numerical": {
            "top_k": 8,
            "threshold": 0.66,
            "max_chunks": 4,
            "min_chunks": 1,
        },
        "research_question": {
            "top_k": 14,
            "threshold": 0.58,
            "max_chunks": 7,
            "min_chunks": 3,
        },
        "factual": {
            "top_k": 6,
            "threshold": 0.70,
            "max_chunks": 3,
            "min_chunks": 1,
        },
        "open": {
            "top_k": 10,
            "threshold": 0.62,
            "max_chunks": 5,
            "min_chunks": 2,
        },
        "formula": {
            "top_k": 8,
            "threshold": 0.66,
            "max_chunks": 4,
            "min_chunks": 1,
        },
        "list_extraction": {
            "top_k": 12,
            "threshold": 0.60,
            "max_chunks": 6,
            "min_chunks": 2,
        },
        # ── New types ──────────────────────────────────────────────────
        "advantages": {
            # Benefits are typically concentrated; a few high-quality
            # chunks are sufficient.
            "top_k": 10,
            "threshold": 0.62,
            "max_chunks": 5,
            "min_chunks": 2,
        },
        "disadvantages": {
            # Same shape as advantages.
            "top_k": 10,
            "threshold": 0.62,
            "max_chunks": 5,
            "min_chunks": 2,
        },
        "code": {
            # Code questions need precise, verbatim chunks — tight threshold.
            "top_k": 8,
            "threshold": 0.65,
            "max_chunks": 4,
            "min_chunks": 1,
        },
        "table": {
            # Tabular content is dense; fewer, high-scoring chunks work best.
            "top_k": 8,
            "threshold": 0.65,
            "max_chunks": 4,
            "min_chunks": 1,
        },
        "architecture": {
            # Architecture spans multiple sections — fetch broadly.
            "top_k": 12,
            "threshold": 0.58,
            "max_chunks": 6,
            "min_chunks": 2,
        },
        "methodology": {
            # Methods / approaches often discussed across several sections.
            "top_k": 12,
            "threshold": 0.60,
            "max_chunks": 6,
            "min_chunks": 2,
        },
        "research_gap": {
            # Gaps appear in intro / conclusion — wide net needed.
            "top_k": 14,
            "threshold": 0.56,
            "max_chunks": 6,
            "min_chunks": 2,
        },
        "future_work": {
            # Future scope is usually a short final section.
            "top_k": 10,
            "threshold": 0.58,
            "max_chunks": 5,
            "min_chunks": 2,
        },
        "literature_survey": {
            # Survey synthesis must gather broader evidence across papers.
            "top_k": 18,
            "threshold": 0.54,
            "max_chunks": 9,
            "min_chunks": 4,
        },
        "novelty": {
            # Novelty is usually stated near contributions/results.
            "top_k": 14,
            "threshold": 0.56,
            "max_chunks": 7,
            "min_chunks": 3,
        },
        "paper_similarity": {
            # Similarity requires balanced evidence across uploaded papers.
            "top_k": 16,
            "threshold": 0.54,
            "max_chunks": 8,
            "min_chunks": 4,
        },
    }

    # Hard ceilings to keep inference lightweight on 8 GB RAM laptops.
    _TOP_K_MIN = 4
    _TOP_K_MAX = 20
    _MAX_CHUNKS_ABSOLUTE = 10

    def __init__(self) -> None:
        self.classifier = QueryClassifier()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify_question(self, question: str) -> str:
        """Return the query type string for *question*."""
        return self.classifier.classify(question).query_type

    def get_policy(
        self,
        question: str,
        retrieval_signals: dict | None = None,
    ) -> dict:
        """
        Build a complete retrieval policy for *question*.

        Parameters
        ----------
        question : str
            Raw user question.
        retrieval_signals : dict, optional
            Optional live signals from a first-pass retrieval:
            ``similarity_spread`` (float) and ``document_density`` (float).

        Returns
        -------
        dict
            Policy dictionary used by :class:`RAGPipeline` and
            :class:`BinarySearchOptimizer`.
        """
        profile = self.classifier.classify(question)
        query_type = profile.query_type

        # Fall back to "open" policy if a new type somehow isn't listed.
        policy = self._BASE_POLICIES.get(
            query_type, self._BASE_POLICIES["open"]
        ).copy()

        # Complexity boost: more complex questions need more candidates.
        complexity_boost = round(profile.complexity * 4)
        policy["top_k"] += complexity_boost

        # High-complexity → allow one extra chunk in context.
        if profile.complexity >= 0.70:
            policy["max_chunks"] += 1

        # Follow-up questions reference prior context — fetch a bit more.
        if profile.has_follow_up:
            policy["top_k"] += 2
            policy["max_chunks"] += 1

        # Adjust from live retrieval signals when available.
        if retrieval_signals:
            spread = retrieval_signals.get("similarity_spread", 0.0)
            density = retrieval_signals.get("document_density", 0.0)

            # Narrow score spread → scores are clustered → fetch more candidates.
            if spread < 0.08:
                policy["top_k"] += 2

            # Dense corpus → more relevant chunks expected → fetch more.
            if density > 0.70:
                policy["top_k"] += 1

        # Clamp to hard limits.
        policy["top_k"] = min(
            max(policy["top_k"], self._TOP_K_MIN), self._TOP_K_MAX
        )
        policy["max_chunks"] = min(
            max(policy["max_chunks"], policy["min_chunks"]),
            self._MAX_CHUNKS_ABSOLUTE,
        )

        # Propagate classifier metadata for downstream logging.
        policy["query_type"] = query_type
        policy["complexity"] = profile.complexity
        policy["query_tokens"] = profile.token_count
        policy["has_follow_up"] = profile.has_follow_up

        return policy

    def select_chunks(
        self,
        retrieved_chunks: list[dict],
        policy: dict,
    ) -> list[dict]:
        """
        Filter *retrieved_chunks* above the threshold defined in *policy*.

        Falls back to the single best chunk when nothing meets the threshold,
        so the LLM can still produce a grounded "no evidence" response.
        """
        if not retrieved_chunks:
            return []

        selected = [
            chunk
            for chunk in retrieved_chunks
            if chunk.get("score", 0.0) >= policy["threshold"]
        ]

        if not selected:
            selected = retrieved_chunks[:1]

        return selected[: policy["max_chunks"]]
