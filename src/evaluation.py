"""
evaluation.py
-------------
Runtime evaluation metrics for retrieval and context optimization.

Records granular per-stage timing (retrieval, reranking, BSCO, LLM
generation) in addition to the existing retrieval quality proxies, giving
the Research Evaluation Dashboard in the UI a complete picture of where
latency is spent and how much context compression BSCO achieved.

Improvements in this version
-----------------------------
Research Metrics (new)
    Full suite of research-grade metrics now captured from the redesigned
    EnhancedBSCO module:

    - coverage_score / semantic_coverage  — embedding-based query coverage
    - compression_score                   — token compression ratio (0-1)
    - context_diversity                   — page/doc/section diversity score
    - retrieval_recall_proxy              — fraction of retrieved chunks selected
    - prompt_efficiency                   — selected tokens / initial tokens
    - token_reduction_percent             — percentage token reduction
    - estimated_latency_saved             — estimated seconds saved
    - pages_covered                       — unique pages in selected context
    - documents_covered                   — unique documents in selected context
    - unique_sections                     — unique sections in selected context
    - avg_importance                      — average importance score
    - avg_confidence                      — average confidence score
    - verification_rounds                 — self-verification iterations
    - dedup_details                       — dedup, redundancy, sentence dedup counts

No labelled dataset is required — all metrics are observable at inference time.
"""

from __future__ import annotations

from typing import List, Optional


class EvaluationTracker:
    """
    Computes observable metrics for every question-answering cycle.

    Now captures the full suite of research-grade metrics from EnhancedBSCO
    for comprehensive evaluation and future research.
    """

    def calculate(
        self,
        retrieved_chunks: List[dict],
        selected_chunks: List[dict],
        bsco_stats: dict,
        response_time: float,
        answer: str,
        *,
        retrieval_time: float = 0.0,
        rerank_time: float = 0.0,
        bsco_time: float = 0.0,
        llm_time: float = 0.0,
        similarity_scores: Optional[List[float]] = None,
    ) -> dict:
        """
        Compute all evaluation metrics for one QA cycle.

        Parameters
        ----------
        retrieved_chunks : list[dict]
            Chunks returned by hybrid search (before BSCO).
        selected_chunks : list[dict]
            Chunks selected by BSCO (sent to LLM).
        bsco_stats : dict
            Statistics dictionary from :class:`EnhancedBSCO`.
        response_time : float
            Total end-to-end wall-clock time in seconds.
        answer : str
            Generated (and possibly verified) answer string.
        retrieval_time : float
            Time spent in hybrid search.
        rerank_time : float
            Time spent in reranking.
        bsco_time : float
            Time spent in BSCO optimization.
        llm_time : float
            Time spent waiting for LLM generation.
        similarity_scores : list[float], optional
            Raw similarity scores of retrieved chunks.

        Returns
        -------
        dict
            Flat dictionary of evaluation metrics including research-grade
            coverage, compression, diversity, and efficiency metrics.
        """
        retrieved_ids = [chunk.get("chunk_id") for chunk in retrieved_chunks]
        selected_ids  = {chunk.get("chunk_id") for chunk in selected_chunks}

        # MRR proxy: rank of the first selected chunk in retrieved list.
        first_selected_rank: Optional[int] = None
        for index, chunk_id in enumerate(retrieved_ids, start=1):
            if chunk_id in selected_ids:
                first_selected_rank = index
                break

        mrr = 0.0 if first_selected_rank is None else 1.0 / first_selected_rank
        precision_at_selected = len(selected_ids) / max(len(retrieved_chunks), 1)
        recall_proxy = len(selected_ids) / max(
            bsco_stats.get("initial_retrieved_chunks", 1), 1
        )

        # Similarity score statistics over retrieved chunks.
        scores = similarity_scores or [
            chunk.get("score", 0.0) for chunk in retrieved_chunks
        ]
        avg_similarity = round(sum(scores) / len(scores), 4) if scores else 0.0
        max_similarity = round(max(scores), 4) if scores else 0.0
        min_similarity = round(min(scores), 4) if scores else 0.0

        # Token accounting.
        prompt_tokens   = bsco_stats.get("final_tokens", 0)
        response_tokens = len(answer.split())

        # Inference speed: tokens generated per second during LLM call.
        if llm_time > 0 and response_tokens > 0:
            inference_speed = round(response_tokens / llm_time, 2)
        else:
            inference_speed = 0.0

        # Overhead = total - sum of known stages.
        known_stage_time = retrieval_time + rerank_time + bsco_time + llm_time
        overhead_time    = max(round(response_time - known_stage_time, 3), 0.0)

        # ── Research-grade metrics from BSCO ──────────────────────────
        coverage_score = bsco_stats.get("coverage_score", bsco_stats.get("semantic_coverage", 0.0))
        compression_score = bsco_stats.get("compression_score", bsco_stats.get("compression_ratio", 0.0))
        context_diversity = bsco_stats.get("context_diversity", 0.0)
        retrieval_recall_proxy_bsco = bsco_stats.get("retrieval_recall_proxy", recall_proxy)
        prompt_efficiency = bsco_stats.get("prompt_efficiency", 0.0)
        token_reduction_percent = bsco_stats.get("token_reduction_percent", 0.0)
        estimated_latency_saved = bsco_stats.get("estimated_latency_saved_seconds", 0.0)
        pages_covered = bsco_stats.get("pages_covered", 0)
        documents_covered = bsco_stats.get("documents_covered", 0)
        unique_sections = bsco_stats.get("unique_sections", 0)
        avg_importance = bsco_stats.get("avg_importance", 0.0)
        avg_confidence = bsco_stats.get("avg_confidence", 0.0)
        verification_rounds = bsco_stats.get("verification_rounds", 0)
        dedup_removed = bsco_stats.get("dedup_removed", 0)
        redundancy_removed = bsco_stats.get("redundancy_removed", 0)
        sentence_dedup_removed = bsco_stats.get("sentence_dedup_removed", 0)

        return {
            # ── Retrieval quality ─────────────────────────────────────
            "retrieval_precision_proxy": round(precision_at_selected, 4),
            "retrieval_recall_proxy":    round(retrieval_recall_proxy_bsco, 4),
            "mrr_proxy":                 round(mrr, 4),

            # ── Similarity scores ─────────────────────────────────────
            "avg_similarity": avg_similarity,
            "max_similarity": max_similarity,
            "min_similarity": min_similarity,

            # ── Context compression ───────────────────────────────────
            "prompt_tokens":             prompt_tokens,
            "response_tokens":           response_tokens,
            "response_length":           len(answer),
            "context_compression_ratio": bsco_stats.get("context_reduction_percent", 0.0),
            "term_coverage":             bsco_stats.get("term_coverage", 0.0),

            # ── Research-grade coverage ───────────────────────────────
            "coverage_score":            round(coverage_score, 4),
            "semantic_coverage":         round(coverage_score, 4),
            "compression_score":         round(compression_score, 4),
            "context_diversity":         round(context_diversity, 4),
            "prompt_efficiency":         round(prompt_efficiency, 4),
            "token_reduction_percent":   round(token_reduction_percent, 2),
            "estimated_latency_saved_seconds": round(estimated_latency_saved, 4),

            # ── Coverage breadth ──────────────────────────────────────
            "pages_covered":             pages_covered,
            "documents_covered":         documents_covered,
            "unique_sections":           unique_sections,

            # ── Quality scores ────────────────────────────────────────
            "avg_importance":            round(avg_importance, 4),
            "avg_confidence":            round(avg_confidence, 4),

            # ── BSCO detail ───────────────────────────────────────────
            "sentence_dedup_removed":    sentence_dedup_removed,
            "dedup_removed":             dedup_removed,
            "redundancy_removed":        redundancy_removed,
            "verification_rounds":       verification_rounds,

            # ── Stage timing ──────────────────────────────────────────
            "retrieval_time":            round(retrieval_time, 3),
            "rerank_time":               round(rerank_time, 3),
            "bsco_time":                 round(bsco_time, 3),
            "llm_time":                  round(llm_time, 3),
            "overhead_time":             overhead_time,
            "total_latency_seconds":     round(response_time, 3),

            # ── Throughput ────────────────────────────────────────────
            "inference_speed_tokens_per_sec": inference_speed,
        }
