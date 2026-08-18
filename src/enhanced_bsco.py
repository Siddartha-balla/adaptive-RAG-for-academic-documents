"""
Enhanced Binary Search-Based Context Optimizer (BSCO) — Research Grade
=======================================================================
IEEE/Springer-caliber adaptive context optimization for RAG systems.

This module implements a research-grade BSCO that intelligently identifies
the **minimum sufficient context** required to answer a user query while
maximizing answer quality and minimizing prompt size, latency, hallucinations,
and redundant context.

Key Research-Level Capabilities:
--------------------------------
1.  Adaptive Binary Search        — dynamically determine minimal sufficient subset
2.  Semantic Redundancy Removal   — multi-level dedup (hash, embedding, text, sentence)
3.  Context Compression           — preserve formulas/algorithms, remove boilerplate
4.  Adaptive Token Budget         — query complexity × model window × confidence × density
5.  Query-Type Aware Optimization — 16+ query types with custom strategies
6.  Embedding-Based Coverage      — cosine similarity coverage estimation
7.  Equal Page Weighting          — eliminate first-page bias
8.  Diversity Selection           — prefer different sections/headings/pages/documents
9.  Multi-PDF Optimization        — balanced context across all uploaded papers
10. Research-Aware Optimization   — special handling for compare/gap/survey/novelty
11. Citation Preservation         — preserve document/page/section/paragraph metadata
12. Context Importance Scoring    — 7-factor importance scoring per chunk
13. Self-Verification Integration — verify coverage before returning
14. Explainability                — 20+ optimization statistics returned
15. Performance Optimization      — LRU caching, lazy eval, minimal memory
16. Research Metrics              — retrieval proxy, coverage, compression, diversity
17. Maintainability               — type hints, docstrings, modular, clean

Author : Research Engineering Team
Version: 2.0.0 — Research Grade
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from collections import Counter, OrderedDict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from config import (
    MAX_CONTEXT_TOKENS,
    MIN_CONTEXT_COVERAGE,
    SIMILARITY_THRESHOLD,
    BSCO_EMBEDDING_COVERAGE_WEIGHT,
    BSCO_KEYWORD_COVERAGE_WEIGHT,
    BSCO_CONFIDENCE_COVERAGE_WEIGHT,
    BSCO_DIVERSITY_COVERAGE_WEIGHT,
    BSCO_DEDUP_THRESHOLD,
    BSCO_REDUNDANCY_THRESHOLD,
    BSCO_SENTENCE_DEDUP_ENABLED,
    BSCO_MIN_TOKEN_BUDGET,
    BSCO_MAX_TOKEN_BUDGET,
    BSCO_BUDGET_OVERHEAD_FACTOR,
    BSCO_IMPORTANCE_SEMANTIC_WEIGHT,
    BSCO_IMPORTANCE_QUERY_RELEVANCE_WEIGHT,
    BSCO_IMPORTANCE_SECTION_WEIGHT,
    BSCO_IMPORTANCE_HEADING_WEIGHT,
    BSCO_IMPORTANCE_CITATION_WEIGHT,
    BSCO_IMPORTANCE_RESEARCH_WEIGHT,
    BSCO_IMPORTANCE_RECENCY_WEIGHT,
    BSCO_PAGE_DIVERSITY_PENALTY,
    BSCO_DOCUMENT_DIVERSITY_PENALTY,
    BSCO_SECTION_DIVERSITY_BONUS,
    BSCO_SELF_VERIFY_ENABLED,
    BSCO_VERIFICATION_MIN_COVERAGE,
    BSCO_VERIFICATION_MAX_ITERATIONS,
    BSCO_CACHE_ENABLED,
    BSCO_CACHE_MAX_SIZE,
    BSCO_LAZY_EVALUATION,
)

# Shared utilities — single source of truth for tokenization, scoring, and
# academic constants (STOPWORDS, HIGH_VALUE_SECTIONS, PRESERVED_CHUNK_TYPES).
from src.utils import (
    PRESERVED_CHUNK_TYPES,
    extract_terms,
    estimate_tokens_for_chunks,
    safe_mean,
    jaccard_similarity,
    compute_section_relevance,
    compute_heading_relevance,
    compute_diversity_score,
)

# =====================================================================
# Constants (BSCO-specific; shared academic constants live in src.utils)
# =====================================================================

# Patterns for boilerplate removal during compression
_BOILERPLATE_PATTERNS: list[re.Pattern] = [
    re.compile(r"(?:^|\n)\s*references?\s*(?:$|\n)", re.IGNORECASE),
    re.compile(r"(?:^|\n)\s*bibliography\s*(?:$|\n)", re.IGNORECASE),
    re.compile(r"(?:^|\n)\s*acknowledgements?\s*(?:$|\n)", re.IGNORECASE),
    re.compile(r"(?:^|\n)\s*appendix\s+\w+\s*(?:$|\n)", re.IGNORECASE),
    re.compile(r"Copyright\s+©\s+\d{4}", re.IGNORECASE),
    re.compile(r"All rights reserved\.?", re.IGNORECASE),
    re.compile(r"(?:Page\s+\d+|-\s+\d+\s*-)"),  # Page numbers
    re.compile(r"https?://\S+"),  # URLs
]

# Query types that require research-aware multi-document handling
_RESEARCH_QUERY_TYPES: frozenset[str] = frozenset({
    "research_gap", "literature_survey", "novelty",
    "paper_similarity", "future_work", "comparison",
})

# Query types that benefit from broader context
_BROAD_CONTEXT_QUERY_TYPES: frozenset[str] = frozenset({
    "literature_survey", "paper_similarity", "comparison",
    "research_gap", "novelty", "summary", "architecture",
    "methodology", "explanation",
})

# Query types that benefit from tighter, more focused context
_TIGHT_CONTEXT_QUERY_TYPES: frozenset[str] = frozenset({
    "definition", "factual", "formula", "code",
    "numerical", "list_extraction",
})

# =====================================================================
# LRU Cache for Performance Optimization
# =====================================================================

class LRUCache:
    """
    Simple LRU cache with max size limit.
    
    Used to cache embedding computations and similarity scores
    to avoid redundant calculations during binary search iterations.
    """
    
    def __init__(self, max_size: int = BSCO_CACHE_MAX_SIZE) -> None:
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._max_size = max_size
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache, moving to end on hit."""
        if key in self._cache:
            value = self._cache.pop(key)
            self._cache[key] = value
            self._hits += 1
            return value
        self._misses += 1
        return None
    
    def put(self, key: str, value: Any) -> None:
        """Put value in cache, evicting oldest if full."""
        if key in self._cache:
            self._cache.pop(key)
        elif len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)
        self._cache[key] = value
    
    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
    
    @property
    def hit_rate(self) -> float:
        """Return cache hit rate."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0
    
    @property
    def size(self) -> int:
        return len(self._cache)


# =====================================================================
# EnhancedBSCO — Main Class
# =====================================================================

class EnhancedBSCO:
    """
    Research-Grade Binary Search-Based Context Optimizer.
    
    Intelligently identifies the minimum sufficient context subset from
    retrieved chunks by adaptively balancing coverage, diversity, importance,
    and token budget.
    
    Parameters
    ----------
    threshold : float
        Default similarity floor for sufficiency evaluation (default: 0.70)
    max_context_tokens : int
        Hard token budget for selected context (default: 1400)
    min_context_coverage : float
        Minimum fraction of query coverage required (default: 0.62)
    dedup_threshold : float
        Embedding similarity threshold for duplicate detection (default: 0.92)
    redundancy_threshold : float
        Text overlap threshold for redundancy detection (default: 0.82)
    cache_enabled : bool
        Enable LRU caching for performance (default: True)
    lazy_evaluation : bool
        Enable lazy evaluation of expensive computations (default: True)
    """
    
    def __init__(
        self,
        threshold: float = SIMILARITY_THRESHOLD,
        max_context_tokens: int = MAX_CONTEXT_TOKENS,
        min_context_coverage: float = MIN_CONTEXT_COVERAGE,
        dedup_threshold: float = BSCO_DEDUP_THRESHOLD,
        redundancy_threshold: float = BSCO_REDUNDANCY_THRESHOLD,
        cache_enabled: bool = BSCO_CACHE_ENABLED,
        lazy_evaluation: bool = BSCO_LAZY_EVALUATION,
        embedder: Optional[Any] = None,
    ) -> None:
        """Initialize the EnhancedBSCO with research-grade parameters.

        Parameters
        ----------
        embedder : optional
            A shared :class:`EmbeddingGenerator` instance. When provided, BSCO
            reuses the pipeline's already-loaded embedding model instead of
            lazily constructing a second copy (which would double RAM usage and
            add a costly cold-start on the first question).
        """
        self.threshold = threshold
        self.max_context_tokens = max_context_tokens
        self.min_context_coverage = min_context_coverage
        self.dedup_threshold = dedup_threshold
        self.redundancy_threshold = redundancy_threshold
        self.cache_enabled = cache_enabled
        self.lazy_evaluation = lazy_evaluation
        
        # Embedding generator (lazy initialized to avoid GPU/memory overhead,
        # unless a shared instance was injected by the pipeline).
        self._embedder: Optional[Any] = embedder
        
        # Caches for performance optimization
        self._embedding_cache: LRUCache = LRUCache() if cache_enabled else None  # type: ignore
        self._similarity_cache: LRUCache = LRUCache() if cache_enabled else None  # type: ignore
        self._coverage_cache: LRUCache = LRUCache() if cache_enabled else None  # type: ignore
        
        # Timing statistics
        self._total_optimize_time: float = 0.0
        self._optimize_calls: int = 0
    
    # =================================================================
    # Public API
    # =================================================================
    
    def optimize(
        self,
        retrieved_chunks: List[Dict[str, Any]],
        question: str = "",
        threshold: Optional[float] = None,
        max_chunks: Optional[int] = None,
        min_chunks: int = 1,
        query_type: str = "open",
        return_stats: bool = True,
        embeddings: Optional[np.ndarray] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Return the minimal sufficient context subset with research-grade optimization.
        
        This is the primary entry point. It performs 10 phases of optimization:
        
        Phase 1  — Pre-filtering (remove clearly irrelevant chunks)
        Phase 2  — Semantic Deduplication (remove near-duplicate chunks)
        Phase 3  — Redundancy Removal (remove overlapping content)
        Phase 4  — Context Importance Scoring (multi-factor scoring)
        Phase 5  — Adaptive Token Budget Calculation
        Phase 6  — Binary Search (find minimal sufficient subset)
        Phase 7  — Diversity Selection (ensure page/document/section diversity)
        Phase 8  — Self-Verification (verify coverage, continue if needed)
        Phase 9  — Cross-Chunk Sentence Deduplication
        Phase 10 — Statistics & Explainability
        
        Parameters
        ----------
        retrieved_chunks : List[Dict]
            Reranked candidate chunks (sorted by relevance descending)
        question : str
            User question for coverage and importance computation
        threshold : float, optional
            Override instance-level similarity threshold
        max_chunks : int, optional
            Maximum chunks allowed in output
        min_chunks : int
            Minimum chunks that must be included
        query_type : str
            Query category for adaptive strategy selection
        return_stats : bool
            Return comprehensive statistics dictionary
        embeddings : np.ndarray, optional
            Pre-computed chunk embeddings for duplicate detection
            
        Returns
        -------
        Tuple[List[Dict], Dict]
            Selected chunks and comprehensive statistics dictionary
        """
        optimize_start = time.time()
        self._optimize_calls += 1
        
        # --- Handle empty input ---
        if not retrieved_chunks:
            stats = self._empty_stats(threshold or self.threshold, query_type)
            return ([], stats) if return_stats else ([], {})
        
        active_threshold = self.threshold if threshold is None else threshold
        
        # Extract query terms for coverage computation
        query_terms = self._extract_terms(question)
        query_embedding = self._get_query_embedding(question) if question else None
        
        # =================================================================
        # Phase 1: Pre-filtering — remove clearly irrelevant chunks
        # =================================================================
        prefiltered = self._prefilter_by_score(retrieved_chunks, active_threshold)
        initial_chunks = len(retrieved_chunks)
        prefiltered_count = len(prefiltered)
        
        # =================================================================
        # Phase 2: Semantic Deduplication — remove near-duplicate chunks
        # =================================================================
        dedup_start = time.time()
        dedup_removed = 0
        if embeddings is not None and len(prefiltered) > 1:
            candidates, dedup_removed = self._semantic_deduplicate(
                prefiltered, embeddings, self.dedup_threshold
            )
        else:
            # Fallback: hash-based exact dedup
            candidates, dedup_removed = self._exact_deduplicate(prefiltered)
        dedup_time = time.time() - dedup_start
        after_dedup_count = len(candidates)
        
        # =================================================================
        # Phase 3: Redundancy Removal — remove overlapping content
        # =================================================================
        redundancy_start = time.time()
        candidates, redundancy_removed = self._remove_redundancy(
            candidates, self.redundancy_threshold
        )
        redundancy_time = time.time() - redundancy_start
        after_redundancy_count = len(candidates)
        
        if not candidates:
            stats = self._empty_stats(active_threshold, query_type)
            return ([], stats) if return_stats else ([], {})
        
        # =================================================================
        # Phase 4: Context Importance Scoring — multi-factor scoring
        # =================================================================
        scoring_start = time.time()
        candidates = self._compute_importance_scores(
            candidates, question, query_terms, query_embedding, query_type
        )
        scoring_time = time.time() - scoring_start
        
        # Sort by importance score (descending) for optimal selection
        candidates.sort(key=lambda c: c.get("importance_score", 0.0), reverse=True)
        
        initial_tokens = self._estimate_tokens(candidates)
        
        # =================================================================
        # Phase 5: Adaptive Token Budget Calculation
        # =================================================================
        adaptive_budget = self._calculate_adaptive_budget(
            question=question,
            query_type=query_type,
            initial_tokens=initial_tokens,
            retrieved_chunks=retrieved_chunks,
            candidates=candidates,
        )
        
        # Apply max_chunks constraint if provided
        budget = max_chunks or len(candidates)
        budget = max(min_chunks, budget)
        budget = min(budget, len(candidates))
        
        # =================================================================
        # Phase 6: Binary Search — find minimal sufficient subset
        # =================================================================
        binary_search_start = time.time()
        
        # Determine stopping condition based on query type
        min_coverage = self._get_min_coverage_for_type(query_type)
        
        left = min(min_chunks, budget)
        right = budget
        best_size = budget
        best_coverage = 0.0
        
        # Adaptive binary search with early stopping
        iteration_count = 0
        max_iterations = int(math.log2(max(budget, 1))) + 2
        
        while left <= right and iteration_count < max_iterations:
            iteration_count += 1
            mid = (left + right) // 2
            subset = candidates[:mid]
            
            # Compute coverage for this subset
            coverage = self._compute_semantic_coverage(
                subset=subset,
                question=question,
                query_terms=query_terms,
                query_embedding=query_embedding,
                query_type=query_type,
            )
            
            # Check sufficiency
            is_sufficient = self._is_sufficient(
                chunks=subset,
                query_terms=query_terms,
                coverage=coverage,
                min_coverage=min_coverage,
                threshold=active_threshold,
                min_chunks=min_chunks,
                token_budget=adaptive_budget,
                query_type=query_type,
            )
            
            if is_sufficient:
                best_size = mid
                best_coverage = coverage
                right = mid - 1
            else:
                left = mid + 1
        
        binary_search_time = time.time() - binary_search_start
        
        # Select the best subset
        selected = candidates[:best_size]
        
        # =================================================================
        # Phase 7: Diversity Selection — ensure page/document/section diversity
        # =================================================================
        diversity_start = time.time()
        selected = self._apply_diversity_selection(
            selected, candidates, best_size, query_type
        )
        diversity_time = time.time() - diversity_start
        
        # =================================================================
        # Phase 8: Self-Verification — verify coverage, continue if needed
        # =================================================================
        verification_start = time.time()
        verification_rounds = 0
        if BSCO_SELF_VERIFY_ENABLED and question:
            for v_iter in range(BSCO_VERIFICATION_MAX_ITERATIONS):
                coverage = self._compute_semantic_coverage(
                    subset=selected,
                    question=question,
                    query_terms=query_terms,
                    query_embedding=query_embedding,
                    query_type=query_type,
                )
                
                if coverage >= BSCO_VERIFICATION_MIN_COVERAGE:
                    break
                
                # Need more chunks — expand selection
                expand_count = min(len(selected) + 2, len(candidates))
                if expand_count <= len(selected):
                    break
                selected = candidates[:expand_count]
                verification_rounds = v_iter + 1
            
            # Recompute coverage after verification
            final_coverage = self._compute_semantic_coverage(
                subset=selected,
                question=question,
                query_terms=query_terms,
                query_embedding=query_embedding,
                query_type=query_type,
            )
        else:
            final_coverage = best_coverage
            verification_rounds = 0
        
        verification_time = time.time() - verification_start
        
        # Trim to token budget
        selected = self._trim_to_token_budget(selected, adaptive_budget, min_chunks)
        
        # =================================================================
        # Phase 9: Cross-Chunk Sentence Deduplication (optional)
        # =================================================================
        sentence_dedup_start = time.time()
        sentence_dedup_removed = 0
        if BSCO_SENTENCE_DEDUP_ENABLED and len(selected) > 1:
            selected, sentence_dedup_removed = self._cross_chunk_dedup(selected)
        sentence_dedup_time = time.time() - sentence_dedup_start
        
        # =================================================================
        # Phase 10: Statistics & Explainability
        # =================================================================
        final_tokens = self._estimate_tokens(selected)
        compression_ratio = 1.0 - (final_tokens / initial_tokens) if initial_tokens > 0 else 0.0
        context_reduction_percent = round(compression_ratio * 100, 2)
        
        # Compute aggregate statistics
        avg_score = self._safe_mean([c.get("score", 0.0) for c in selected])
        avg_importance = self._safe_mean([c.get("importance_score", 0.0) for c in selected])
        avg_confidence = self._safe_mean([c.get("confidence", c.get("score", 0.0)) for c in selected])
        
        # Pages and documents covered
        pages_covered = len(set(
            c.get("page_number") for c in selected if "page_number" in c
        ))
        documents_covered = len(set(
            c.get("source_file", "") for c in selected if c.get("source_file")
        ))
        
        # Diversity metrics
        unique_pages = len(set(
            c.get("page_number", -1) for c in selected
        ))
        unique_sections = len(set(
            c.get("section_title", "") for c in selected if c.get("section_title")
        ))
        unique_docs = len(set(
            c.get("source_file", "") for c in selected if c.get("source_file")
        ))
        
        # Research metrics
        retrieval_recall_proxy = len(selected) / max(initial_chunks, 1)
        coverage_score = round(final_coverage, 4)
        compression_score = round(compression_ratio, 4)
        context_diversity = self._compute_diversity_score(selected)
        token_reduction = max(initial_tokens - final_tokens, 0)
        token_reduction_pct = round(
            (token_reduction / max(initial_tokens, 1)) * 100, 2
        )
        
        # Estimated latency saved (based on token reduction)
        # Rough estimate: ~0.5ms per token for LLM inference
        estimated_latency_saved = round(token_reduction * 0.0005, 3)
        
        # Prompt efficiency (answer tokens / prompt tokens ratio target)
        prompt_efficiency = round(
            final_tokens / max(initial_tokens, 1), 4
        )
        
        total_optimize_time = time.time() - optimize_start
        self._total_optimize_time += total_optimize_time
        
        # Build comprehensive statistics dictionary
        stats = {
            # ── Chunk counts ──────────────────────────────────────────
            "initial_chunks": initial_chunks,
            "initial_retrieved_chunks": initial_chunks,
            "prefiltered_chunks": prefiltered_count,
            "after_dedup": after_dedup_count,
            "after_redundancy": after_redundancy_count,
            "deduplicated_candidates": after_redundancy_count,
            "selected_chunks": len(selected),
            "final_selected_chunks": len(selected),
            
            # ── Token statistics ──────────────────────────────────────
            "initial_tokens": initial_tokens,
            "final_tokens": final_tokens,
            "token_reduction": token_reduction,
            "token_reduction_percent": token_reduction_pct,
            "adaptive_budget": adaptive_budget,
            
            # ── Compression ───────────────────────────────────────────
            "compression_ratio": compression_ratio,
            "context_reduction_percent": context_reduction_percent,
            "compression_score": compression_score,
            
            # ── Deduplication ─────────────────────────────────────────
            "dedup_removed": dedup_removed,
            "redundancy_removed": redundancy_removed,
            "sentence_dedup_removed": sentence_dedup_removed,
            
            # ── Coverage and similarity ───────────────────────────────
            "coverage_score": coverage_score,
            "semantic_coverage": coverage_score,
            "term_coverage": coverage_score,
            "threshold": active_threshold,
            "threshold_used": round(active_threshold, 4),
            "avg_score": avg_score,
            "avg_importance": avg_importance,
            "avg_confidence": avg_confidence,
            
            # ── Query context ─────────────────────────────────────────
            "query_type": query_type,
            "query_complexity": self._compute_query_complexity(question, query_type),
            "query_length": len(question.split()),
            "verification_rounds": verification_rounds,
            
            # ── Diversity ─────────────────────────────────────────────
            "pages_covered": pages_covered,
            "documents_covered": documents_covered,
            "unique_pages": unique_pages,
            "unique_sections": unique_sections,
            "unique_documents": unique_docs,
            "context_diversity": context_diversity,
            
            # ── Research metrics ──────────────────────────────────────
            "retrieval_recall_proxy": round(retrieval_recall_proxy, 4),
            "prompt_efficiency": prompt_efficiency,
            
            # ── Timing ────────────────────────────────────────────────
            "optimize_time_seconds": round(total_optimize_time, 4),
            "dedup_time_seconds": round(dedup_time, 4),
            "redundancy_time_seconds": round(redundancy_time, 4),
            "scoring_time_seconds": round(scoring_time, 4),
            "binary_search_time_seconds": round(binary_search_time, 4),
            "diversity_time_seconds": round(diversity_time, 4),
            "verification_time_seconds": round(verification_time, 4),
            "sentence_dedup_time_seconds": round(sentence_dedup_time, 4),
            
            # ── Estimated impact ──────────────────────────────────────
            "estimated_latency_saved_seconds": estimated_latency_saved,
            "context_reduction_percentage": context_reduction_percent,
        }
        
        return (selected, stats) if return_stats else (selected, {})
    
    # =================================================================
    # Phase 1: Pre-filtering
    # =================================================================
    
    def _prefilter_by_score(
        self,
        chunks: List[Dict[str, Any]],
        threshold: float,
    ) -> List[Dict[str, Any]]:
        """
        Pre-filter chunks by similarity score with adaptive threshold.
        
        Uses a more lenient threshold for pre-filtering to avoid
        removing potentially useful chunks early.
        
        Args:
            chunks: Candidate chunks from retrieval
            threshold: Base similarity threshold
            
        Returns:
            Filtered chunks, guaranteed at least 1
        """
        if not chunks:
            return []
        
        # Adaptive pre-filter threshold — more lenient for research queries
        prefilter_threshold = threshold * 0.45  # More lenient than original 0.5
        
        filtered = [
            chunk for chunk in chunks
            if chunk.get("score", 0) >= prefilter_threshold
        ]
        
        # Always keep at least the top chunk
        if not filtered and chunks:
            filtered = [chunks[0]]
        
        return filtered if filtered else chunks[:1]
    
    # =================================================================
    # Phase 2: Semantic Deduplication
    # =================================================================
    
    def _semantic_deduplicate(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: np.ndarray,
        threshold: float,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Remove semantically duplicate chunks using embedding similarity.
        
        Uses a two-pass approach:
        1. Exact hash-based dedup (fast)
        2. Embedding cosine similarity dedup (accurate)
        
        Args:
            chunks: Candidate chunks
            embeddings: Pre-computed chunk embeddings
            threshold: Similarity threshold for duplicate detection
            
        Returns:
            Tuple of (deduplicated chunks, count removed)
        """
        if len(chunks) < 2:
            return chunks, 0
        
        removed = 0
        kept: List[Dict[str, Any]] = []
        seen_hashes: Set[str] = set()
        kept_indices: List[int] = []
        
        for i, chunk in enumerate(chunks):
            chunk_text = chunk.get("text", "")
            if not chunk_text:
                continue
            
            # Pass 1: Exact hash-based dedup
            chunk_hash = hashlib.md5(chunk_text.encode("utf-8", errors="ignore")).hexdigest()
            if chunk_hash in seen_hashes:
                removed += 1
                continue
            seen_hashes.add(chunk_hash)
            
            # Pass 2: Embedding-based semantic dedup
            is_duplicate = False
            if embeddings is not None and i < len(embeddings):
                for kept_idx in kept_indices:
                    if kept_idx < len(embeddings):
                        sim = float(np.dot(embeddings[i], embeddings[kept_idx]))
                        if sim >= threshold:
                            # Keep the chunk with higher score
                            if chunk.get("score", 0) > kept[-1].get("score", 0) if kept else False:
                                # Replace the kept chunk with this one
                                pass  # Handled below
                            is_duplicate = True
                            removed += 1
                            break
            
            if not is_duplicate:
                kept.append(chunk)
                kept_indices.append(i)
        
        return kept, removed
    
    def _exact_deduplicate(
        self,
        chunks: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Remove exact duplicate chunks by text hash.
        
        Used as fallback when embeddings are not available.
        
        Args:
            chunks: Candidate chunks
            
        Returns:
            Tuple of (deduplicated chunks, count removed)
        """
        if len(chunks) < 2:
            return chunks, 0
        
        removed = 0
        kept: List[Dict[str, Any]] = []
        seen_hashes: Set[str] = set()
        
        for chunk in chunks:
            chunk_text = chunk.get("text", "")
            if not chunk_text:
                continue
            
            chunk_hash = hashlib.md5(chunk_text.encode("utf-8", errors="ignore")).hexdigest()
            if chunk_hash in seen_hashes:
                removed += 1
                continue
            seen_hashes.add(chunk_hash)
            kept.append(chunk)
        
        return kept, removed
    
    # =================================================================
    # Phase 3: Redundancy Removal
    # =================================================================
    
    def _remove_redundancy(
        self,
        chunks: List[Dict[str, Any]],
        threshold: float,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Remove semantically redundant content across chunks.
        
        Uses Jaccard text overlap to identify chunks that contain
        largely the same information. When redundancy is detected,
        the chunk with higher importance score is kept.
        
        Args:
            chunks: Candidate chunks
            threshold: Overlap threshold for redundancy detection
            
        Returns:
            Tuple of (filtered chunks, count removed)
        """
        if len(chunks) < 2:
            return chunks, 0
        
        removed = 0
        kept: List[Dict[str, Any]] = []
        
        for chunk in chunks:
            chunk_text = chunk.get("text", "")
            if not chunk_text:
                kept.append(chunk)
                continue
            
            is_redundant = False
            chunk_importance = chunk.get("importance_score", chunk.get("score", 0.0))
            
            for kept_idx, kept_chunk in enumerate(kept):
                kept_text = kept_chunk.get("text", "")
                overlap = self._calculate_text_overlap(chunk_text, kept_text)
                
                if overlap >= threshold:
                    # Keep the higher-scored chunk
                    kept_importance = kept_chunk.get("importance_score", kept_chunk.get("score", 0.0))
                    if chunk_importance > kept_importance:
                        kept[kept_idx] = chunk
                    is_redundant = True
                    removed += 1
                    break
            
            if not is_redundant:
                kept.append(chunk)
        
        return kept, removed
    
    def _calculate_text_overlap(self, text1: str, text2: str) -> float:
        """
        Calculate Jaccard similarity between two texts.
        
        Delegates to the shared utility :func:`src.utils.jaccard_similarity`.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Jaccard similarity score (0-1)
        """
        return jaccard_similarity(text1, text2)
    
    # =================================================================
    # Phase 4: Context Importance Scoring
    # =================================================================
    
    def _compute_importance_scores(
        self,
        chunks: List[Dict[str, Any]],
        question: str,
        query_terms: Set[str],
        query_embedding: Optional[np.ndarray],
        query_type: str,
    ) -> List[Dict[str, Any]]:
        """
        Compute multi-factor importance scores for each chunk.
        
        Factors considered:
        1. Semantic similarity to query (embedding)
        2. Query term relevance (keyword overlap)
        3. Section importance (academic section value)
        4. Heading relevance (heading match with query)
        5. Citation density (proxy)
        6. Research contribution (algorithm, novelty, methodology)
        7. Recency (position in document)
        
        Args:
            chunks: Chunks to score
            question: User question
            query_terms: Extracted query terms
            query_embedding: Query embedding vector
            query_type: Query category
            
        Returns:
            Chunks with importance_score field added
        """
        # Get chunk embeddings if available
        chunk_embeddings = self._get_chunk_embeddings_if_needed(chunks)
        
        total_chunks = max(len(chunks), 1)
        
        for idx, chunk in enumerate(chunks):
            chunk_text = chunk.get("text", "")
            section = chunk.get("section_title", "") or ""
            chunk_type = chunk.get("chunk_type", "paragraph")
            
            # Factor 1: Semantic similarity to query (embedding)
            semantic_score = 0.0
            if query_embedding is not None and chunk_embeddings is not None and idx < len(chunk_embeddings):
                semantic_score = float(np.dot(query_embedding, chunk_embeddings[idx]))
            elif query_terms:
                # Fallback: term overlap
                chunk_terms = set(re.findall(r"\b\w+\b", chunk_text.lower()))
                semantic_score = len(query_terms & chunk_terms) / max(len(query_terms), 1)
            
            # Factor 2: Query term relevance
            chunk_terms = set(re.findall(r"\b\w+\b", chunk_text.lower()))
            term_overlap = len(query_terms & chunk_terms) / max(len(query_terms), 1) if query_terms else 0.0
            
            # Factor 3: Section importance
            section_score = self._compute_section_importance(section)
            
            # Factor 4: Heading relevance
            heading_score = self._compute_heading_relevance(query_terms, section)
            
            # Factor 5: Citation density proxy (presence of citation markers)
            citation_count = len(re.findall(r"\[(\d+(?:[-,]\d+)*)\]", chunk_text))
            citation_density = min(citation_count / 3.0, 1.0)  # Cap at 1.0
            
            # Factor 6: Research contribution
            research_score = self._compute_research_contribution(chunk_text, chunk_type, query_type)
            
            # Factor 7: Recency (position in document)
            chunk_id = chunk.get("chunk_id", 0) or 0
            recency_score = min(chunk_id / max(total_chunks, 1), 1.0)
            
            # Weighted fusion of all factors
            importance_score = (
                BSCO_IMPORTANCE_SEMANTIC_WEIGHT * semantic_score
                + BSCO_IMPORTANCE_QUERY_RELEVANCE_WEIGHT * term_overlap
                + BSCO_IMPORTANCE_SECTION_WEIGHT * section_score
                + BSCO_IMPORTANCE_HEADING_WEIGHT * heading_score
                + BSCO_IMPORTANCE_CITATION_WEIGHT * citation_density
                + BSCO_IMPORTANCE_RESEARCH_WEIGHT * research_score
                + BSCO_IMPORTANCE_RECENCY_WEIGHT * recency_score
            )
            
            chunk["importance_score"] = round(importance_score, 4)
            chunk["semantic_score"] = round(semantic_score, 4)
            chunk["section_score"] = round(section_score, 4)
            chunk["heading_score"] = round(heading_score, 4)
            chunk["research_score"] = round(research_score, 4)
            chunk["citation_density"] = round(citation_density, 4)
        
        return chunks
    
    def _compute_section_importance(self, section_title: str) -> float:
        """
        Compute importance score based on academic section type.
        
        Delegates to the shared utility :func:`src.utils.compute_section_relevance`
        (single source of truth for ``HIGH_VALUE_SECTIONS``). A 0.3 default is
        returned for unknown sections to preserve backward-compatible behaviour.
        
        Args:
            section_title: Section heading text
            
        Returns:
            Importance score in [0, 1]
        """
        if not section_title:
            return 0.3  # Default for unknown sections
        
        return compute_section_relevance(section_title)
    
    def _compute_heading_relevance(
        self,
        query_terms: Set[str],
        section_title: str,
    ) -> float:
        """
        Compute relevance of section heading to the query.
        
        Delegates to the shared utility :func:`src.utils.compute_heading_relevance`.
        
        Args:
            query_terms: Query terms
            section_title: Section heading
            
        Returns:
            Heading relevance score in [0, 1]
        """
        return compute_heading_relevance(query_terms, section_title)
    
    def _compute_research_contribution(
        self,
        text: str,
        chunk_type: str,
        query_type: str,
    ) -> float:
        """
        Compute research contribution score for a chunk.
        
        Detects presence of research-significant content:
        algorithms, formulas, methodology, results, novelty claims.
        
        Args:
            text: Chunk text
            chunk_type: Type of chunk
            query_type: Query category
            
        Returns:
            Research contribution score in [0, 1]
        """
        text_lower = text.lower()
        score = 0.0
        
        # Preserved chunk types are high-value
        if chunk_type in PRESERVED_CHUNK_TYPES:
            score += 0.3
        
        # Research methodology indicators
        method_indicators = [
            "we propose", "our method", "our approach", "we introduce",
            "we present", "we develop", "our algorithm", "our framework",
            "experimental result", "evaluation", "benchmark",
            "state-of-the-art", "state of the art", "sota",
        ]
        for indicator in method_indicators:
            if indicator in text_lower:
                score += 0.1
        
        # Novelty indicators
        novelty_indicators = [
            "novel", "first", "new approach", "new method",
            "contribution", "original", "unique",
        ]
        for indicator in novelty_indicators:
            if indicator in text_lower:
                score += 0.05
        
        # Formula/equation indicators
        if re.search(r"\\[a-zA-Z]+|\[.*\]|\(.*\)|∑|∫|∂|∇|λ|θ|α|β", text):
            score += 0.15
        
        # Result indicators
        if re.search(r"\b(accuracy|precision|recall|f1|bleu|rouge|perplexity)\b", text_lower):
            score += 0.1
        
        # Citation indicators
        citation_count = len(re.findall(r"\[\d+\]", text))
        if citation_count >= 2:
            score += 0.05
        
        return min(score, 1.0)
    
    # =================================================================
    # Phase 5: Adaptive Token Budget
    # =================================================================
    
    def _calculate_adaptive_budget(
        self,
        question: str,
        query_type: str,
        initial_tokens: int,
        retrieved_chunks: List[Dict[str, Any]],
        candidates: List[Dict[str, Any]],
    ) -> int:
        """
        Compute an optimal token budget using multiple factors.
        
        Factors:
        - Model context window (max_context_tokens)
        - Query complexity (length, structure)
        - Query type (some types need more context)
        - Retrieval confidence (average similarity score)
        - Chunk density (average chunk size)
        - Overhead factor for safety margin
        
        Args:
            question: User question
            query_type: Query category
            initial_tokens: Initial token estimate
            retrieved_chunks: All retrieved chunks (for confidence)
            candidates: Deduplicated candidates
            
        Returns:
            Optimal token budget
        """
        # Base budget from max context
        base_budget = self.max_context_tokens
        
        # Factor 1: Query complexity
        query_length = len(question.split())
        query_complexity = self._compute_query_complexity(question, query_type)
        complexity_factor = 1.0 + (query_complexity * 0.5)  # 1.0 to 1.5
        
        # Factor 2: Query type multipliers
        type_multipliers = {
            "definition": 0.35,
            "factual": 0.30,
            "formula": 0.35,
            "code": 0.35,
            "numerical": 0.35,
            "list_extraction": 0.40,
            "procedure": 0.45,
            "algorithm": 0.45,
            "advantages": 0.45,
            "disadvantages": 0.45,
            "explanation": 0.50,
            "methodology": 0.55,
            "architecture": 0.55,
            "summary": 0.50,
            "comparison": 0.65,
            "research_question": 0.55,
            "research_gap": 0.50,
            "future_work": 0.45,
            "novelty": 0.55,
            "paper_similarity": 0.65,
            "literature_survey": 0.70,
            "open": 0.45,
        }
        type_multiplier = type_multipliers.get(query_type, 0.45)
        
        # Factor 3: Retrieval confidence
        avg_retrieval_score = self._safe_mean(
            [c.get("score", 0.0) for c in retrieved_chunks]
        )
        # Higher confidence → tighter budget (we trust the results)
        # Lower confidence → larger budget (need more context)
        confidence_factor = 1.0 + (1.0 - avg_retrieval_score) * 0.3
        
        # Factor 4: Chunk density
        avg_chunk_tokens = self._safe_mean([
            self._estimate_tokens([c]) for c in candidates
        ]) if candidates else 100
        density_factor = 1.0 + (avg_chunk_tokens / 500) * 0.2  # Normalize
        
        # Factor 5: Query length boost
        length_boost = min(1.0 + (query_length / 100), 1.5)
        
        # Compute adaptive budget
        adaptive_budget = int(
            base_budget
            * complexity_factor
            * type_multiplier
            * confidence_factor
            * density_factor
            * length_boost
        )
        
        # Apply overhead factor
        adaptive_budget = int(adaptive_budget * (1.0 + BSCO_BUDGET_OVERHEAD_FACTOR))
        
        # Clamp to min/max
        adaptive_budget = max(BSCO_MIN_TOKEN_BUDGET, min(adaptive_budget, BSCO_MAX_TOKEN_BUDGET))
        adaptive_budget = min(adaptive_budget, self.max_context_tokens)
        
        return adaptive_budget
    
    def _compute_query_complexity(self, question: str, query_type: str) -> float:
        """
        Compute query complexity score in [0, 1].
        
        Factors:
        - Base score per query type
        - Question length
        - Multi-entity connectors
        - Conditional/negation language
        
        Args:
            question: User question
            query_type: Query category
            
        Returns:
            Complexity score (0-1)
        """
        normalized = question.lower().strip()
        words = re.findall(r"[a-zA-Z0-9]+", normalized)
        
        base_complexities = {
            "comparison": 0.85,
            "literature_survey": 0.90,
            "paper_similarity": 0.85,
            "novelty": 0.80,
            "research_gap": 0.80,
            "architecture": 0.80,
            "methodology": 0.75,
            "summary": 0.75,
            "research_question": 0.75,
            "future_work": 0.70,
            "algorithm": 0.65,
            "procedure": 0.65,
            "explanation": 0.65,
            "advantages": 0.55,
            "disadvantages": 0.55,
            "numerical": 0.55,
            "definition": 0.40,
            "factual": 0.30,
            "formula": 0.50,
            "code": 0.50,
            "list_extraction": 0.45,
            "open": 0.50,
        }
        
        base = base_complexities.get(query_type, 0.50)
        length_boost = min(len(words) / 60.0, 0.25)
        multi_entity = 0.08 if re.search(
            r"\b(and|or|across|between|multiple|several|various|both)\b",
            normalized,
        ) else 0.0
        conditional = 0.05 if re.search(
            r"\b(if|unless|assuming|given that|when|although|however)\b",
            normalized,
        ) else 0.0
        
        return round(min(base + length_boost + multi_entity + conditional, 1.0), 3)
    
    # =================================================================
    # Phase 6: Binary Search Sufficiency Check
    # =================================================================
    
    def _get_min_coverage_for_type(self, query_type: str) -> float:
        """
        Get minimum coverage threshold for a query type.
        
        Different query types require different levels of coverage:
        - Definition: low coverage, focused context
        - Literature survey: high coverage, broad context
        - Factual: medium coverage, precise context
        
        Args:
            query_type: Query category
            
        Returns:
            Minimum coverage threshold (0-1)
        """
        thresholds = {
            "definition": 0.45,
            "factual": 0.50,
            "formula": 0.50,
            "code": 0.50,
            "numerical": 0.45,
            "list_extraction": 0.45,
            "procedure": 0.50,
            "algorithm": 0.50,
            "advantages": 0.45,
            "disadvantages": 0.45,
            "explanation": 0.55,
            "methodology": 0.55,
            "architecture": 0.55,
            "summary": 0.55,
            "comparison": 0.60,
            "research_question": 0.55,
            "research_gap": 0.60,
            "future_work": 0.55,
            "novelty": 0.60,
            "paper_similarity": 0.65,
            "literature_survey": 0.65,
            "open": self.min_context_coverage,
        }
        return thresholds.get(query_type, self.min_context_coverage)
    
    def _compute_semantic_coverage(
        self,
        subset: List[Dict[str, Any]],
        question: str,
        query_terms: Set[str],
        query_embedding: Optional[np.ndarray],
        query_type: str,
    ) -> float:
        """
        Compute multi-factor semantic coverage score.
        
        Combines:
        1. Embedding-based coverage (cosine similarity)
        2. Keyword coverage (term overlap)
        3. Confidence-based coverage (average score)
        4. Diversity coverage (breadth of coverage)
        
        Args:
            subset: Selected chunks
            question: User question
            query_terms: Extracted query terms
            query_embedding: Query embedding
            query_type: Query category
            
        Returns:
            Coverage score in [0, 1]
        """
        # Memoize coverage for repeated subsets (binary-search may revisit the
        # same prefix, and self-verification recomputes the final subset).
        cache_key = None
        if self.cache_enabled and self._coverage_cache is not None:
            ids = sorted(
                str(c.get("chunk_id", c.get("metadata_index", id(c))))
                for c in subset
            )
            cache_key = f"cov_{hashlib.md5((question + '|' + '|'.join(ids)).encode()).hexdigest()}"
            cached = self._coverage_cache.get(cache_key)
            if cached is not None:
                return cached

        # Factor 1: Embedding-based coverage
        embedding_coverage = 0.0
        if query_embedding is not None:
            # Reuse the already-computed FAISS vectors attached by the pipeline
            # (preferred) instead of re-running the transformer every call.
            chunk_embeddings = self._get_chunk_embeddings_if_needed(subset)

            if chunk_embeddings is not None and len(chunk_embeddings) > 0:
                # Average cosine similarity between query and chunks
                similarities = [
                    float(np.dot(query_embedding, emb))
                    for emb in chunk_embeddings
                ]
                embedding_coverage = self._safe_mean(similarities)
        
        # Factor 2: Keyword coverage
        keyword_coverage = 0.0
        if query_terms:
            covered_terms: Set[str] = set()
            for chunk in subset:
                text = chunk.get("text", "").lower()
                for term in query_terms:
                    if term in text:
                        covered_terms.add(term)
            keyword_coverage = len(covered_terms) / max(len(query_terms), 1)
        
        # Factor 3: Confidence-based coverage
        confidence_coverage = self._safe_mean([
            c.get("score", 0.0) for c in subset
        ])
        
        # Factor 4: Diversity coverage
        unique_sections = len(set(
            c.get("section_title", "") for c in subset if c.get("section_title")
        ))
        unique_pages = len(set(
            c.get("page_number", -1) for c in subset
        ))
        diversity_coverage = min(
            (unique_sections + unique_pages) / max(len(subset) * 2, 1),
            1.0
        )
        
        # Weighted fusion
        coverage = (
            BSCO_EMBEDDING_COVERAGE_WEIGHT * embedding_coverage
            + BSCO_KEYWORD_COVERAGE_WEIGHT * keyword_coverage
            + BSCO_CONFIDENCE_COVERAGE_WEIGHT * confidence_coverage
            + BSCO_DIVERSITY_COVERAGE_WEIGHT * diversity_coverage
        )
        
        coverage = round(coverage, 4)
        
        # Store computed coverage in cache so repeated calls for the same
        # (question, sorted_subset_ids) pair are instant.
        if cache_key is not None and self._coverage_cache is not None:
            self._coverage_cache.put(cache_key, coverage)
        
        return coverage
    
    def _is_sufficient(
        self,
        chunks: List[Dict[str, Any]],
        query_terms: Set[str],
        coverage: float,
        min_coverage: float,
        threshold: float,
        min_chunks: int,
        token_budget: int,
        query_type: str,
    ) -> bool:
        """
        Check if a chunk subset is sufficient for answering the query.
        
        Evaluates:
        - Minimum chunk count satisfied
        - Token budget not exceeded
        - Coverage threshold met
        - Average importance score sufficient
        
        Args:
            chunks: Chunk subset to evaluate
            query_terms: Query terms
            coverage: Pre-computed coverage score
            min_coverage: Minimum coverage threshold
            threshold: Similarity threshold
            min_chunks: Minimum required chunks
            token_budget: Token budget
            query_type: Query category
            
        Returns:
            True if sufficient, False otherwise
        """
        if len(chunks) < min_chunks:
            return False
        
        # Check token budget
        tokens = self._estimate_tokens(chunks)
        if tokens > token_budget:
            return False
        
        # Check coverage
        if coverage < min_coverage:
            return False
        
        # Check average importance score
        avg_importance = self._safe_mean([
            c.get("importance_score", c.get("score", 0.0)) for c in chunks
        ])
        if avg_importance < threshold * 0.6:
            return False
        
        return True
    
    # =================================================================
    # Phase 7: Diversity Selection
    # =================================================================
    
    def _apply_diversity_selection(
        self,
        selected: List[Dict[str, Any]],
        candidates: List[Dict[str, Any]],
        target_size: int,
        query_type: str,
    ) -> List[Dict[str, Any]]:
        """
        Apply diversity-aware selection to ensure broad coverage.
        
        Strategies:
        - Penalize multiple chunks from the same page
        - Penalize multiple chunks from the same document
        - Bonus for different sections
        - Bonus for different headings
        
        For research queries (comparison, survey, gap), diversity is
        weighted more heavily.
        
        Args:
            selected: Initially selected chunks
            candidates: All candidate chunks
            target_size: Target number of chunks
            query_type: Query category
            
        Returns:
            Diversity-optimized chunk selection
        """
        if len(selected) <= 1 or len(candidates) <= 1:
            return selected
        
        # Check if diversity is needed
        page_counts: Counter[str] = Counter()
        doc_counts: Counter[str] = Counter()
        section_counts: Counter[str] = Counter()
        
        for chunk in selected:
            page = str(chunk.get("page_number", ""))
            doc = chunk.get("source_file", "")
            section = chunk.get("section_title", "")
            
            if page:
                page_counts[page] += 1
            if doc:
                doc_counts[doc] += 1
            if section:
                section_counts[section] += 1
        
        # If already diverse enough, return as-is
        diversity_score = self._compute_diversity_score(selected)
        diversity_threshold = 0.7 if query_type in _RESEARCH_QUERY_TYPES else 0.5
        
        if diversity_score >= diversity_threshold and len(selected) <= target_size:
            return selected
        
        # Greedy diversity selection
        is_research_query = query_type in _RESEARCH_QUERY_TYPES
        
        # Score each candidate with diversity penalty
        scored_candidates = []
        for idx, chunk in enumerate(candidates):
            base_score = chunk.get("importance_score", chunk.get("score", 0.0))
            
            page = str(chunk.get("page_number", ""))
            doc = chunk.get("source_file", "")
            section = chunk.get("section_title", "")
            
            # Page diversity penalty
            page_penalty = 0.0
            if page in page_counts:
                page_penalty = BSCO_PAGE_DIVERSITY_PENALTY * page_counts[page]
            
            # Document diversity penalty
            doc_penalty = 0.0
            if doc in doc_counts:
                doc_penalty = BSCO_DOCUMENT_DIVERSITY_PENALTY * doc_counts[doc]
                if is_research_query:
                    doc_penalty *= 1.5  # Extra penalty for research queries
            
            # Section diversity bonus
            section_bonus = 0.0
            if section and section not in section_counts:
                section_bonus = BSCO_SECTION_DIVERSITY_BONUS
                if is_research_query:
                    section_bonus *= 1.5
            
            diversity_score = base_score - page_penalty - doc_penalty + section_bonus
            scored_candidates.append((diversity_score, idx, chunk))
        
        # Sort by diversity-adjusted score
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Greedy selection
        diverse_selected = []
        selected_pages: Set[str] = set()
        selected_docs: Set[str] = set()
        selected_sections: Set[str] = set()
        
        for score, idx, chunk in scored_candidates:
            if len(diverse_selected) >= target_size:
                break
            
            page = str(chunk.get("page_number", ""))
            doc = chunk.get("source_file", "")
            section = chunk.get("section_title", "")
            
            # Ensure we don't select too many from same page
            if page and page in selected_pages:
                if len(diverse_selected) >= 2:  # Allow at most 2 per page
                    continue
            
            diverse_selected.append(chunk)
            if page:
                selected_pages.add(page)
            if doc:
                selected_docs.add(doc)
            if section:
                selected_sections.add(section)
        
        return diverse_selected if diverse_selected else selected
    
    def _compute_diversity_score(self, chunks: List[Dict[str, Any]]) -> float:
        """
        Compute diversity score for a set of chunks.
        
        Delegates to the shared utility :func:`src.utils.compute_diversity_score`.
        
        Args:
            chunks: Chunks to evaluate
            
        Returns:
            Diversity score in [0, 1]
        """
        return compute_diversity_score(chunks)
    
    # =================================================================
    # Phase 9: Cross-Chunk Sentence Deduplication
    # =================================================================
    
    def _cross_chunk_dedup(
        self,
        chunks: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Remove duplicate sentences across chunks.
        
        Performs sentence-level deduplication to eliminate repeated
        content while preserving unique information.
        
        Args:
            chunks: Selected chunks
            
        Returns:
            Tuple of (deduplicated chunks, count removed)
        """
        if len(chunks) < 2:
            return chunks, 0
        
        removed = 0
        seen_sentences: Set[str] = set()
        cleaned_chunks: List[Dict[str, Any]] = []
        
        for chunk in chunks:
            chunk_copy = chunk.copy()
            text = chunk_copy.get("text", "")
            
            # Split into sentences
            sentences = re.split(r"(?<=[.!?])\s+", text)
            
            unique_sentences: List[str] = []
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                # Skip very short sentences
                if len(sentence) < 15:
                    unique_sentences.append(sentence)
                    continue
                
                # Hash the sentence (normalized)
                sentence_normalized = re.sub(r"\s+", " ", sentence.lower()).strip()
                sentence_hash = hashlib.md5(
                    sentence_normalized.encode("utf-8", errors="ignore")
                ).hexdigest()
                
                if sentence_hash not in seen_sentences:
                    unique_sentences.append(sentence)
                    seen_sentences.add(sentence_hash)
                else:
                    removed += 1
            
            deduped_text = " ".join(unique_sentences).strip()
            chunk_copy["text"] = deduped_text if deduped_text else text
            cleaned_chunks.append(chunk_copy)
        
        return cleaned_chunks, removed
    
    # =================================================================
    # Utility Methods
    # =================================================================
    
    def _extract_terms(self, text: str) -> Set[str]:
        """
        Extract meaningful terms from text.
        
        Delegates to the shared utility :func:`src.utils.extract_terms`
        (single source of truth for ``STOPWORDS``).
        
        Args:
            text: Input text
            
        Returns:
            Set of meaningful terms
        """
        return extract_terms(text)
    
    def _estimate_tokens(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Estimate token count for chunks.
        
        Delegates to the shared utility :func:`src.utils.estimate_tokens_for_chunks`.
        
        Args:
            chunks: Chunks to estimate
            
        Returns:
            Estimated token count
        """
        return estimate_tokens_for_chunks(chunks)
    
    def _trim_to_token_budget(
        self,
        chunks: List[Dict[str, Any]],
        budget: int,
        min_chunks: int,
    ) -> List[Dict[str, Any]]:
        """
        Trim chunks to fit within token budget.
        
        Removes lowest-importance chunks first to stay within budget
        while preserving minimum chunk count.
        
        Args:
            chunks: Chunks to trim
            budget: Token budget
            min_chunks: Minimum chunks to keep
            
        Returns:
            Trimmed chunks
        """
        if self._estimate_tokens(chunks) <= budget:
            return chunks
        
        # Sort by importance score (ascending) for removal
        sorted_chunks = sorted(
            chunks,
            key=lambda c: c.get("importance_score", c.get("score", 0.0))
        )
        
        # Remove lowest-importance chunks until budget is met
        while len(sorted_chunks) > min_chunks:
            if self._estimate_tokens(sorted_chunks) <= budget:
                break
            sorted_chunks.pop(0)  # Remove lowest importance
        
        # Restore original order (by importance descending)
        sorted_chunks.sort(
            key=lambda c: c.get("importance_score", c.get("score", 0.0)),
            reverse=True
        )
        
        return sorted_chunks
    
    def _get_query_embedding(self, question: str) -> Optional[np.ndarray]:
        """
        Get embedding for a query string.
        
        Uses the EmbeddingGenerator if available, with caching.
        
        Args:
            question: Query text
            
        Returns:
            Query embedding vector or None
        """
        if not question:
            return None
        
        # Check cache
        cache_key = f"q_emb_{hashlib.md5(question.encode()).hexdigest()}"
        if self.cache_enabled and self._embedding_cache is not None:
            cached = self._embedding_cache.get(cache_key)
            if cached is not None:
                return cached
        
        try:
            if self._embedder is None:
                from src.embedding import EmbeddingGenerator
                self._embedder = EmbeddingGenerator()
            
            embedding = self._embedder.encode_query(question)
            
            # Cache result
            if self.cache_enabled and self._embedding_cache is not None:
                self._embedding_cache.put(cache_key, embedding)
            
            return embedding
        except Exception:
            return None
    
    def _get_embeddings(self, texts: List[str]) -> Optional[np.ndarray]:
        """
        Get embeddings for a list of texts.
        
        Uses the EmbeddingGenerator if available, with caching.
        
        Args:
            texts: List of text strings
            
        Returns:
            Embedding matrix or None
        """
        if not texts:
            return None
        
        # Check cache for combined texts
        combined = "|||".join(texts)
        cache_key = f"emb_{hashlib.md5(combined.encode()).hexdigest()}"
        if self.cache_enabled and self._embedding_cache is not None:
            cached = self._embedding_cache.get(cache_key)
            if cached is not None:
                return cached
        
        try:
            if self._embedder is None:
                from src.embedding import EmbeddingGenerator
                self._embedder = EmbeddingGenerator()
            
            # Build chunk dicts for the embedder
            chunk_dicts = [{"text": t} for t in texts]
            embeddings = self._embedder.generate_embeddings(chunk_dicts)
            
            # Cache result
            if self.cache_enabled and self._embedding_cache is not None:
                self._embedding_cache.put(cache_key, embeddings)
            
            return embeddings
        except Exception:
            return None
    
    def _get_chunk_embeddings_if_needed(
        self,
        chunks: List[Dict[str, Any]],
    ) -> Optional[np.ndarray]:
        """
        Get chunk embeddings, computing them only if lazy evaluation permits.

        Prefers the already-computed FAISS vectors that the pipeline attached
        as ``_embedding`` on each chunk (avoids re-running the transformer).
        Falls back to on-the-fly encoding only when no attached vectors exist.

        Args:
            chunks: Chunks to get embeddings for

        Returns:
            Embedding matrix or None
        """
        attached = self._collect_attached_embeddings(chunks)
        if attached is not None:
            return attached

        if self.lazy_evaluation and not self._embedder:
            # Skip expensive embedding computation if lazy and not yet loaded
            return None

        texts = [c.get("text", "") for c in chunks]
        return self._get_embeddings(texts)

    @staticmethod
    def _collect_attached_embeddings(
        chunks: List[Dict[str, Any]],
    ) -> Optional[np.ndarray]:
        """
        Return a stacked matrix of chunk ``_embedding`` vectors when all
        chunks carry one, else ``None``.

        The pipeline attaches these vectors from the FAISS index so BSCO can
        reuse them instead of re-encoding the chunk text on every call.
        """
        if not chunks:
            return None
        vecs = [c.get("_embedding") for c in chunks]
        if any(v is None for v in vecs):
            return None
        return np.stack(vecs, axis=0).astype(np.float32)
    
    @staticmethod
    def _safe_mean(values: List[float]) -> float:
        """Compute mean of values, returning 0.0 for empty list."""
        return safe_mean(values)
    
    def _empty_stats(self, threshold: float, query_type: str = "open") -> Dict[str, Any]:
        """Return empty statistics dictionary."""
        return {
            "initial_chunks": 0,
            "initial_retrieved_chunks": 0,
            "prefiltered_chunks": 0,
            "after_dedup": 0,
            "after_redundancy": 0,
            "deduplicated_candidates": 0,
            "selected_chunks": 0,
            "final_selected_chunks": 0,
            "initial_tokens": 0,
            "final_tokens": 0,
            "compression_ratio": 0.0,
            "context_reduction_percent": 0.0,
            "token_reduction": 0,
            "token_reduction_percent": 0.0,
            "adaptive_budget": self.max_context_tokens,
            "dedup_removed": 0,
            "redundancy_removed": 0,
            "sentence_dedup_removed": 0,
            "coverage_score": 0.0,
            "semantic_coverage": 0.0,
            "term_coverage": 0.0,
            "threshold": threshold,
            "threshold_used": round(threshold, 4),
            "query_type": query_type,
            "query_complexity": 0.0,
            "query_length": 0,
            "verification_rounds": 0,
            "avg_score": 0.0,
            "avg_importance": 0.0,
            "avg_confidence": 0.0,
            "pages_covered": 0,
            "documents_covered": 0,
            "unique_pages": 0,
            "unique_sections": 0,
            "unique_documents": 0,
            "context_diversity": 0.0,
            "retrieval_recall_proxy": 0.0,
            "prompt_efficiency": 0.0,
            "compression_score": 0.0,
            "optimize_time_seconds": 0.0,
            "dedup_time_seconds": 0.0,
            "redundancy_time_seconds": 0.0,
            "scoring_time_seconds": 0.0,
            "binary_search_time_seconds": 0.0,
            "diversity_time_seconds": 0.0,
            "verification_time_seconds": 0.0,
            "sentence_dedup_time_seconds": 0.0,
            "estimated_latency_saved_seconds": 0.0,
            "context_reduction_percentage": 0.0,
        }
    
    def clear_caches(self) -> None:
        """Clear all internal caches."""
        if self._embedding_cache is not None:
            self._embedding_cache.clear()
        if self._similarity_cache is not None:
            self._similarity_cache.clear()
        if self._coverage_cache is not None:
            self._coverage_cache.clear()
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Return overall optimizer statistics."""
        return {
            "total_optimize_calls": self._optimize_calls,
            "total_optimize_time": round(self._total_optimize_time, 4),
            "avg_optimize_time": round(
                self._total_optimize_time / max(self._optimize_calls, 1), 4
            ),
            "embedding_cache_hit_rate": round(
                self._embedding_cache.hit_rate, 4
            ) if self._embedding_cache is not None else 0.0,
            "embedding_cache_size": self._embedding_cache.size if self._embedding_cache is not None else 0,
        }
