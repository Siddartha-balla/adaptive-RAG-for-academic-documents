"""Quick import test to verify all refactored modules load correctly."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing imports...")

# Test shared utilities
from src.utils import (
    tokenize,
    extract_terms,
    safe_mean,
    estimate_tokens,
    estimate_tokens_for_chunks,
    compute_section_relevance,
    section_relevance,
    compute_diversity_score,
    jaccard_similarity,
    STOPWORDS,
    HIGH_VALUE_SECTIONS,
    logger,
)
print("  [OK] src.utils")

# Test citations
from src.citations import CitationGenerator
print("  [OK] src.citations")

# Test confidence
from src.confidence import ConfidenceEstimator
print("  [OK] src.confidence")

# Test self_verifier
from src.self_verifier import SelfVerifier
print("  [OK] src.self_verifier")

# Test reranker
from src.reranker import CrossEncoderReranker
print("  [OK] src.reranker")

# Test adaptive_hybrid_retrieval
from src.adaptive_hybrid_retrieval import AdaptiveHybridRetrieval, BM25Index
print("  [OK] src.adaptive_hybrid_retrieval")

# Verify utility functions work
print("\nTesting utility functions...")
assert tokenize("Hello World!") == ["hello", "world"]
print("  [OK] tokenize")

terms = extract_terms("The quick brown fox jumps over the lazy dog")
assert "quick" in terms
assert "the" not in terms  # stopword
print("  [OK] extract_terms")

assert estimate_tokens("Hello world test") == 4
print("  [OK] estimate_tokens")

sim = jaccard_similarity("hello world", "hello there")
assert 0.0 < sim < 1.0
print(f"  [OK] jaccard_similarity ({sim:.3f})")

rel = section_relevance("Introduction")
assert rel > 0.0
print(f"  [OK] section_relevance ({rel})")

print("\nAll tests passed!")
