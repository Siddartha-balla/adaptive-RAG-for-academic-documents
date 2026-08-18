"""Quick test to verify EnhancedBSCO module imports and basic logic without embedding dependencies."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# First kill any hanging test
from src.enhanced_bsco import EnhancedBSCO
from src.evaluation import EvaluationTracker

print("✓ Module imports successful")

# Test 1: Basic initialization
bsco = EnhancedBSCO()
print(f"✓ BSCO initialized — threshold={bsco.threshold}, max_tokens={bsco.max_context_tokens}")

# Test 2: Empty input
result, stats = bsco.optimize([], question="test", return_stats=True)
assert len(result) == 0
assert stats["initial_chunks"] == 0
assert stats["selected_chunks"] == 0
print(f"✓ Empty optimize — {len(stats)} stats keys returned")

# Test 3: Mock chunks with all metadata
mock_chunks = [
    {"text": "Deep learning is a subset of machine learning that uses neural networks with multiple layers.", "score": 0.85, "page_number": 1, "section_title": "Introduction", "chunk_id": 0, "source_file": "paper1.pdf", "chunk_type": "paragraph"},
    {"text": "Neural networks consist of layers of interconnected neurons that process information.", "score": 0.78, "page_number": 2, "section_title": "Background", "chunk_id": 1, "source_file": "paper1.pdf", "chunk_type": "paragraph"},
    {"text": "Transformers use self-attention mechanisms for sequence processing tasks.", "score": 0.72, "page_number": 3, "section_title": "Methodology", "chunk_id": 2, "source_file": "paper1.pdf", "chunk_type": "paragraph"},
    {"text": "The model achieves state-of-the-art results on benchmark datasets.", "score": 0.68, "page_number": 4, "section_title": "Results", "chunk_id": 3, "source_file": "paper1.pdf", "chunk_type": "paragraph"},
    {"text": "Future work includes extending the model to multimodal learning scenarios.", "score": 0.62, "page_number": 5, "section_title": "Future Work", "chunk_id": 4, "source_file": "paper1.pdf", "chunk_type": "paragraph"},
]

result, stats = bsco.optimize(
    mock_chunks,
    question="What is deep learning and how do transformers work?",
    query_type="explanation",
    return_stats=True
)

print(f"✓ Optimize with {len(mock_chunks)} chunks: {len(result)} selected")
print(f"  - Initial tokens: {stats['initial_tokens']}, Final tokens: {stats['final_tokens']}")
print(f"  - Coverage score: {stats['coverage_score']}")
print(f"  - Compression ratio: {stats['compression_ratio']}")
print(f"  - Context reduction: {stats['context_reduction_percent']}%")
print(f"  - Pages covered: {stats['pages_covered']}")
print(f"  - Documents covered: {stats['documents_covered']}")
print(f"  - Unique sections: {stats['unique_sections']}")
print(f"  - Avg importance: {stats['avg_importance']}")
print(f"  - Avg confidence: {stats['avg_confidence']}")
print(f"  - Query complexity: {stats['query_complexity']}")
print(f"  - Adaptive budget: {stats['adaptive_budget']}")

# Test 4: Dedup stats
assert stats["dedup_removed"] >= 0
assert stats["redundancy_removed"] >= 0
assert stats["sentence_dedup_removed"] >= 0
print(f"✓ Dedup: {stats['dedup_removed']} exact, {stats['redundancy_removed']} redundant, {stats['sentence_dedup_removed']} sentence")

# Test 5: Query type variations
type_results = {}
for qt in ["definition", "factual", "comparison", "literature_survey", "research_gap", "novelty", "future_work", "summary", "algorithm", "methodology"]:
    _, s = bsco.optimize(mock_chunks, question="test question about research", query_type=qt, return_stats=True)
    type_results[qt] = s["adaptive_budget"]
    print(f"  ✓ Query type '{qt}': budget={s['adaptive_budget']}, coverage={s['coverage_score']}, complexity={s['query_complexity']}")

# Test 6: Diversity with multi-document
multi_doc_chunks = [
    {"text": "Paper A introduces a novel deep learning approach.", "score": 0.90, "page_number": 1, "section_title": "Introduction", "chunk_id": 0, "source_file": "paperA.pdf", "chunk_type": "paragraph"},
    {"text": "Paper A methodology uses transformer architectures.", "score": 0.85, "page_number": 3, "section_title": "Methodology", "chunk_id": 1, "source_file": "paperA.pdf", "chunk_type": "paragraph"},
    {"text": "Paper B proposes a different approach using CNNs.", "score": 0.80, "page_number": 1, "section_title": "Introduction", "chunk_id": 2, "source_file": "paperB.pdf", "chunk_type": "paragraph"},
    {"text": "Paper B achieves superior results on ImageNet benchmark.", "score": 0.75, "page_number": 5, "section_title": "Results", "chunk_id": 3, "source_file": "paperB.pdf", "chunk_type": "paragraph"},
    {"text": "Paper C compares both approaches in a comprehensive study.", "score": 0.70, "page_number": 2, "section_title": "Comparison", "chunk_id": 4, "source_file": "paperC.pdf", "chunk_type": "paragraph"},
]
result, stats = bsco.optimize(
    multi_doc_chunks,
    question="Compare the approaches in Paper A and Paper B",
    query_type="comparison",
    return_stats=True
)
print(f"✓ Multi-doc: {len(result)} selected, {stats['documents_covered']} docs, diversity={stats['context_diversity']}")

# Test 7: Evaluation module
et = EvaluationTracker()
eval_metrics = et.calculate(
    mock_chunks,
    result if result else mock_chunks[:2],
    stats,
    response_time=1.5,
    answer="Deep learning is a subset of machine learning.",
    retrieval_time=0.3,
    rerank_time=0.2,
    bsco_time=0.4,
    llm_time=0.6,
)

print(f"✓ Evaluation metrics: {len(eval_metrics)} metrics")
research_keys = ["coverage_score", "compression_score", "context_diversity", 
                 "prompt_efficiency", "token_reduction_percent", "pages_covered",
                 "documents_covered", "avg_importance", "avg_confidence",
                 "verification_rounds", "dedup_removed", "redundancy_removed"]
for k in research_keys:
    print(f"  - {k}: {eval_metrics.get(k, 'N/A')}")

# Test 8: Token budget range validation
assert stats["adaptive_budget"] >= 300  # MIN_TOKEN_BUDGET
assert stats["adaptive_budget"] <= 1400  # max_context_tokens
print("✓ Token budget clamped correctly")

# Test 9: Cache management
bsco.clear_caches()
print("✓ Cache clearing works")
opt_stats = bsco.stats
print(f"✓ Optimizer stats: {opt_stats['total_optimize_calls']} calls, avg {opt_stats['avg_optimize_time']}s")

print("\n" + "=" * 60)
print("ALL TESTS PASSED SUCCESSFULLY!")
print("=" * 60)
