import unittest

from src.adaptive_retriever import AdaptiveRetriever
from src.binary_search import BinarySearchOptimizer
from src.confidence import ConfidenceEstimator


class ResearchComponentTests(unittest.TestCase):
    def test_adaptive_retriever_classifies_comparison(self):
        retriever = AdaptiveRetriever()

        policy = retriever.get_policy("Compare supervised and unsupervised learning methods.")

        self.assertEqual(policy["query_type"], "comparison")
        self.assertGreaterEqual(policy["top_k"], 12)
        self.assertGreaterEqual(policy["max_chunks"], policy["min_chunks"])

    def test_bsco_returns_small_sufficient_context_with_stats(self):
        chunks = [
            {
                "chunk_id": 1,
                "page_number": 1,
                "score": 0.82,
                "text": "Adaptive retrieval uses query complexity and similarity distribution.",
            },
            {
                "chunk_id": 2,
                "page_number": 2,
                "score": 0.76,
                "text": "The optimizer reduces prompt tokens by selecting enough context.",
            },
            {
                "chunk_id": 3,
                "page_number": 3,
                "score": 0.60,
                "text": "Unrelated implementation details are omitted from prompts.",
            },
        ]
        optimizer = BinarySearchOptimizer(max_context_tokens=100)

        selected, stats = optimizer.optimize(
            chunks,
            question="How does adaptive retrieval use query complexity?",
            max_chunks=3,
            return_stats=True,
        )

        self.assertGreaterEqual(len(selected), 1)
        self.assertLessEqual(len(selected), len(chunks))
        self.assertEqual(stats["initial_retrieved_chunks"], 3)
        self.assertIn("context_reduction_percent", stats)

    def test_confidence_uses_multiple_components(self):
        chunks = [
            {
                "page_number": 1,
                "score": 0.8,
                "cross_encoder_score": 0.7,
                "text": "Retrieval confidence uses citation coverage and verification.",
            }
        ]
        citations = {"pages": [1]}
        verification = {"score": 0.9}

        confidence = ConfidenceEstimator().calculate(
            chunks,
            citations=citations,
            verification=verification,
        )

        self.assertIn(confidence["level"], {"Low", "Medium", "High"})
        self.assertIn("components", confidence)
        self.assertGreater(confidence["score"], 0.0)


if __name__ == "__main__":
    unittest.main()
