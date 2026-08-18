"""Tests for confidence estimator."""

import pytest
from src.confidence import ConfidenceEstimator


class TestConfidenceEstimator:
    """Test confidence estimation."""

    @pytest.fixture
    def estimator(self):
        return ConfidenceEstimator()

    def test_empty_chunks(self, estimator):
        """Test confidence with no chunks."""
        result = estimator.calculate([])
        assert result["score"] == 0.0
        assert result["level"] == "Low"

    def test_chunks_no_scores(self, estimator):
        """Test confidence with chunks but no scores."""
        chunks = [{"text": "sample text"}]
        result = estimator.calculate(chunks)
        assert result["score"] == 0.0

    def test_high_confidence(self, estimator):
        """Test high confidence scenario."""
        chunks = [
            {"text": "machine learning is a subset of artificial intelligence", "score": 0.9},
            {"text": "neural networks are used in deep learning", "score": 0.85},
        ]
        result = estimator.calculate(chunks)
        assert result["score"] > 0.7
        assert result["level"] in ["High", "Very High"]

    def test_low_confidence(self, estimator):
        """Test low confidence scenario."""
        chunks = [
            {"text": "random text", "score": 0.3},
            {"text": "unrelated content", "score": 0.2},
        ]
        result = estimator.calculate(chunks)
        assert result["score"] < 0.6

    def test_very_high_level(self, estimator):
        """Test very high confidence level threshold."""
        chunks = [{"text": "exact match", "score": 0.95}]
        result = estimator.calculate(chunks)
        if result["score"] >= 0.85:
            assert result["level"] == "Very High"

    def test_keyword_coverage(self, estimator):
        """Test keyword coverage component."""
        chunks = [{"text": "machine learning algorithms"}]
        result = estimator.calculate(chunks, question="What is machine learning?")
        assert "keyword_coverage" in result["components"]

    def test_retrieval_consistency(self, estimator):
        """Test retrieval consistency component."""
        chunks = [
            {"text": "text1", "score": 0.8},
            {"text": "text2", "score": 0.82},
        ]
        result = estimator.calculate(chunks)
        assert "retrieval_consistency" in result["components"]

    def test_chunk_agreement(self, estimator):
        """Test chunk agreement component."""
        chunks = [
            {"text": "machine learning is important"},
            {"text": "machine learning applications are vast"},
        ]
        result = estimator.calculate(chunks)
        assert "chunk_agreement" in result["components"]

    def test_citation_coverage(self, estimator):
        """Test citation coverage component."""
        chunks = [
            {"text": "text1", "page_number": 1},
            {"text": "text2", "page_number": 2},
        ]
        citations = {"pages": [1, 2]}
        result = estimator.calculate(chunks, citations=citations)
        assert result["components"]["citation_coverage"] == 1.0
