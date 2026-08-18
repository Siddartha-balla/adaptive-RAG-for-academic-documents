"""Tests for query classifier."""

import pytest
from src.query_classifier import QueryClassifier, QueryProfile


class TestQueryClassifier:
    """Test query classification."""

    @pytest.fixture
    def classifier(self):
        return QueryClassifier()

    def test_classify_definition(self, classifier):
        """Test definition query classification."""
        result = classifier.classify("What is machine learning?")
        assert result.query_type == "definition"

    def test_classify_explanation(self, classifier):
        """Test explanation query classification."""
        result = classifier.classify("How does neural network work?")
        assert result.query_type == "explanation"

    def test_classify_comparison(self, classifier):
        """Test comparison query classification."""
        result = classifier.classify("Compare CNN and RNN")
        assert result.query_type == "comparison"

    def test_classify_advantages(self, classifier):
        """Test advantages query classification."""
        result = classifier.classify("What are the benefits of using transformers?")
        assert result.query_type == "advantages"

    def test_classify_disadvantages(self, classifier):
        """Test disadvantages query classification."""
        result = classifier.classify("What are the limitations of this approach?")
        assert result.query_type == "disadvantages"

    def test_classify_algorithm(self, classifier):
        """Test algorithm query classification."""
        result = classifier.classify("What is the time complexity of quicksort?")
        assert result.query_type == "algorithm"

    def test_classify_code(self, classifier):
        """Test code query classification."""
        result = classifier.classify("Write a function to sort an array")
        assert result.query_type == "code"

    def test_classify_numerical(self, classifier):
        """Test numerical query classification."""
        result = classifier.classify("Calculate the accuracy")
        assert result.query_type == "numerical"

    def test_classify_follow_up(self, classifier):
        """Test follow-up detection."""
        result = classifier.classify("How does it work?")
        assert result.has_follow_up is True

    def test_token_count(self, classifier):
        """Test token counting."""
        result = classifier.classify("What is machine learning?")
        assert result.token_count == 4

    def test_complexity_score(self, classifier):
        """Test complexity scoring."""
        result1 = classifier.classify("What is ML?")
        result2 = classifier.classify("Compare the advantages and disadvantages of using CNN versus RNN for image classification tasks")
        assert result2.complexity > result1.complexity

    def test_open_fallback(self, classifier):
        """Test fallback to 'open' for unmatched queries."""
        result = classifier.classify("Tell me something interesting")
        assert result.query_type == "open"


class TestQueryProfile:
    """Test QueryProfile dataclass."""

    def test_profile_immutable(self):
        """Test that QueryProfile is immutable."""
        profile = QueryProfile("definition", 0.5, 10, False)
        with pytest.raises(Exception):  # Frozen dataclass raises on assignment
            profile.query_type = "explanation"
