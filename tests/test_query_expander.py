"""Tests for query expander."""

import pytest
from src.query_expander import QueryExpander, ExpansionResult


class TestQueryExpander:
    """Test query expansion."""

    @pytest.fixture
    def expander(self):
        return QueryExpander()

    def test_abbreviation_expansion(self, expander):
        """Test abbreviation expansion."""
        result = expander.expand("What is ML?")
        assert "machine learning" in result.expanded_query.lower()
        assert "machine learning" in result.abbreviations_found

    def test_ai_expansion(self, expander):
        """Test AI abbreviation expansion."""
        result = expander.expand("What is AI?")
        assert "artificial intelligence" in result.expanded_query.lower()

    def test_rag_expansion(self, expander):
        """Test RAG abbreviation expansion."""
        result = expander.expand("Explain RAG")
        assert "retrieval augmented generation" in result.expanded_query.lower()

    def test_synonym_expansion(self, expander):
        """Test synonym expansion."""
        result = expander.expand("What are the advantages?")
        assert "benefits" in result.expanded_query.lower() or "merits" in result.expanded_query.lower()

    def test_reverse_lookup(self, expander):
        """Test reverse lookup (full form to abbreviation)."""
        result = expander.expand("What is machine learning?")
        assert "ml" in result.expanded_query.lower()

    def test_no_expansion_needed(self, expander):
        """Test query that doesn't need expansion."""
        result = expander.expand("simple query")
        # Should still return original query
        assert result.original_query in result.expanded_query

    def test_expansion_result_structure(self, expander):
        """Test ExpansionResult structure."""
        result = expander.expand("What is ML?")
        assert isinstance(result, ExpansionResult)
        assert result.original_query == "What is ML?"
        assert len(result.expansion_terms) > 0
        assert isinstance(result.abbreviations_found, list)
        assert isinstance(result.synonyms_found, list)

    def test_multiple_abbreviations(self, expander):
        """Test expansion with multiple abbreviations."""
        result = expander.expand("AI and ML")
        assert "artificial intelligence" in result.expanded_query.lower()
        assert "machine learning" in result.expanded_query.lower()

    def test_get_expansion_terms_convenience(self, expander):
        """Test convenience method for getting expansion terms."""
        terms = expander.get_expansion_terms("What is ML?")
        assert isinstance(terms, list)
        assert len(terms) > 0

    def test_case_insensitive(self, expander):
        """Test case-insensitive matching."""
        result1 = expander.expand("What is ML?")
        result2 = expander.expand("What is ml?")
        assert len(result1.expansion_terms) == len(result2.expansion_terms)

    def test_stopwords_not_expanded(self, expander):
        """Test that stopwords are not expanded."""
        result = expander.expand("the and is")
        # Should not add expansions for stopwords
        assert len(result.expansion_terms) == 0
