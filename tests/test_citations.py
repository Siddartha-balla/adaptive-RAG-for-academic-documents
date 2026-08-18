"""Tests for citation generator."""

import pytest
from src.citations import CitationGenerator


class TestCitationGenerator:
    """Test citation generation."""

    @pytest.fixture
    def generator(self):
        return CitationGenerator()

    def test_generate_empty_chunks(self, generator):
        """Test citation generation with no chunks."""
        result = generator.generate([])
        assert result["pages"] == []
        assert result["total_pages"] == 0
        assert result["total_chunks"] == 0

    def test_generate_basic_citations(self, generator):
        """Test basic citation generation."""
        chunks = [
            {"page_number": 1, "text": "sample text", "score": 0.8},
            {"page_number": 3, "text": "more text", "score": 0.7},
        ]
        result = generator.generate(chunks)
        assert result["pages"] == [1, 3]
        assert result["total_pages"] == 2
        assert result["total_chunks"] == 2
        assert len(result["chunks"]) == 2

    def test_unique_pages(self, generator):
        """Test page deduplication."""
        chunks = [
            {"page_number": 1, "text": "text1"},
            {"page_number": 1, "text": "text2"},
            {"page_number": 2, "text": "text3"},
        ]
        result = generator.generate(chunks)
        assert result["pages"] == [1, 2]
        assert result["total_pages"] == 2
        assert result["total_chunks"] == 3

    def test_citation_text_format(self, generator):
        """Test citation text formatting."""
        chunks = [{"page_number": 1, "text": "text"}]
        result = generator.generate(chunks)
        assert "Page 1" in result["citation_text"]

    def test_no_citations_message(self, generator):
        """Test message when no citations available."""
        result = generator.generate([])
        assert "No citations available" in result["citation_text"]

    def test_chunk_citation_details(self, generator):
        """Test per-chunk citation details."""
        chunks = [
            {
                "page_number": 1,
                "text": "machine learning is important",
                "score": 0.85,
                "section_title": "Introduction",
                "chunk_id": "chunk_1",
                "chunk_type": "paragraph",
            }
        ]
        result = generator.generate(chunks, query="machine learning")
        chunk_citation = result["chunks"][0]
        assert chunk_citation["page"] == 1
        assert chunk_citation["score"] == 0.85
        assert chunk_citation["section"] == "Introduction"
        assert chunk_citation["chunk_type"] == "paragraph"
        assert "matched_keywords" in chunk_citation
        assert "evidence" in chunk_citation

    def test_matched_keywords(self, generator):
        """Test keyword matching."""
        chunks = [{"text": "machine learning algorithms", "page_number": 1}]
        result = generator.generate(chunks, query="What is machine learning?")
        keywords = result["chunks"][0]["matched_keywords"]
        assert "machine" in keywords or "learning" in keywords

    def test_evidence_extraction(self, generator):
        """Test evidence sentence extraction."""
        chunks = [{"text": "Sentence one. Sentence two with keyword. Sentence three.", "page_number": 1}]
        result = generator.generate(chunks, query="keyword")
        evidence = result["chunks"][0]["evidence"]
        assert "keyword" in evidence or len(evidence) > 0

    def test_backward_compatibility(self, generator):
        """Test backward compatibility with legacy keys."""
        chunks = [{"page_number": 1, "text": "text"}]
        result = generator.generate(chunks)
        assert "pages" in result
        assert "citation_text" in result
        assert "total_pages" in result
