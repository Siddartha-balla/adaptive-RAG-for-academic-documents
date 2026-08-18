"""Tests for PDF processor."""

import pytest
import tempfile
import os
from src.pdf_processor import PDFProcessor
from src.utils import ValidationError, PDFProcessingError
from unittest.mock import Mock, patch


class TestPDFProcessor:
    """Test PDF processing functionality."""

    @pytest.fixture
    def processor(self):
        """Create PDFProcessor instance."""
        return PDFProcessor()

    def test_clean_text(self, processor):
        """Test text cleaning."""
        text = "This is a test.\n\n\nWith extra   spaces.\n"
        cleaned = processor.clean_text(text)
        assert "\n\n\n" not in cleaned
        assert "  " not in cleaned

    def test_clean_text_hyphenated(self, processor):
        """Test hyphenated line break removal."""
        text = "This is a hyphen-\nated word."
        cleaned = processor.clean_text(text)
        assert "hyphenated" in cleaned

    def test_extract_text_file_not_found(self, processor):
        """Test error when file doesn't exist."""
        with pytest.raises(ValidationError):
            processor.extract_text("nonexistent.pdf")

    def test_extract_text_empty_file(self, processor, tmp_path):
        """Test error when file is empty."""
        empty_file = tmp_path / "empty.pdf"
        empty_file.write_bytes(b"")
        with pytest.raises(ValidationError):
            processor.extract_text(str(empty_file))

    def test_extract_text_too_large(self, processor, tmp_path):
        """Test error when file is too large."""
        from src.utils import MAX_FILE_SIZE
        large_file = tmp_path / "large.pdf"
        large_file.write_bytes(b"x" * (MAX_FILE_SIZE + 1))
        with pytest.raises(ValidationError):
            processor.extract_text(str(large_file))


class TestCleanText:
    """Test text cleaning edge cases."""

    @pytest.fixture
    def processor(self):
        return PDFProcessor()

    def test_empty_string(self, processor):
        """Test cleaning empty string."""
        assert processor.clean_text("") == ""

    def test_none_input(self, processor):
        """Test cleaning None input."""
        assert processor.clean_text(None) == ""

    def test_tabs_to_spaces(self, processor):
        """Test tab to space conversion."""
        text = "word1\tword2"
        cleaned = processor.clean_text(text)
        assert "\t" not in cleaned
        assert "word1 word2" in cleaned
