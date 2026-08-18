"""Tests for utility functions."""

import pytest
import os
import tempfile
from src.utils import (
    validate_file_upload,
    sanitize_filename,
    ValidationError,
    MAX_FILE_SIZE
)


class TestValidateFileUpload:
    """Test file upload validation."""

    def test_valid_pdf(self):
        """Test validation of valid PDF file."""
        is_valid, error = validate_file_upload("document.pdf", 1024)
        assert is_valid is True
        assert error is None

    def test_invalid_extension(self):
        """Test rejection of non-PDF file."""
        is_valid, error = validate_file_upload("document.txt", 1024)
        assert is_valid is False
        assert "not allowed" in error

    def test_file_too_large(self):
        """Test rejection of oversized file."""
        is_valid, error = validate_file_upload("document.pdf", MAX_FILE_SIZE + 1)
        assert is_valid is False
        assert "too large" in error

    def test_empty_file(self):
        """Test rejection of empty file."""
        is_valid, error = validate_file_upload("document.pdf", 0)
        assert is_valid is False
        assert "empty" in error

    def test_path_traversal(self):
        """Test rejection of path traversal attempts."""
        is_valid, error = validate_file_upload("../../../etc/passwd", 1024)
        assert is_valid is False
        assert "Invalid filename" in error

    def test_invalid_characters(self):
        """Test rejection of filenames with invalid characters."""
        is_valid, error = validate_file_upload("document<script>.pdf", 1024)
        assert is_valid is False
        assert "invalid characters" in error


class TestSanitizeFilename:
    """Test filename sanitization."""

    def test_basic_sanitization(self):
        """Test basic filename sanitization."""
        assert sanitize_filename("document.pdf") == "document.pdf"

    def test_remove_path(self):
        """Test removal of path components."""
        assert sanitize_filename("/path/to/document.pdf") == "document.pdf"
        assert sanitize_filename("../document.pdf") == "document.pdf"

    def test_remove_special_chars(self):
        """Test removal of special characters."""
        assert sanitize_filename("document<script>.pdf") == "document.pdf"
        assert sanitize_filename("doc@#$%ument.pdf") == "document.pdf"

    def test_add_extension(self):
        """Test adding PDF extension if missing."""
        assert sanitize_filename("document") == "document.pdf"

    def test_empty_filename(self):
        """Test handling of empty filename."""
        assert sanitize_filename("") == "document.pdf"

    def test_preserve_safe_chars(self):
        """Test preservation of safe characters."""
        assert sanitize_filename("my_document-v1.0.pdf") == "my_document-v1.0.pdf"
