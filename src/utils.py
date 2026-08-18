"""
Shared utility functions for validation, logging, tokenization, and runtime health checks.

All shared constants (STOPWORDS, HIGH_VALUE_SECTIONS, etc.) and utility functions
(tokenize, extract_terms, estimate_tokens) live here to avoid duplication across modules.
"""

from __future__ import annotations

import hashlib
import html
import logging
import math
import os
import re
import threading
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Callable, List, Optional, Set, TypeVar

from config import MAX_FILE_SIZE_MB, SUPPORTED_FILE_TYPES

MAX_FILE_SIZE = MAX_FILE_SIZE_MB * 1024 * 1024  # Convert MB to bytes
ALLOWED_EXTENSIONS = {f".{ext}" for ext in SUPPORTED_FILE_TYPES}
SAFE_FILENAME_PATTERN = re.compile(r"^[\w\-.\s]+$")


# ══════════════════════════════════════════════════════════════════════════════
# Shared Constants — Single Source of Truth
# ══════════════════════════════════════════════════════════════════════════════

# Stopwords for keyword extraction (used by citations, confidence, self_verifier, etc.)
STOPWORDS: frozenset[str] = frozenset({
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "with",
    "on", "by", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "to", "of", "in",
    "for", "on", "with", "at", "by", "from", "as", "into", "through",
    "during", "before", "after", "above", "below", "between", "under",
    "again", "further", "then", "once", "here", "there", "when", "where",
    "why", "how", "all", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than",
    "too", "very", "just", "what", "which", "who", "whom",
    "please", "tell", "me", "give", "provide", "show", "list",
    "this", "that", "these", "those",
})

# Academic sections ranked by information value (used by reranker, BSCO)
HIGH_VALUE_SECTIONS: frozenset[str] = frozenset({
    "abstract", "introduction", "conclusion", "conclusions",
    "result", "results", "discussion", "findings",
    "methodology", "methods", "approach", "proposed method",
    "algorithm", "implementation", "experiment", "evaluation",
    "contribution", "contributions", "novelty",
    "formulation", "mathematical model", "problem definition",
    "dataset", "data collection", "experimental setup",
    "ablation study", "comparison", "comparative analysis",
    "state of the art", "related work", "literature review",
})

# Chunk types that should be preserved during compression
PRESERVED_CHUNK_TYPES: frozenset[str] = frozenset({
    "formula", "algorithm", "table", "definition",
    "equation", "pseudocode", "code",
})

# Abbreviations that should not be treated as sentence-ending punctuation
ABBREVIATIONS: frozenset[str] = frozenset({
    "dr", "mr", "mrs", "ms", "prof", "sr", "jr", "dept", "est",
    "approx", "fig", "eq", "al", "vs", "etc", "i.e", "e.g",
    "vol", "no", "pp", "sec", "ch", "ex", "ref",
})

# Ensure logs directory exists before configuring the file handler
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/scholar_ai.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("scholar_ai")

T = TypeVar("T")


# ── Custom Exceptions ─────────────────────────────────────────────────────────

class ValidationError(Exception):
    """Raised when input validation fails."""


class PDFProcessingError(Exception):
    """Raised when PDF processing fails."""


class ModelLoadError(Exception):
    """Raised when model loading fails."""


class OllamaConnectionError(Exception):
    """Raised when Ollama is not reachable."""


# ── Thread-Safe Helpers ────────────────────────────────────────────────────────

class AtomicCounter:
    """Thread-safe counter for tracking metrics."""
    def __init__(self, initial_value: int = 0) -> None:
        self._value = initial_value
        self._lock = threading.Lock()
    
    def increment(self, amount: int = 1) -> int:
        with self._lock:
            self._value += amount
            return self._value
    
    def decrement(self, amount: int = 1) -> int:
        with self._lock:
            self._value -= amount
            return self._value
    
    def get(self) -> int:
        with self._lock:
            return self._value
    
    def reset(self, value: int = 0) -> int:
        with self._lock:
            self._value = value
            return self._value


def synchronized(lock: threading.Lock) -> Callable:
    """Decorator to make a method synchronized with a lock."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            with lock:
                return func(*args, **kwargs)
        return wrapper
    return decorator


# ── File Validation ────────────────────────────────────────────────────────────

def validate_file_upload(
    filename: str,
    file_size: int,
    max_size: int = MAX_FILE_SIZE,
) -> tuple[bool, Optional[str]]:
    """
    Validate an uploaded PDF before it is persisted or processed.
    """
    if not filename or not filename.strip():
        return False, "Invalid filename detected."

    if file_size <= 0:
        return False, "File is empty."

    if file_size > max_size:
        return False, f"File too large. Maximum allowed size is {format_bytes(max_size)}."

    if ".." in filename or filename.startswith(("/", "\\")):
        return False, "Invalid filename detected."

    safe_name = Path(filename).name
    file_ext = Path(safe_name).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        return False, f"File type {file_ext or '<none>'} not allowed. Allowed: {allowed}"

    if not SAFE_FILENAME_PATTERN.match(safe_name):
        return False, "Filename contains invalid characters."

    return True, None


def sanitize_filename(filename: str) -> str:
    """Return a filesystem-safe PDF filename."""
    sanitized = os.path.basename(filename or "")
    sanitized = re.sub(r"[^\w\-.\s]", "", sanitized).strip(". ")

    if not sanitized:
        sanitized = "document.pdf"

    if not sanitized.lower().endswith(".pdf"):
        sanitized += ".pdf"

    if len(sanitized) > 255:
        stem = Path(sanitized).stem
        suffix = Path(sanitized).suffix
        sanitized = stem[: 255 - len(suffix)] + suffix

    return sanitized


# ── Formatting ─────────────────────────────────────────────────────────────────

def format_bytes(size_bytes: int) -> str:
    """Format a byte count as a human-readable string."""
    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"


# ── Input Sanitization ─────────────────────────────────────────────────────────

# SQL injection patterns
SQL_INJECTION_PATTERNS = [
    re.compile(r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|EXEC|UNION)\b)", re.IGNORECASE),
    re.compile(r"(--|;|\/\*|\*\/)"),
    re.compile(r"(\bOR\b|\bAND\b)\s+\d+\s*=\s*\d+", re.IGNORECASE),
]

# XSS patterns
XSS_PATTERNS = [
    re.compile(r"<script[^>]*>.*?</script>", re.IGNORECASE),
    re.compile(r"javascript:", re.IGNORECASE),
    re.compile(r"on\w+\s*=", re.IGNORECASE),
    re.compile(r"<iframe", re.IGNORECASE),
    re.compile(r"<object", re.IGNORECASE),
    re.compile(r"<embed", re.IGNORECASE),
]


def validate_query(query: str) -> tuple[bool, Optional[str]]:
    """
    Validate and sanitize user query.
    
    Returns (is_valid, error_message_or_None).
    """
    if not query or not query.strip():
        return False, "Query cannot be empty."
    
    if len(query) < 3:
        return False, "Query too short (minimum 3 characters)."
    
    if len(query) > 2000:
        return False, "Query too long (maximum 2000 characters)."
    
    for pattern in SQL_INJECTION_PATTERNS:
        if pattern.search(query):
            return False, "Query contains potentially malicious content."
    
    for pattern in XSS_PATTERNS:
        if pattern.search(query):
            return False, "Query contains potentially malicious content."
    
    return True, None


def sanitize_text(text: str) -> str:
    """Sanitize text input (HTML escape, strip null bytes)."""
    sanitized = html.escape(text)
    sanitized = sanitized.replace("\x00", "")
    return sanitized.strip()


def extract_keywords(query: str) -> List[str]:
    """Extract meaningful keywords from query for filtering/search."""
    words = re.findall(r"\b\w+\b", query.lower())
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "shall", "can", "to", "of", "in",
        "for", "on", "with", "at", "by", "from", "as", "into", "through",
    }
    return [w for w in words if len(w) > 2 and w not in stop_words]


# ── Connectivity Checks ────────────────────────────────────────────────────────

def check_ollama_connection(host: str = "http://127.0.0.1:11434") -> bool:
    """Return True when the configured Ollama host responds."""
    try:
        import httpx
        response = httpx.get(f"{host}/api/tags", timeout=5.0)
        return response.status_code == 200
    except Exception as exc:
        logger.warning("Ollama connection check failed: %s", exc)
        return False


def check_model_availability(
    model_name: str,
    host: str = "http://127.0.0.1:11434",
) -> bool:
    """Return True when an Ollama model matching *model_name* is installed."""
    try:
        import httpx
        response = httpx.get(f"{host}/api/tags", timeout=5.0)
        if response.status_code != 200:
            return False
        models = response.json().get("models", [])
        return any(model_name in model.get("name", "") for model in models)
    except Exception as exc:
        logger.warning("Model availability check failed: %s", exc)
        return False


# ── File I/O ───────────────────────────────────────────────────────────────────

def safe_read_file(file_path: str, max_size: int = MAX_FILE_SIZE) -> bytes:
    """Read a local file after existence and size validation."""
    if not os.path.exists(file_path):
        raise ValidationError(f"File not found: {file_path}")

    file_size = os.path.getsize(file_path)
    if file_size > max_size:
        raise ValidationError(f"File too large: {file_size} bytes")

    try:
        with open(file_path, "rb") as handle:
            return handle.read()
    except OSError as exc:
        raise ValidationError(f"Failed to read file: {exc}") from exc


def ensure_directory(path: str) -> None:
    """Create *path* and parent directories when needed."""
    Path(path).mkdir(parents=True, exist_ok=True)


# ── Memory ─────────────────────────────────────────────────────────────────────

def get_memory_usage() -> dict[str, Any]:
    """Return current process memory statistics in MB when psutil is present."""
    try:
        import psutil
        process = psutil.Process()
        mem_info = process.memory_info()
        return {
            "rss": mem_info.rss / 1024 / 1024,
            "vms": mem_info.vms / 1024 / 1024,
            "percent": process.memory_percent(),
        }
    except ImportError:
        logger.debug("psutil not available for memory monitoring")
        return {}
    except Exception as exc:
        logger.debug("Failed to get memory usage: %s", exc)
        return {}


def process_in_chunks(items: List[Any], chunk_size: int = 100):
    """Yield items in chunks to reduce memory pressure."""
    for i in range(0, len(items), chunk_size):
        yield items[i:i + chunk_size]


# ── Hash Helpers ───────────────────────────────────────────────────────────────

def hash_text(text: str, length: int = 8) -> str:
    """Return short hex digest of text for fingerprinting."""
    return hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()[:length]


# ══════════════════════════════════════════════════════════════════════════════
# Shared Tokenization & Text Utilities
# ══════════════════════════════════════════════════════════════════════════════

def tokenize(text: str) -> List[str]:
    """
    Split *text* into lower-case alphanumeric tokens.
    
    Used by reranker, BSCO, adaptive_hybrid_retrieval, and others.
    """
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def extract_terms(text: str, min_length: int = 3) -> Set[str]:
    """
    Extract meaningful terms from *text*: lower-cased, no stopwords, min length filter.
    
    Used by citations, confidence, self_verifier, BSCO, and others.
    """
    terms = set(tokenize(text))
    return {t for t in terms if len(t) >= min_length and t not in STOPWORDS}


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for a text string.
    
    Uses conservative heuristic: ~4 characters per token on average.
    More accurate for academic/English text than character-count alone.
    """
    return len(text) // 4


def estimate_tokens_for_chunks(chunks: List[dict]) -> int:
    """
    Estimate total token count for a list of chunks.
    """
    return sum(estimate_tokens(c.get("text", "")) for c in chunks)


def safe_mean(values: List[float]) -> float:
    """
    Compute mean of *values*, returning 0.0 for empty list.
    """
    if not values:
        return 0.0
    return sum(values) / len(values)


def jaccard_similarity(text1: str, text2: str) -> float:
    """
    Calculate Jaccard similarity between two texts.
    
    Returns a value in [0, 1] where 1 = identical word sets.
    Used by BSCO for redundancy detection.
    """
    words1 = set(tokenize(text1))
    words2 = set(tokenize(text2))
    
    if not words1 or not words2:
        return 0.0
    
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    return intersection / union if union > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Shared Academic Scoring Utilities
# ══════════════════════════════════════════════════════════════════════════════

def compute_section_relevance(section_title: Optional[str]) -> float:
    """
    Return a bonus score [0, 1] based on whether the chunk's section
    heading is a high-value academic section.
    
    Used by reranker and BSCO to avoid duplicating this logic.
    """
    if not section_title:
        return 0.0
    title_lower = section_title.lower()
    for kw in HIGH_VALUE_SECTIONS:
        if kw in title_lower:
            return 0.85
    # Moderate value sections
    if any(kw in title_lower for kw in ["background", "preliminary", "problem", "formulation"]):
        return 0.65
    # Lower value sections
    if any(kw in title_lower for kw in ["appendix", "acknowledgement", "reference", "footnote"]):
        return 0.25
    return 0.4


def compute_heading_relevance(query_terms: Set[str], section_title: Optional[str]) -> float:
    """
    Compute relevance of section heading to the query terms.
    
    When a section heading contains query terms, it signals
    strong topical alignment.
    """
    if not query_terms or not section_title:
        return 0.0
    heading_terms = set(tokenize(section_title))
    overlap = len(query_terms & heading_terms)
    return min(overlap * 0.2, 1.0)


def compute_diversity_score(chunks: List[dict]) -> float:
    """
    Compute diversity score for a set of chunks.
    
    Measures how well-distributed the chunks are across pages, documents,
    and sections. Returns a value in [0, 1].
    """
    if len(chunks) <= 1:
        return 1.0
    
    n = len(chunks)
    
    # Page diversity
    unique_pages = len({c.get("page_number", -1) for c in chunks})
    page_diversity = unique_pages / min(n, max(unique_pages, 1))
    
    # Document diversity
    unique_docs = len({c.get("source_file", "") for c in chunks if c.get("source_file")})
    doc_diversity = unique_docs / min(n, max(unique_docs, 1)) if unique_docs > 0 else 1.0
    
    # Section diversity
    unique_sections = len({c.get("section_title", "") for c in chunks if c.get("section_title")})
    section_diversity = unique_sections / min(n, max(unique_sections, 1)) if unique_sections > 0 else 1.0
    
# Weighted combination
    return (0.4 * page_diversity + 0.35 * doc_diversity + 0.25 * section_diversity)


# Backward-compatible aliases
section_relevance = compute_section_relevance
heading_relevance = compute_heading_relevance
diversity_score = compute_diversity_score
