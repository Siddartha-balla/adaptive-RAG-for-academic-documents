"""
Configuration file for Adaptive RAG Academic Chatbot
SaaS-grade configuration with environment variable support
"""

import os
from pathlib import Path

# ==========================================================
# Project Paths
# ==========================================================

BASE_DIR = Path(__file__).parent.absolute()

UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", str(BASE_DIR / "uploads"))
VECTOR_DB_FOLDER = os.getenv("VECTOR_DB_FOLDER", str(BASE_DIR / "vector_db"))
PROCESSED_DATA_FOLDER = os.getenv("PROCESSED_DATA_FOLDER", str(BASE_DIR / "processed_data"))

# Create folders automatically if they don't exist
Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
Path(VECTOR_DB_FOLDER).mkdir(parents=True, exist_ok=True)
Path(PROCESSED_DATA_FOLDER).mkdir(parents=True, exist_ok=True)

# ==========================================================
# PDF Processing
# ==========================================================

SUPPORTED_FILE_TYPES = os.getenv("SUPPORTED_FILE_TYPES", "pdf").split(",")
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))

# ==========================================================
# Adaptive Chunking
# ==========================================================

MIN_CHUNK_SIZE = int(os.getenv("MIN_CHUNK_SIZE", "150"))
MAX_CHUNK_SIZE = int(os.getenv("MAX_CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# ==========================================================
# Embedding Model
# ==========================================================

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
DEVICE = os.getenv("DEVICE", "cpu")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))

# ==========================================================
# Vector Database
# ==========================================================

FAISS_INDEX_FILE = os.path.join(VECTOR_DB_FOLDER, "faiss_index.bin")
METADATA_FILE = os.path.join(VECTOR_DB_FOLDER, "chunk_metadata.pkl")
TOP_K = int(os.getenv("TOP_K", "10"))

# ==========================================================
# Binary Search Context Optimizer (BSCO)
# ==========================================================

SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.70"))
# Smaller prompt budget = smaller KV cache = substantially faster LLM calls on
# CPU. Reduced from 1400 -> 900; the selected chunks are still enough for a
# citation-led answer and this typically shaves a noticeable amount off latency.
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "900"))
MIN_CONTEXT_COVERAGE = float(os.getenv("MIN_CONTEXT_COVERAGE", "0.62"))

# BSCO — Adaptive Binary Search
BSCO_EMBEDDING_COVERAGE_WEIGHT = float(os.getenv("BSCO_EMBEDDING_COVERAGE_WEIGHT", "0.40"))
BSCO_KEYWORD_COVERAGE_WEIGHT = float(os.getenv("BSCO_KEYWORD_COVERAGE_WEIGHT", "0.30"))
BSCO_CONFIDENCE_COVERAGE_WEIGHT = float(os.getenv("BSCO_CONFIDENCE_COVERAGE_WEIGHT", "0.20"))
BSCO_DIVERSITY_COVERAGE_WEIGHT = float(os.getenv("BSCO_DIVERSITY_COVERAGE_WEIGHT", "0.10"))

# BSCO — Deduplication & Redundancy
BSCO_DEDUP_THRESHOLD = float(os.getenv("BSCO_DEDUP_THRESHOLD", "0.92"))
BSCO_REDUNDANCY_THRESHOLD = float(os.getenv("BSCO_REDUNDANCY_THRESHOLD", "0.82"))
BSCO_SENTENCE_DEDUP_ENABLED = os.getenv("BSCO_SENTENCE_DEDUP_ENABLED", "True").lower() == "true"

# BSCO — Token Budget
BSCO_MIN_TOKEN_BUDGET = int(os.getenv("BSCO_MIN_TOKEN_BUDGET", "300"))
BSCO_MAX_TOKEN_BUDGET = int(os.getenv("BSCO_MAX_TOKEN_BUDGET", "2800"))
BSCO_BUDGET_OVERHEAD_FACTOR = float(os.getenv("BSCO_BUDGET_OVERHEAD_FACTOR", "0.15"))

# BSCO — Importance Scoring
BSCO_IMPORTANCE_SEMANTIC_WEIGHT = float(os.getenv("BSCO_IMPORTANCE_SEMANTIC_WEIGHT", "0.25"))
BSCO_IMPORTANCE_QUERY_RELEVANCE_WEIGHT = float(os.getenv("BSCO_IMPORTANCE_QUERY_RELEVANCE_WEIGHT", "0.20"))
BSCO_IMPORTANCE_SECTION_WEIGHT = float(os.getenv("BSCO_IMPORTANCE_SECTION_WEIGHT", "0.15"))
BSCO_IMPORTANCE_HEADING_WEIGHT = float(os.getenv("BSCO_IMPORTANCE_HEADING_WEIGHT", "0.10"))
BSCO_IMPORTANCE_CITATION_WEIGHT = float(os.getenv("BSCO_IMPORTANCE_CITATION_WEIGHT", "0.10"))
BSCO_IMPORTANCE_RESEARCH_WEIGHT = float(os.getenv("BSCO_IMPORTANCE_RESEARCH_WEIGHT", "0.10"))
BSCO_IMPORTANCE_RECENCY_WEIGHT = float(os.getenv("BSCO_IMPORTANCE_RECENCY_WEIGHT", "0.10"))

# BSCO — Diversity
BSCO_PAGE_DIVERSITY_PENALTY = float(os.getenv("BSCO_PAGE_DIVERSITY_PENALTY", "0.35"))
BSCO_DOCUMENT_DIVERSITY_PENALTY = float(os.getenv("BSCO_DOCUMENT_DIVERSITY_PENALTY", "0.40"))
BSCO_SECTION_DIVERSITY_BONUS = float(os.getenv("BSCO_SECTION_DIVERSITY_BONUS", "0.15"))

# BSCO — Self-Verification
BSCO_SELF_VERIFY_ENABLED = os.getenv("BSCO_SELF_VERIFY_ENABLED", "True").lower() == "true"
BSCO_VERIFICATION_MIN_COVERAGE = float(os.getenv("BSCO_VERIFICATION_MIN_COVERAGE", "0.50"))
BSCO_VERIFICATION_MAX_ITERATIONS = int(os.getenv("BSCO_VERIFICATION_MAX_ITERATIONS", "3"))

# BSCO — Performance
BSCO_CACHE_ENABLED = os.getenv("BSCO_CACHE_ENABLED", "True").lower() == "true"
BSCO_CACHE_MAX_SIZE = int(os.getenv("BSCO_CACHE_MAX_SIZE", "256"))
BSCO_LAZY_EVALUATION = os.getenv("BSCO_LAZY_EVALUATION", "True").lower() == "true"

# ==========================================================
# Hybrid Retrieval and Reranking
# ==========================================================

HYBRID_DENSE_WEIGHT = float(os.getenv("HYBRID_DENSE_WEIGHT", "0.55"))
HYBRID_BM25_WEIGHT = float(os.getenv("HYBRID_BM25_WEIGHT", "0.30"))
HYBRID_KEYWORD_WEIGHT = float(os.getenv("HYBRID_KEYWORD_WEIGHT", "0.15"))
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "12"))
CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", None)

# ==========================================================
# Ollama / LLM
# ==========================================================

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
# Lower output ceiling = dramatically faster CPU inference.
# Default reduced 2048 -> 600 (a concise, citation-led answer is fully
# achievable within this budget and typically cuts latency by 3-4x).
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "600"))

# A real per-call timeout so a stuck Ollama server cannot hang the UI
# indefinitely. Customise via OLLAMA_TIMEOUT (seconds).
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "180"))

# Ollama context window passed via num_ctx. 8192 is comfortable for the
# selected chunks + conversation history without wasting CPU on a huge KV cache.
OLLAMA_CONTEXT_WINDOW = int(os.getenv("OLLAMA_CONTEXT_WINDOW", "8192"))

# Number of previous Ollama requests to keep the model hot in memory.
# Higher values keep the model resident (faster follow-ups) at the cost of RAM.
# Set OLLAMA_KEEP_ALIVE=0 to unload after every request.
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")

# CPU inference levers — pass through to Ollama so it can use more cores.
# Set OLLAMA_NUM_THREADS to your physical/core count (e.g. 8). Leave unset
# (None) to let Ollama autodetect. OLLAMA_NUM_PARALLEL controls how many
# requests can share the loaded model concurrently.
OLLAMA_NUM_THREADS = os.getenv("OLLAMA_NUM_THREADS")
OLLAMA_NUM_PARALLEL = int(os.getenv("OLLAMA_NUM_PARALLEL", "1"))

# ==========================================================
# Streamlit
# ==========================================================

APP_TITLE = os.getenv("APP_TITLE", "Scholar AI")
APP_ICON = os.getenv("APP_ICON", "🔬")

# ==========================================================
# Confidence Score
# ==========================================================

MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.50"))

# ==========================================================
# Voice Configuration
# ==========================================================

# STT model — "small" (≈460 MB at int8) improves multilingual accuracy
# (English + Telugu) while remaining comfortably within 8 GB RAM.
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "small")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

PIPER_VOICES_DIR = os.getenv("PIPER_VOICES_DIR", str(BASE_DIR / "models" / "voices"))

# Voice mode logic — Auto mode runs forced en/te passes when the auto-detect
# confidence is below this threshold OR the utterance is shorter than this
# many words. The best transcription is selected by comparing average log
# probability, no-speech probability and compression ratio.
VOICE_LANGUAGE_CONFIDENCE_THRESHOLD = float(
    os.getenv("VOICE_LANGUAGE_CONFIDENCE_THRESHOLD", "0.80")
)
VOICE_SHORT_UTTERANCE_WORDS = int(os.getenv("VOICE_SHORT_UTTERANCE_WORDS", "5"))

# ==========================================================
# Logging
# ==========================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "logs/app.log")

# ==========================================================
# Environment
# ==========================================================

ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
DEBUG = os.getenv("DEBUG", "True").lower() == "true"
