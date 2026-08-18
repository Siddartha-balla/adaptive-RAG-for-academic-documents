"""
Embedding Module
----------------
Generates embeddings for text chunks and user queries.

Improvements in this version
-----------------------------
In-memory caching
    generate_embeddings() now caches results per unique text hash so that
    re-embedding the same chunk set (e.g. during pipeline re-runs) is
    instantaneous.

Batch size from config
    The encode calls use EMBEDDING_BATCH_SIZE from config.py so users can
    tune the trade-off between speed and memory on their hardware.
"""

from __future__ import annotations

import hashlib
from typing import Optional

from sentence_transformers import SentenceTransformer
import numpy as np

from config import EMBEDDING_MODEL, DEVICE, EMBEDDING_BATCH_SIZE


class EmbeddingGenerator:

    def __init__(self, cache_size: int = 500):

        print("Loading embedding model...")

        self.model = SentenceTransformer(
            EMBEDDING_MODEL,
            device=DEVICE
        )

        self._cache: dict[str, np.ndarray] = {}
        self._cache_size = cache_size

        print("Embedding model loaded.")

    # ===================================================
    # Cache helpers
    # ===================================================

    def _cache_key(self, texts: list[str]) -> str:
        """MD5 of joined texts for cache lookup."""
        combined = "".join(texts).encode("utf-8", errors="ignore")
        return hashlib.md5(combined).hexdigest()

    def _cache_put(self, key: str, embeddings: np.ndarray) -> None:
        if len(self._cache) >= self._cache_size:
            # Evict oldest entry (dict maintains insertion order in Python 3.7+)
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = embeddings

    def _cache_get(self, key: str) -> Optional[np.ndarray]:
        return self._cache.get(key)

    # ===================================================
    # Document Embeddings
    # ===================================================

    def generate_embeddings(self, chunks):

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        if not texts:
            return np.array([], dtype=np.float32)

        # Check cache
        key = self._cache_key(texts)
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=EMBEDDING_BATCH_SIZE,
        )

        # Cache for subsequent calls
        self._cache_put(key, embeddings)

        return embeddings

    # ===================================================
    # Query Embedding
    # ===================================================

    def encode_query(self, query):

        # Memoize per-query embeddings. The same raw question is embedded
        # multiple times per turn (hybrid search, BSCO coverage), so skipping
        # repeat encodes saves a couple of model forward passes per query.
        cache_key = "q_" + hashlib.md5(
            query.encode("utf-8", errors="ignore")
        ).hexdigest()
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        self._cache_put(cache_key, embedding)
        return embedding

    # ===================================================
    # Cache management
    # ===================================================

    def clear_cache(self) -> None:
        """Clear the in-memory embedding cache."""
        self._cache.clear()

