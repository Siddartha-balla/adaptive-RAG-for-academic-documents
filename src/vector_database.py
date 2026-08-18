"""
vector_database.py
------------------
FAISS Vector Database with metadata filtering.

Improvements in this version
-----------------------------
Metadata filter search
    search_with_filters() supports filtering by source_file, section_title
    (substring match), chunk_type, page_number range, and score threshold.

Better stats
    get_stats() now returns per-source_file breakdown and section distribution
    so the Research Dashboard can display richer corpus information.
"""

import os
import pickle
from collections import Counter
from typing import Any, Optional

import faiss
import numpy as np

from config import (
    FAISS_INDEX_FILE,
    METADATA_FILE
)


class VectorDatabase:

    def __init__(self):

        self.index = None
        self.metadata = None

    # =====================================================
    # Build FAISS Index
    # =====================================================

    def build_index(self, embeddings):

        embeddings = np.array(
            embeddings,
            dtype=np.float32
        )

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)

        self.index.add(embeddings)

        print(f"Indexed {self.index.ntotal} vectors.")

    # =====================================================
    # Save Database
    # =====================================================

    def save(self, chunks):

        os.makedirs(
            os.path.dirname(FAISS_INDEX_FILE),
            exist_ok=True
        )

        faiss.write_index(
            self.index,
            FAISS_INDEX_FILE
        )

        with open(
            METADATA_FILE,
            "wb"
        ) as f:

            pickle.dump(chunks, f)

        self.metadata = chunks

        print("Vector database saved.")

    # =====================================================
    # Load Database
    # =====================================================

    def load(self):

        if not os.path.exists(FAISS_INDEX_FILE) or not os.path.exists(METADATA_FILE):
            raise FileNotFoundError(
                "Vector database files were not found. Process a PDF before searching."
            )

        self.index = faiss.read_index(
            FAISS_INDEX_FILE
        )

        with open(
            METADATA_FILE,
            "rb"
        ) as f:

            self.metadata = pickle.load(f)

        print("Vector database loaded.")

    # =====================================================
    # Search
    # =====================================================

    def search(
        self,
        query_embedding,
        top_k=10
    ):

        if self.index is None or self.metadata is None:
            raise RuntimeError(
                "Vector database is not loaded. Process a PDF before asking questions."
            )

        top_k = min(top_k, self.index.ntotal)

        query_embedding = np.array(
            [query_embedding],
            dtype=np.float32
        )

        scores, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for score, idx in zip(
            scores[0],
            indices[0]
        ):

            if idx == -1:
                continue

            chunk = self.metadata[idx].copy()

            chunk["score"] = float(score)

            results.append(chunk)

        return results

    def search_with_filters(
        self,
        query_embedding,
        top_k: int = 10,
        source_file: Optional[str] = None,
        section_filter: Optional[str] = None,
        chunk_type: Optional[str] = None,
        page_min: Optional[int] = None,
        page_max: Optional[int] = None,
        min_score: float = 0.0,
    ) -> list[dict[str, Any]]:
        """
        Search with metadata filters applied after vector retrieval.

        Parameters
        ----------
        query_embedding : np.ndarray
            Query embedding vector.
        top_k : int
            Max results to return after filtering.
        source_file : str, optional
            Only return chunks from this source file.
        section_filter : str, optional
            Substring match against section_title.
        chunk_type : str, optional
            Only return chunks of this type (e.g. "paragraph", "table").
        page_min, page_max : int, optional
            Inclusive page-number range.
        min_score : float
            Minimum similarity score threshold.

        Returns
        -------
        list[dict]
            Filtered results.
        """
        results = self.search(query_embedding, top_k=top_k * 3)

        filtered = []
        for chunk in results:
            if chunk.get("score", 0.0) < min_score:
                continue
            if source_file and chunk.get("source_file") != source_file:
                continue
            if section_filter:
                section = (chunk.get("section_title") or "").lower()
                if section_filter.lower() not in section:
                    continue
            if chunk_type and chunk.get("chunk_type") != chunk_type:
                continue
            page = chunk.get("page_number")
            if page is not None:
                if page_min is not None and page < page_min:
                    continue
                if page_max is not None and page > page_max:
                    continue
            filtered.append(chunk)

        return filtered[:top_k]

    def attach_embeddings(self, chunks: list[dict]) -> list[dict]:
        """
        Reconstruct each chunk's stored FAISS vector and attach it to the
        chunk dict as ``_embedding`` (float32 ndarray).

        This lets downstream stages (BSCO dedup / coverage / importance)
        reuse the already-computed embeddings instead of re-running the
        sentence-transformer encoder on every binary-search iteration.

        Parameters
        ----------
        chunks : list[dict]
            Chunk dicts that carry a ``metadata_index`` key (set by
            AdaptiveHybridRetrieval). Chunks are modified **in place**.

        Returns
        -------
        list[dict]
            The same list (in place), for convenience.
        """
        if self.index is None:
            return chunks

        for chunk in chunks:
            idx = chunk.get("metadata_index")
            if idx is None:
                continue
            try:
                vec = self.index.reconstruct(int(idx))
                chunk["_embedding"] = np.asarray(vec, dtype=np.float32)
            except Exception:
                # Index type may not support reconstruction; skip silently.
                continue
        return chunks

    @staticmethod
    def strip_embeddings(chunks: list[dict]) -> list[dict]:
        """
        Remove the temporary ``_embedding`` key from chunk dicts.

        Called by the pipeline before results are passed to the UI/session
        state so large numpy arrays never leave the pipeline.

        Parameters
        ----------
        chunks : list[dict]
            Chunk dicts.

        Returns
        -------
        list[dict]
            The same list (in place), with ``_embedding`` removed.
        """
        for chunk in chunks:
            chunk.pop("_embedding", None)
        return chunks

    def get_vectors(self, indices: Optional[list[int]] = None):
        """
        Reconstruct raw FAISS vectors for the given metadata *indices*.

        When *indices* is ``None``, returns all vectors in the index.

        This lets the pipeline attach precomputed vectors to retrieved
        chunks so that downstream components (BSCO coverage scoring,
        semantic dedup) can reuse them instead of re-encoding text
        through the transformer.

        Parameters
        ----------
        indices : list[int] or None
            Metadata indices to reconstruct. ``None`` → all vectors.

        Returns
        -------
        np.ndarray
            Float32 array of shape ``(n, dimension)``.
        """
        if self.index is None:
            return np.empty((0, 0), dtype=np.float32)

        n_total = self.index.ntotal
        if n_total == 0:
            return np.empty((0, 0), dtype=np.float32)

        if indices is None:
            indices = list(range(n_total))

        vectors = np.empty((len(indices), self.index.d), dtype=np.float32)
        for dst, idx in enumerate(indices):
            if idx < n_total:
                vectors[dst] = self.index.reconstruct(int(idx))
            else:
                vectors[dst] = np.zeros(self.index.d, dtype=np.float32)
        return vectors

    @staticmethod
    def _attach_vectors_to_chunks(
        chunks: list[dict],
        vectors: np.ndarray,
    ) -> list[dict]:
        """
        Attach the raw embedding vector (``_embedding`` key) to each chunk.

        The BSCO optimizer uses these attached vectors for coverage scoring
        and deduplication, avoiding a costly re-encode through the transformer.
        """
        if vectors is None or len(vectors) == 0 or not chunks:
            return chunks
        out = []
        for i, c in enumerate(chunks):
            c = c.copy() if hasattr(c, "copy") else dict(c)
            if i < len(vectors):
                c["_embedding"] = vectors[i].astype(np.float32)
            out.append(c)
        return out

    def get_stats(self):
        """
        Return rich index statistics for UI and experiments.
        """

        vector_count = 0

        if self.index is not None:
            vector_count = self.index.ntotal

        metadata_count = 0

        if self.metadata is not None:
            metadata_count = len(self.metadata)

        pages = set()
        source_files = set()
        chunk_types: Counter[str] = Counter()
        sections: Counter[str] = Counter()

        if self.metadata:
            for item in self.metadata:
                page = item.get("page_number")
                if page is not None:
                    pages.add(page)
                sf = item.get("source_file")
                if sf:
                    source_files.add(sf)
                ct = item.get("chunk_type")
                if ct:
                    chunk_types[ct] += 1
                sec = item.get("section_title")
                if sec:
                    sections[sec] += 1

        return {
            "vector_count": vector_count,
            "metadata_count": metadata_count,
            "page_count": len(pages),
            "source_files": list(source_files),
            "chunk_type_distribution": dict(chunk_types.most_common(10)),
            "section_distribution": dict(sections.most_common(10)),
        }

    def clear(self):
        """
        Remove persisted FAISS and metadata files.
        """

        for file_path in [FAISS_INDEX_FILE, METADATA_FILE]:
            if os.path.exists(file_path):
                os.remove(file_path)

        self.index = None
        self.metadata = None

