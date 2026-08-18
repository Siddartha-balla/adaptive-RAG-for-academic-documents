"""
Adaptive Chunking Module
------------------------
Creates intelligent chunks for academic documents.

Improvements in this version
-----------------------------
Better sentence boundary detection
    Handles common abbreviations (e.g., "Dr.", "e.g.", "i.e.", "Fig.", "et al.")
    so they don't trigger false sentence splits.

Reference/citation detection
    split_into_blocks now marks blocks that contain academic references
    (e.g. "[1]", "(Author, 2020)") with a "reference" block type, which
    helps the chunker keep references with their surrounding context.
"""

import re

from config import (
    MIN_CHUNK_SIZE,
    MAX_CHUNK_SIZE,
    CHUNK_OVERLAP
)

# Abbreviations that should not be treated as sentence-ending punctuation.
_ABBREVIATIONS = frozenset({
    "dr", "mr", "mrs", "ms", "prof", "sr", "jr", "dept", "est",
    "approx", "fig", "eq", "al", "vs", "etc", "i.e", "e.g",
    "vol", "no", "pp", "sec", "ch", "ex", "ref",
})


class AdaptiveChunker:

    def __init__(
        self,
        min_size=MIN_CHUNK_SIZE,
        max_size=MAX_CHUNK_SIZE,
        overlap=CHUNK_OVERLAP
    ):

        self.min_size = min_size
        self.max_size = max_size
        self.overlap = overlap

    def split_into_sentences(self, text):
        """
        Split paragraph into sentences, handling common abbreviations.
        """
        # Protect known abbreviations from being split
        protected = text
        for abbr in _ABBREVIATIONS:
            protected = protected.replace(abbr + ".", abbr + "\u0000")

        sentences = re.split(r'(?<=[.!?])\s+', protected)

        # Restore dots in abbreviations
        result = []
        for s in sentences:
            s = s.replace("\u0000", ".")
            s = s.strip()
            if s:
                result.append(s)

        return result

    def split_into_blocks(self, text):
        """Split extracted page text into structure-aware blocks."""

        raw_blocks = re.split(r"\n{2,}|(?<=\.)\s+(?=[A-Z][A-Za-z ]{2,}:)", text)
        blocks = []
        current_section = None

        for raw_block in raw_blocks:
            block = raw_block.strip()

            if not block:
                continue

            block_type = self.detect_block_type(block)

            if block_type == "heading":
                current_section = block[:120]

            blocks.append(
                {
                    "text": block,
                    "type": block_type,
                    "section_title": current_section,
                }
            )

        return blocks

    def detect_block_type(self, text):
        """Classify a text block from academic PDF extraction."""

        stripped = text.strip()

        if re.match(r"^(\d+(\.\d+)*\.?\s+)?[A-Z][A-Z0-9\s\-:]{4,}$", stripped) and len(stripped) < 140:
            return "heading"

        if re.match(r"^(\u2022|-|\*|\d+\.|\([a-zA-Z0-9]\))\s+", stripped):
            return "list"

        if re.search(r"\b(table|figure|fig\.)\s*\d+", stripped.lower()):
            return "figure_or_table"

        if stripped.count("|") >= 2 or len(re.findall(r"\s{2,}", stripped)) >= 3:
            return "table"

        # Detect reference sections (dense citation markers)
        ref_count = len(re.findall(r"\[\d+\]|\([A-Z][a-z]+.*?\d{4}\)", stripped))
        if ref_count >= 3 and len(stripped) < 300:
            return "reference"

        return "paragraph"

    def adaptive_chunking(self, pages):

        chunks = []

        chunk_id = 1

        for page in pages:

            page_number = page["page_number"]

            source_file = page.get("source_file", "Uploaded PDF")
            blocks = self.split_into_blocks(page["text"])

            current_chunk = ""
            current_types = []
            current_section = None

            for block in blocks:
                block_text = block["text"]
                block_units = (
                    [block_text]
                    if block["type"] in {"heading", "list", "table", "figure_or_table", "reference"}
                    else self.split_into_sentences(block_text)
                )

                for sentence in block_units:

                    if not sentence:
                        continue

                    if block["type"] == "heading" and current_chunk:
                        chunks.append(self._make_chunk(
                            chunk_id,
                            page_number,
                            current_chunk,
                            source_file,
                            current_types,
                            current_section
                        ))
                        chunk_id += 1
                        current_chunk = ""
                        current_types = []

                    current_section = block.get("section_title") or current_section

                    if len(current_chunk) + len(sentence) <= self.max_size:
                        current_chunk += " " + sentence
                        current_types.append(block["type"])
                    else:
                        if current_chunk and len(current_chunk) >= self.min_size:
                            chunks.append(self._make_chunk(
                                chunk_id,
                                page_number,
                                current_chunk,
                                source_file,
                                current_types,
                                current_section
                            ))
                            chunk_id += 1
                            overlap_text = current_chunk[-self.overlap:]
                            current_chunk = overlap_text + " " + sentence
                            current_types = [block["type"]]
                        else:
                            current_chunk += " " + sentence
                            current_types.append(block["type"])

            if current_chunk.strip():
                chunks.append(self._make_chunk(
                    chunk_id,
                    page_number,
                    current_chunk,
                    source_file,
                    current_types,
                    current_section
                ))

                chunk_id += 1

        return chunks

    def _make_chunk(self, chunk_id, page_number, text, source_file, chunk_types, section_title):
        type_counts = {
            chunk_type: chunk_types.count(chunk_type)
            for chunk_type in set(chunk_types)
        }
        dominant_type = max(type_counts, key=type_counts.get) if type_counts else "paragraph"

        return {
            "chunk_id": chunk_id,
            "page_number": page_number,
            "source_file": source_file,
            "section_title": section_title,
            "chunk_type": dominant_type,
            "text": text.strip(),
            "length": len(text)
        }

    def chunk_pages(self, pages):
        """
        Public wrapper used by the pipeline.
        """

        return self.adaptive_chunking(pages)

