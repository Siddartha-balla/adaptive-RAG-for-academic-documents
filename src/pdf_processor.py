"""
pdf_processor.py
-----------------
PDF Processing Module — extracts text, metadata, and renders page images
from academic PDFs.

Improvements in this version
-----------------------------
Metadata extraction
    extract_metadata() uses heuristics on the first pages to identify the
    paper title (largest text on page 1), authors (lines following title that
    contain commas or 'and'), and abstract section text.  This metadata is
    attached to the result of extract_text() so downstream components can
    display the paper name instead of just the filename.

extract_page_as_image()
    Renders a specific PDF page to a PNG byte-string using PyMuPDF.  Used by
    the Streamlit UI to show evidence highlighting when a citation is clicked.
"""

from __future__ import annotations

import fitz  # PyMuPDF
import os
import re
from typing import Optional


class PDFProcessor:
    """Extract text, metadata, and render pages from academic PDF documents."""

    def clean_text(self, text: str) -> str:
        """
        Normalize extracted PDF text.

        - Rejoins hyphenated line-breaks.
        - Collapses excessive blank lines.
        - Removes mid-sentence newlines (keeps paragraph breaks).
        - Collapses repeated spaces.
        """
        text = re.sub(r"-\n", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
        text = text.replace("\t", " ")
        while "  " in text:
            text = text.replace("  ", " ")
        return text.strip()

    def extract_metadata(self, pdf_path: str) -> dict:
        """
        Extract academic paper metadata (title, authors, abstract) from
        the first few pages using heuristics.

        Parameters
        ----------
        pdf_path : str
            Absolute or relative path to the PDF file.

        Returns
        -------
        dict
            Keys: ``title`` (str or None), ``authors`` (list[str]),
            ``abstract`` (str or None), ``page_count`` (int).
        """
        if not os.path.exists(pdf_path):
            return {"title": None, "authors": [], "abstract": None, "page_count": 0}

        document = fitz.open(pdf_path)
        page_count = len(document)
        title: Optional[str] = None
        authors: list[str] = []
        abstract: Optional[str] = None

        try:
            # --- Extract title from page 1 (largest font-size text block) ---
            if page_count > 0:
                page = document.load_page(0)
                blocks = page.get_text("dict")["blocks"]

                text_blocks = []
                for block in blocks:
                    if block["type"] == 0:  # text block
                        for line in block["lines"]:
                            text = "".join(
                                span["text"] for span in line["spans"]
                            ).strip()
                            if text and len(text) > 10:
                                font_size = line["spans"][0]["size"] if line["spans"] else 0
                                text_blocks.append((font_size, text))

                # Title is typically the largest text on page 1
                if text_blocks:
                    text_blocks.sort(key=lambda x: x[0], reverse=True)
                    possible_title = text_blocks[0][1]
                    # Title should be a few words, not a full paragraph
                    if len(possible_title.split()) <= 25:
                        title = possible_title

            # --- Extract authors from page 1 ---
            if page_count > 0:
                page = document.load_page(0)
                page_text = page.get_text()
                lines = [l.strip() for l in page_text.split("\n") if l.strip()]

                # Find the title position, then look for author lines below it
                start_idx = 0
                if title:
                    for i, line in enumerate(lines):
                        if title.lower() in line.lower():
                            start_idx = i + 1
                            break

                for line in lines[start_idx : start_idx + 8]:
                    # Author lines typically contain commas or "and"
                    if ("," in line or " and " in line) and len(line) < 300:
                        # Split on commas and "and"
                        parts = re.split(r",\s*|\s+and\s+", line)
                        for part in parts:
                            part = part.strip()
                            if part and len(part) > 3:
                                authors.append(part)
                        if authors:
                            break

            # --- Extract abstract ---
            abstract_start = -1
            for page_idx in range(min(page_count, 3)):
                page = document.load_page(page_idx)
                page_text = page.get_text()
                match = re.search(
                    r"\b(abstract|ABSTRACT)\b\s*[.\-:]*\s*\n?(.*?)(?=\n\s*(introduction|INTRODUCTION|1\.\s*|keywords|KEYWORDS)\b)",
                    page_text,
                    re.DOTALL,
                )
                if match:
                    abstract = match.group(2).strip()[:1000]
                    break

        finally:
            document.close()

        return {
            "title": title,
            "authors": authors[:10],
            "abstract": abstract,
            "page_count": page_count,
        }

    def extract_text(self, pdf_path: str) -> list[dict]:
        """
        Extract cleaned text from every page of *pdf_path*.

        Parameters
        ----------
        pdf_path:
            Absolute or relative path to the PDF file.

        Returns
        -------
        list[dict]
            One dictionary per page with keys ``page_number`` (1-indexed)
            and ``text``.  Empty pages are skipped.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        document = fitz.open(pdf_path)
        pages: list[dict] = []

        for page_index in range(len(document)):
            page = document.load_page(page_index)
            text = page.get_text()
            text = self.clean_text(text)

            if not text:
                continue

            pages.append(
                {
                    "page_number": page_index + 1,
                    "text": text,
                }
            )

        document.close()
        return pages

    def extract_page_as_image(
        self,
        pdf_path: str,
        page_number: int,
        dpi: int = 130,
    ) -> Optional[bytes]:
        """
        Render a single PDF page to a PNG byte-string.

        Used by the Streamlit UI to display evidence highlighting when the
        user clicks a citation page button.

        Parameters
        ----------
        pdf_path:
            Absolute or relative path to the PDF file.
        page_number:
            1-indexed page number (matches the ``page_number`` field in
            extracted chunks).
        dpi:
            Resolution of the rendered image.  130 DPI gives a legible
            preview at a reasonable file size.

        Returns
        -------
        bytes or None
            PNG image bytes, or ``None`` if the page or file cannot be found.
        """
        if not pdf_path or not os.path.exists(pdf_path):
            return None

        try:
            document = fitz.open(pdf_path)
            page_index = page_number - 1  # convert to 0-indexed

            if page_index < 0 or page_index >= len(document):
                document.close()
                return None

            page = document.load_page(page_index)
            zoom = dpi / 72.0  # 72 is PyMuPDF's base DPI
            matrix = fitz.Matrix(zoom, zoom)
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            image_bytes = pixmap.tobytes("png")
            document.close()
            return image_bytes

        except Exception as exc:
            print(f"[PDFProcessor] Could not render page {page_number}: {exc}")
            return None

