"""
pipeline.py
-----------
Complete Adaptive RAG Pipeline.

Orchestrates every stage from PDF ingestion to verified answer generation.

Changes in this version
-----------------------
question passed to ConfidenceEstimator
    ``confidence_estimator.calculate()`` now receives the original question
    so it can compute the ``keyword_coverage`` component added in this release.

question passed to CitationGenerator
    ``citation_generator.generate()`` receives the question so per-chunk
    citations can highlight matched keywords and best evidence sentences.

Stage timing
    Each pipeline stage (hybrid search, reranking, BSCO, LLM generation) is
    timed independently and passed to :class:`EvaluationTracker` so the
    Research Evaluation Dashboard can show per-stage latency.

PDF path registry
    ``self.pdf_paths`` maps base filename → absolute path so the UI can
    retrieve the original PDF for evidence highlighting.

Result key additions
    ``stage_times`` — dict of per-stage wall-clock seconds.
    ``pdf_paths``   — filename → path mapping for evidence highlighting.
"""

from __future__ import annotations

import os
import time
from typing import Callable, Optional

from src.pdf_processor import PDFProcessor
from src.adaptive_chunker import AdaptiveChunker
from src.embedding import EmbeddingGenerator
from src.vector_database import VectorDatabase

from src.adaptive_hybrid_retrieval import AdaptiveHybridRetrieval
from src.enhanced_bsco import EnhancedBSCO
from src.adaptive_retriever import AdaptiveRetriever
from src.prompt_builder import PromptBuilder
from src.answer_generator import AnswerGenerator
from src.citations import CitationGenerator
from src.confidence import ConfidenceEstimator
from src.reranker import CrossEncoderReranker
from src.self_verifier import SelfVerifier
from src.evaluation import EvaluationTracker
from src.research_gap_detector import ResearchGapDetector
from src.paper_comparator import PaperComparator
from src.novelty_detector import NoveltyDetector
from src.utils import logger

from config import RERANK_TOP_N


class RAGPipeline:
    """
    End-to-end Adaptive RAG pipeline for academic document question-answering.

    Attributes
    ----------
    pdf_paths : dict[str, str]
        Maps each ingested PDF's base filename to its absolute file-system path.
        Used by the UI for PDF evidence highlighting.
    """

    def __init__(self) -> None:
        logger.info("Initializing Adaptive RAG Pipeline")

        # --- Ingestion components ---
        self.pdf_processor = PDFProcessor()
        self.chunker       = AdaptiveChunker()
        self.embedder      = EmbeddingGenerator()
        self.vector_db     = VectorDatabase()

        # --- QA components ---
        self.search = AdaptiveHybridRetrieval(
            embedder=self.embedder,
            vector_db=self.vector_db,
            load_existing=True,
        )
        self.optimizer         = EnhancedBSCO(embedder=self.embedder)
        self.adaptive_retriever = AdaptiveRetriever()
        self.prompt_builder    = PromptBuilder()
        self.answer_generator  = AnswerGenerator()
        self.citation_generator = CitationGenerator()
        self.confidence_estimator = ConfidenceEstimator()
        self.reranker          = CrossEncoderReranker()
        self.verifier          = SelfVerifier()
        self.evaluator         = EvaluationTracker()

        # --- Research modules ---
        self.research_gap_detector = ResearchGapDetector()
        self.paper_comparator = PaperComparator()
        self.novelty_detector = NoveltyDetector()

        # --- State ---
        self.conversation_history: list[dict] = []
        self.pdf_paths: dict[str, str] = {}   # filename → abs path
        self.uploaded_chunks: list[dict] = [] # all chunks for research analysis

        logger.info("Adaptive RAG Pipeline ready")

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    @staticmethod
    def _notify(
        callback: Optional[Callable],
        stage: str,
        progress: float,
        message: str,
    ) -> None:
        if callback:
            callback(stage, progress, message)

    # ------------------------------------------------------------------
    # Database build
    # ------------------------------------------------------------------

    def build_database(
        self,
        pdf_paths: list[str] | str | os.PathLike,
        progress_callback: Optional[Callable] = None,
    ) -> dict:
        """
        Build a FAISS index from one or more academic PDFs.

        Parameters
        ----------
        pdf_paths : list[str] | str | os.PathLike
            Single path or list of paths to PDF files.
        progress_callback : callable, optional
            Optional callable receiving ``(stage, progress, message)``.

        Returns
        -------
        dict
            Build statistics: pages, chunks, embeddings, vectors, files,
            indexing_time.
        """
        logger.info("Building vector database")

        if isinstance(pdf_paths, (str, os.PathLike)):
            pdf_paths = [pdf_paths]

        if not pdf_paths:
            raise ValueError("No PDF files were provided.")

        build_start = time.time()

        self._notify(progress_callback, "extracting", 0.15, "Extracting PDF text")

        # Step 1 — Extract text.
        pages: list[dict] = []
        for pdf_path in pdf_paths:
            document_pages = self.pdf_processor.extract_text(pdf_path)
            base_name = os.path.basename(pdf_path)
            abs_path  = os.path.abspath(pdf_path)
            self.pdf_paths[base_name] = abs_path

            for page in document_pages:
                page["source_file"] = base_name

            pages.extend(document_pages)

        logger.info("Pages extracted: %s", len(pages))

        if not pages:
            raise ValueError("No readable text was found in the PDF.")

        self._notify(
            progress_callback, "chunking", 0.35,
            "Creating adaptive academic chunks",
        )

        # Step 2 — Chunk.
        chunks = self.chunker.chunk_pages(pages)
        logger.info("Chunks created: %s", len(chunks))

        if not chunks:
            raise ValueError("No chunks were created from the PDF.")

        self._notify(
            progress_callback, "embedding", 0.65,
            "Generating BAAI/bge-small-en-v1.5 embeddings",
        )

        # Step 3 — Embed.
        embeddings = self.embedder.generate_embeddings(chunks)
        logger.info("Embeddings generated: %s", len(embeddings))

        self._notify(progress_callback, "indexing", 0.85, "Building FAISS vector index")

        # Step 4 — Index.
        self.vector_db.build_index(embeddings)
        self.vector_db.save(chunks)
        self.search.invalidate()

        # Store all chunks for research analysis
        self.uploaded_chunks = chunks

        elapsed = round(time.time() - build_start, 2)
        self._notify(progress_callback, "ready", 1.0, "Database ready")
        logger.info("Vector database ready in %.2fs", elapsed)

        return {
            "pages":        len(pages),
            "chunks":       len(chunks),
            "embeddings":   len(embeddings),
            "vectors":      self.vector_db.index.ntotal,
            "files":        len(pdf_paths),
            "indexing_time": elapsed,
        }

    # ------------------------------------------------------------------
    # Question answering
    # ------------------------------------------------------------------

    def ask(self, question: str) -> dict:
        """
        Answer *question* using the indexed PDF documents.

        Returns
        -------
        dict
            Complete result including answer, citations, confidence, BSCO
            stats, evaluation metrics, and per-stage timing.
        """
        pipeline_start = time.time()

        # ── 1. Adaptive retrieval policy (first pass, no signals) ──────
        retrieval_policy = self.adaptive_retriever.get_policy(question)

        # ── 2. Hybrid search ───────────────────────────────────────────
        t0 = time.time()
        retrieved_chunks = self.search.hybrid_search(
            question,
            top_k=retrieval_policy["top_k"],
        )
        retrieval_time = round(time.time() - t0, 3)

        # Attach precomputed FAISS vectors so BSCO reuses the stored
        # embeddings instead of re-encoding chunk text through the
        # transformer on every binary-search iteration (major CPU win).
        self.vector_db.attach_embeddings(retrieved_chunks)

        # Refine policy with live signals from the first pass.
        similarity_scores = [c.get("score", 0.0) for c in retrieved_chunks]
        retrieval_signals = {
            "similarity_spread": (
                max(similarity_scores) - min(similarity_scores)
                if similarity_scores else 0.0
            ),
            "document_density": len(retrieved_chunks) / max(
                self.vector_db.get_stats()["metadata_count"], 1
            ),
        }
        retrieval_policy = self.adaptive_retriever.get_policy(question, retrieval_signals)

        # ── 3. Cross-encoder reranking ─────────────────────────────────
        t0 = time.time()
        reranked_chunks = self.reranker.rerank(
            question,
            retrieved_chunks,
            top_n=min(RERANK_TOP_N, len(retrieved_chunks)),
        )
        rerank_time = round(time.time() - t0, 3)

        # ── 4. BSCO — find the minimal sufficient context ──────────────
        t0 = time.time()
        selected_chunks, bsco_stats = self.optimizer.optimize(
            reranked_chunks,
            question=question,
            threshold=retrieval_policy["threshold"],
            max_chunks=retrieval_policy["max_chunks"],
            min_chunks=retrieval_policy["min_chunks"],
            query_type=retrieval_policy["query_type"],
            return_stats=True,
        )
        bsco_time = round(time.time() - t0, 3)

        # ── 5. Prompt building ─────────────────────────────────────────
        prompt = self.prompt_builder.build_prompt(
            question,
            selected_chunks,
            query_type=retrieval_policy["query_type"],
            conversation_history=self.conversation_history,
        )

        # ── 6. LLM answer generation ───────────────────────────────────
        t0 = time.time()
        answer = self.answer_generator.generate_answer(prompt)
        llm_time = round(time.time() - t0, 3)

        # ── 7. Citations (now with question for keyword matching) ──────
        citations = self.citation_generator.generate(selected_chunks, query=question)

        # ── 8. Self-verification ───────────────────────────────────────
        verification     = self.verifier.verify(answer["answer"], selected_chunks)
        verified_answer  = verification["answer"]

        # ── 9. Confidence (now with question for keyword coverage) ─────
        confidence = self.confidence_estimator.calculate(
            selected_chunks,
            citations=citations,
            verification=verification,
            question=question,
        )

        # ── 10. Evaluation metrics ─────────────────────────────────────
        total_time = round(time.time() - pipeline_start, 3)
        evaluation = self.evaluator.calculate(
            reranked_chunks,
            selected_chunks,
            bsco_stats,
            total_time,
            verified_answer,
            retrieval_time=retrieval_time,
            rerank_time=rerank_time,
            bsco_time=bsco_time,
            llm_time=llm_time,
            similarity_scores=similarity_scores,
        )

        # ── 11. Update conversation memory ─────────────────────────────
        self.conversation_history.append({
            "question": question,
            "answer":   verified_answer,
            "confidence": confidence,
        })
        self.conversation_history = self.conversation_history[-5:]

        return {
            "question":          question,
            "answer":            verified_answer,
            "model":             answer["model"],
            "citations":         citations,
            "confidence":        confidence,
            "retrieved_chunks":  len(retrieved_chunks),
            "selected_chunks":   len(selected_chunks),
            "selected_context":  selected_chunks,
            "retrieval_policy":  retrieval_policy,
            "bsco":              bsco_stats,
            "verification":      verification,
            "evaluation":        evaluation,
            "response_time":     total_time,
            "stage_times": {
                "retrieval": retrieval_time,
                "reranking": rerank_time,
                "bsco":      bsco_time,
                "llm":       llm_time,
            },
            "pdf_paths": dict(self.pdf_paths),
        }

    # ------------------------------------------------------------------
    # Streaming question answering
    # ------------------------------------------------------------------

    def ask_stream(self, question: str, max_tokens: Optional[int] = None) -> tuple[dict, any, float]:
        """
        Run all pipeline stages up to (but not including) the LLM call,
        then return the pre-computed metadata, a streaming token generator,
        and the LLM-start timestamp so the caller can time generation.

        Usage
        -----
        ::

            pre_data, stream, llm_start = pipeline.ask_stream(question)
            full_answer = ""
            for token in stream:
                full_answer += token
            result = pipeline.finalize_stream_result(full_answer, pre_data, llm_start)
        """
        pipeline_start = time.time()

        # ── 1. Adaptive retrieval policy ───────────────────────────────
        retrieval_policy = self.adaptive_retriever.get_policy(question)

        # ── 2. Hybrid search ───────────────────────────────────────────
        t0 = time.time()
        retrieved_chunks = self.search.hybrid_search(
            question, top_k=retrieval_policy["top_k"]
        )
        retrieval_time = round(time.time() - t0, 3)

        # Attach precomputed FAISS vectors so BSCO reuses the stored
        # embeddings instead of re-encoding chunk text on every iteration.
        self.vector_db.attach_embeddings(retrieved_chunks)

        similarity_scores = [c.get("score", 0.0) for c in retrieved_chunks]
        retrieval_signals = {
            "similarity_spread": (
                max(similarity_scores) - min(similarity_scores)
                if similarity_scores else 0.0
            ),
            "document_density": len(retrieved_chunks) / max(
                self.vector_db.get_stats()["metadata_count"], 1
            ),
        }
        retrieval_policy = self.adaptive_retriever.get_policy(question, retrieval_signals)

        # ── 3. Cross-encoder reranking ─────────────────────────────────
        t0 = time.time()
        reranked_chunks = self.reranker.rerank(
            question,
            retrieved_chunks,
            top_n=min(RERANK_TOP_N, len(retrieved_chunks)),
        )
        rerank_time = round(time.time() - t0, 3)

        # ── 4. BSCO ────────────────────────────────────────────────────
        t0 = time.time()
        selected_chunks, bsco_stats = self.optimizer.optimize(
            reranked_chunks,
            question=question,
            threshold=retrieval_policy["threshold"],
            max_chunks=retrieval_policy["max_chunks"],
            min_chunks=retrieval_policy["min_chunks"],
            query_type=retrieval_policy["query_type"],
            return_stats=True,
        )
        bsco_time = round(time.time() - t0, 3)

        # ── 5. Prompt building ─────────────────────────────────────────
        prompt = self.prompt_builder.build_prompt(
            question,
            selected_chunks,
            query_type=retrieval_policy["query_type"],
            conversation_history=self.conversation_history,
        )

        pre_data = {
            "question":          question,
            "retrieved_chunks":  retrieved_chunks,
            "reranked_chunks":   reranked_chunks,
            "selected_chunks":   selected_chunks,
            "bsco_stats":        bsco_stats,
            "retrieval_policy":  retrieval_policy,
            "similarity_scores": similarity_scores,
            "retrieval_time":    retrieval_time,
            "rerank_time":       rerank_time,
            "bsco_time":         bsco_time,
            "pipeline_start":    pipeline_start,
        }

        llm_start = time.time()
        stream = self.answer_generator.generate_answer_stream(prompt, max_tokens=max_tokens)
        return pre_data, stream, llm_start

    def finalize_stream_result(
        self,
        answer: str,
        pre_data: dict,
        llm_start: float,
    ) -> dict:
        """
        Complete the result dict after the streaming answer has been collected.
        Runs stages 7–11 (citations, verification, confidence, evaluation).
        """
        llm_time = round(time.time() - llm_start, 3)

        question         = pre_data["question"]
        selected_chunks  = pre_data["selected_chunks"]
        reranked_chunks  = pre_data["reranked_chunks"]
        bsco_stats       = pre_data["bsco_stats"]
        retrieval_policy = pre_data["retrieval_policy"]

        # Citations with keyword matching.
        citations = self.citation_generator.generate(selected_chunks, query=question)

        verification    = self.verifier.verify(answer, selected_chunks)
        verified_answer = verification["answer"]

        # Confidence with keyword coverage.
        confidence = self.confidence_estimator.calculate(
            selected_chunks,
            citations=citations,
            verification=verification,
            question=question,
        )

        total_time = round(time.time() - pre_data["pipeline_start"], 3)
        evaluation = self.evaluator.calculate(
            reranked_chunks,
            selected_chunks,
            bsco_stats,
            total_time,
            verified_answer,
            retrieval_time=pre_data["retrieval_time"],
            rerank_time=pre_data["rerank_time"],
            bsco_time=pre_data["bsco_time"],
            llm_time=llm_time,
            similarity_scores=pre_data["similarity_scores"],
        )

        self.conversation_history.append({
            "question": question,
            "answer":   verified_answer,
            "confidence": confidence,
        })
        self.conversation_history = self.conversation_history[-5:]

        return {
            "question":         question,
            "answer":           verified_answer,
            "model":            self.answer_generator.model,
            "citations":        citations,
            "confidence":       confidence,
            "retrieved_chunks": len(pre_data["retrieved_chunks"]),
            "selected_chunks":  len(selected_chunks),
            "selected_context": selected_chunks,
            "retrieval_policy": retrieval_policy,
            "bsco":             bsco_stats,
            "verification":     verification,
            "evaluation":       evaluation,
            "response_time":    total_time,
            "stage_times": {
                "retrieval": pre_data["retrieval_time"],
                "reranking": pre_data["rerank_time"],
                "bsco":      pre_data["bsco_time"],
                "llm":       llm_time,
            },
            "pdf_paths": dict(self.pdf_paths),
        }

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_database_stats(self) -> dict:
        """Return FAISS index statistics."""
        return self.vector_db.get_stats()

    def clear_database(self) -> None:
        """Remove the persisted FAISS index and reset conversation memory."""
        self.vector_db.clear()
        self.conversation_history = []
        self.pdf_paths = {}
        self.uploaded_chunks = []

    # ------------------------------------------------------------------
    # Research analysis methods
    # ------------------------------------------------------------------

    def analyze_research(
        self,
        question: str,
        query_type: str = "research_gap",
    ) -> dict:
        """
        Perform advanced research analysis using uploaded papers.

        Supports:
        - research_gap: Identify research gaps across multiple papers
        - literature_survey: Generate literature survey with comparison tables
        - novelty: Detect novel contributions
        - paper_similarity: Compare papers for similarity

        Parameters
        ----------
        question : str
            User's research question / command.
        query_type : str
            Type of research analysis requested.

        Returns
        -------
        dict
            Analysis results with answer, supporting evidence, and metadata.
        """
        pipeline_start = time.time()

        if not self.uploaded_chunks:
            # Load from vector DB if available
            if self.vector_db.metadata:
                self.uploaded_chunks = self.vector_db.metadata
            else:
                return {
                    "answer": "No research papers are loaded. Please upload PDFs first.",
                    "confidence": {"score": 0.0, "level": "Low"},
                    "citations": {"pages": [], "citation_text": "No papers available."},
                    "response_time": 0.0,
                }

        # Retrieve broadly across all papers
        retrieval_policy = self.adaptive_retriever.get_policy(question)
        retrieved_chunks = self.search.hybrid_search(
            question,
            top_k=min(retrieval_policy["top_k"] * 2, 20),
        )

        # Route to appropriate research module
        research_result = {}

        if query_type == "research_gap":
            research_result = self.research_gap_detector.analyze(
                chunks=retrieved_chunks,
                all_chunks=self.uploaded_chunks,
                question=question,
            )
        elif query_type == "literature_survey":
            research_result = self.paper_comparator.generate_literature_survey(
                chunks=retrieved_chunks,
                all_chunks=self.uploaded_chunks,
                question=question,
            )
        elif query_type == "novelty":
            research_result = self.novelty_detector.detect_novelty(
                chunks=retrieved_chunks,
                all_chunks=self.uploaded_chunks,
                question=question,
            )
        elif query_type == "paper_similarity":
            research_result = self.paper_comparator.compare_papers(
                chunks=retrieved_chunks,
                all_chunks=self.uploaded_chunks,
                question=question,
            )
        elif query_type == "future_work":
            research_result = self.research_gap_detector.suggest_future_work(
                chunks=retrieved_chunks,
                question=question,
            )
        else:
            # Default: research gap analysis
            research_result = self.research_gap_detector.analyze(
                chunks=retrieved_chunks,
                all_chunks=self.uploaded_chunks,
                question=question,
            )

        # Optimize context with BSCO using the research query type
        selected_chunks, _ = self.optimizer.optimize(
            retrieved_chunks,
            question=question,
            query_type=query_type,
            max_chunks=6,
            min_chunks=2,
            return_stats=True,
        )
        if not selected_chunks:
            selected_chunks = retrieved_chunks[:6]

        # Build prompt with research context and generate answer
        prompt = self.prompt_builder.build_prompt(
            query=question,
            retrieved_chunks=selected_chunks,
            query_type=query_type,
            conversation_history=self.conversation_history,
        )
        answer = self.answer_generator.generate_answer(prompt)
        verified_answer = self.verifier.verify(answer["answer"], selected_chunks)
        citations = self.citation_generator.generate(selected_chunks, query=question)
        confidence = self.confidence_estimator.calculate(
            selected_chunks,
            citations=citations,
            verification=verified_answer,
            question=question,
        )

        total_time = round(time.time() - pipeline_start, 3)

        return {
            "question": question,
            "answer": verified_answer["answer"],
            "model": answer["model"],
            "citations": citations,
            "confidence": confidence,
            "selected_context": selected_chunks,
            "research_analysis": research_result,
            "response_time": total_time,
            "pdf_paths": dict(self.pdf_paths),
        }
