"""
prompt_builder.py
-----------------
Dynamic Prompt Builder for the Adaptive RAG Academic Chatbot.

Each query type receives its own fully-specified prompt that:
  • Sets a role persona matched to the question style.
  • Gives targeted answering instructions.
  • Specifies the exact output format expected.
  • Includes the standard hallucination-prevention rule.

This version includes specialized templates for academic QA, formulas,
list extraction, literature surveys, novelty analysis, paper similarity,
research gaps, and future-work synthesis.
"""

from __future__ import annotations

from typing import Optional


class PromptBuilder:
    """
    Builds a structured, query-type-specific prompt for the LLM.

    Usage
    -----
    >>> builder = PromptBuilder()
    >>> prompt = builder.build_prompt(question, chunks, query_type="algorithm")
    """

    # ------------------------------------------------------------------
    # Per-type system role descriptions
    # ------------------------------------------------------------------
    _ROLES: dict[str, str] = {
        "definition":       "an Academic Definition Specialist who gives precise, document-grounded definitions",
        "explanation":      "an Academic Reasoning Expert who explains concepts using only document evidence",
        "comparison":       "an Academic Analyst who compares and contrasts items using structured document evidence",
        "advantages":       "an Academic Evaluation Expert who identifies and presents benefits, merits, and strengths from document evidence",
        "disadvantages":    "an Academic Evaluation Expert who identifies and presents drawbacks, limitations, and weaknesses from document evidence",
        "summary":          "an Academic Summarizer who distills key ideas from uploaded document chunks",
        "algorithm":        "an Algorithm Explainer who describes algorithmic logic, steps, and complexity from documents",
        "procedure":        "a Technical Procedure Guide who presents ordered, document-supported steps",
        "code":             "a Code Documentation Expert who explains programs, functions, and syntax using only the uploaded document",
        "table":            "an Academic Data Organizer who presents document information in structured, tabular form",
        "architecture":     "a System Architecture Analyst who explains designs, components, and diagrams using document evidence",
        "methodology":      "a Research Methodology Expert who describes methods, approaches, and techniques from the document",
        "research_gap":     "a Research Gap Analyst who identifies limitations, missing aspects, and open problems from academic documents",
        "literature_survey":"a Literature Survey Synthesizer who compares themes, methods, datasets, findings, and gaps across papers",
        "novelty":          "a Research Novelty Analyst who identifies original contributions and how they differ from prior work",
        "paper_similarity": "a Cross-Paper Similarity Analyst who finds overlap, divergence, and shared research themes",
        "future_work":      "a Future Scope Analyst who presents future directions, open problems, and improvement areas from the document",
        "formula":          "a Mathematical Documentation Expert who extracts equations and variable definitions from documents",
        "list_extraction":  "an Academic Information Extractor who lists requested items completely and concisely",
        "numerical":        "a Quantitative Analyst who extracts and explains numerical data from documents",
        "research_question":"a Research Paper Analyst who identifies findings, contributions, and conclusions from academic documents",
        "factual":          "an Academic Fact Retriever who answers precisely using exact document evidence",
        "open":             "an Academic Document Assistant who answers using only retrieved document context",
    }

    # ------------------------------------------------------------------
    # Per-type answering instructions
    # ------------------------------------------------------------------
    _INSTRUCTIONS: dict[str, str] = {
        "definition": (
            "1. Provide a clear, precise definition as stated or implied by the document.\n"
            "2. Follow with key attributes or properties if the document mentions them.\n"
            "3. Do NOT include external knowledge — only what the document says.\n"
            "4. Cite the supporting page number(s)."
        ),
        "explanation": (
            "1. Explain the concept or relationship using only the retrieved context.\n"
            "2. Structure your explanation logically (cause → effect, premise → conclusion).\n"
            "3. Quote or paraphrase specific passages and cite page numbers.\n"
            "4. Do not speculate beyond the document evidence."
        ),
        "comparison": (
            "1. Identify the items being compared from the document.\n"
            "2. Create a structured comparison: list similarities, then differences.\n"
            "3. Use a table or bullet points for clarity when appropriate.\n"
            "4. Cite the page number(s) for each compared point.\n"
            "5. Conclude with a document-supported summary of the key distinction."
        ),
        "advantages": (
            "1. List all advantages, benefits, or merits mentioned in the document.\n"
            "2. Use a numbered or bulleted list for clarity.\n"
            "3. Provide a brief explanation for each advantage using document evidence.\n"
            "4. Do not invent advantages not mentioned in the document.\n"
            "5. Cite the page number(s) for each point."
        ),
        "disadvantages": (
            "1. List all disadvantages, drawbacks, or limitations mentioned in the document.\n"
            "2. Use a numbered or bulleted list for clarity.\n"
            "3. Provide a brief explanation for each disadvantage using document evidence.\n"
            "4. Do not invent limitations not mentioned in the document.\n"
            "5. Cite the page number(s) for each point."
        ),
        "summary": (
            "1. Identify the main topic and scope from the document chunks.\n"
            "2. Present the most important ideas in order of significance.\n"
            "3. Use concise, academic language — avoid filler sentences.\n"
            "4. Do not add information beyond the provided chunks.\n"
            "5. Mention the pages that contain the summarized content."
        ),
        "algorithm": (
            "1. Name the algorithm and state its purpose as described in the document.\n"
            "2. Describe each step or phase in numbered order.\n"
            "3. If the document mentions time or space complexity, include it.\n"
            "4. If pseudocode or a diagram is described, summarize it accurately.\n"
            "5. Cite the page number(s) for every detail."
        ),
        "procedure": (
            "1. Present the procedure as a numbered, ordered list of steps.\n"
            "2. Use only steps that are explicitly described or implied by the document.\n"
            "3. Include any pre-conditions, tools, or inputs mentioned.\n"
            "4. Conclude with the expected output or outcome if stated.\n"
            "5. Cite supporting page numbers after each major step."
        ),
        "code": (
            "1. Present the code, function, or implementation as described in the document.\n"
            "2. Explain what each major section or function does using the document's own language.\n"
            "3. Format code blocks clearly using Markdown triple backticks.\n"
            "4. Do not write code that is not present in the document.\n"
            "5. Explain the syntax, parameters, or logic if the document does.\n"
            "6. Cite the page number(s) where the code appears."
        ),
        "table": (
            "1. Organize the requested information into a clearly formatted Markdown table.\n"
            "2. Use only data explicitly stated in the document.\n"
            "3. Include appropriate column headers.\n"
            "4. If the document does not contain enough tabular data, provide a structured bullet list instead.\n"
            "5. Cite the page number(s) for each data point."
        ),
        "architecture": (
            "1. Describe the system architecture or design as presented in the document.\n"
            "2. List all major components, modules, or layers and their roles.\n"
            "3. Explain how components interact or communicate if the document describes it.\n"
            "4. If a diagram is referenced in the document, describe its structure in text.\n"
            "5. Cite the page number(s) for each architectural detail."
        ),
        "methodology": (
            "1. Identify the methodology, method, or approach described in the document.\n"
            "2. Explain each step or phase of the method in logical order.\n"
            "3. State the tools, technologies, or frameworks used if mentioned.\n"
            "4. Describe why this methodology was chosen if the document explains it.\n"
            "5. Cite the page number(s) for every methodological detail."
        ),
        "research_gap": (
            "1. Identify all research gaps, limitations, or unresolved problems stated in the document.\n"
            "2. Explain why these gaps exist based on the document's own reasoning.\n"
            "3. Do not infer gaps not mentioned in the document.\n"
            "4. If relevant, connect the gap to the paper's stated scope or boundary.\n"
            "5. Cite the page number(s) where each gap is discussed."
        ),
        "literature_survey": (
            "1. Synthesize the uploaded papers into a literature-survey style answer.\n"
            "2. Group evidence by research theme, method, dataset, evaluation metric, contribution, limitation, and future work.\n"
            "3. Compare papers only when the retrieved context identifies their source files or page evidence.\n"
            "4. Use a compact table when comparing methods, datasets, results, or limitations.\n"
            "5. Clearly separate document-supported findings from missing evidence."
        ),
        "novelty": (
            "1. Identify the novel contributions claimed or implied in the retrieved context.\n"
            "2. Explain what appears different from related methods or baselines using only document evidence.\n"
            "3. Note whether the novelty is methodological, architectural, dataset-related, evaluation-related, or application-focused.\n"
            "4. Do not invent novelty when the context does not support it.\n"
            "5. Cite page numbers for every novelty claim."
        ),
        "paper_similarity": (
            "1. Identify common themes, methods, datasets, evaluation metrics, and limitations across the papers.\n"
            "2. Identify the main differences and unique contributions for each paper when source evidence is available.\n"
            "3. Use a similarity table with columns for aspect, shared evidence, differences, and source pages.\n"
            "4. Avoid declaring two papers similar unless the retrieved context supports the overlap.\n"
            "5. Cite source files and page numbers for each comparison point."
        ),
        "future_work": (
            "1. List all future work directions, open problems, or improvement opportunities from the document.\n"
            "2. Use a numbered or bulleted list for clarity.\n"
            "3. Provide context for each item using the document's own language.\n"
            "4. Do not suggest directions not present in the document.\n"
            "5. Cite the page number(s) where future work is discussed."
        ),
        "formula": (
            "1. Extract the exact formula, equation, or mathematical expression from the context.\n"
            "2. Define every variable, symbol, unit, and assumption mentioned by the document.\n"
            "3. If the document includes derivation steps, present them in order.\n"
            "4. Do not derive or compute anything not supported by the document.\n"
            "5. Cite the page number(s) where the formula appears."
        ),
        "list_extraction": (
            "1. Extract every requested item found in the retrieved context.\n"
            "2. Preserve document terminology and ordering when possible.\n"
            "3. Add a short explanation for each item only when the document provides one.\n"
            "4. If the context is incomplete, state which items may be missing.\n"
            "5. Cite page numbers for the listed items."
        ),
        "numerical": (
            "1. Extract the exact numbers, measures, or statistics from the document.\n"
            "2. State the units and context for each value.\n"
            "3. If a calculation is described, show the formula or method from the document.\n"
            "4. Do not compute values not present in the document.\n"
            "5. Cite the page number(s) where each figure appears."
        ),
        "research_question": (
            "1. Identify the research problem or objective from the document.\n"
            "2. State the methodology or approach used (as described).\n"
            "3. Present the key findings, results, or contributions.\n"
            "4. Mention limitations or future work if the document discusses them.\n"
            "5. Cite specific page numbers for each point."
        ),
        "factual": (
            "1. Answer directly and concisely using exact document evidence.\n"
            "2. Quote or closely paraphrase the relevant passage.\n"
            "3. Do not add background information not in the document.\n"
            "4. Cite the supporting page number(s)."
        ),
        "open": (
            "1. Answer in a concise academic style using only the retrieved context.\n"
            "2. Combine information from multiple chunks when relevant.\n"
            "3. Cite supporting page numbers.\n"
            "4. If the document does not contain enough information, say so explicitly."
        ),
    }

    # ------------------------------------------------------------------
    # Per-type output format specifications
    # ------------------------------------------------------------------
    _OUTPUT_FORMATS: dict[str, str] = {
        "definition": (
            "Answer:\n<Precise definition and key attributes>\n\n"
            "Supporting Pages:\n<Page numbers>\n\nConfidence: High / Medium / Low"
        ),
        "explanation": (
            "Answer:\n<Step-by-step explanation grounded in document evidence>\n\n"
            "Supporting Pages:\n<Page numbers>\n\nConfidence: High / Medium / Low"
        ),
        "comparison": (
            "Answer:\n<Similarities: ...>\n<Differences: ...>\n<Summary: ...>\n\n"
            "Supporting Pages:\n<Page numbers per point>\n\nConfidence: High / Medium / Low"
        ),
        "advantages": (
            "Answer:\nAdvantages / Benefits:\n"
            "1. <Advantage 1> — <brief document-supported explanation>\n"
            "2. <Advantage 2> — <brief document-supported explanation>\n"
            "...\n\n"
            "Supporting Pages:\n<Page numbers>\n\nConfidence: High / Medium / Low"
        ),
        "disadvantages": (
            "Answer:\nDisadvantages / Limitations:\n"
            "1. <Disadvantage 1> — <brief document-supported explanation>\n"
            "2. <Disadvantage 2> — <brief document-supported explanation>\n"
            "...\n\n"
            "Supporting Pages:\n<Page numbers>\n\nConfidence: High / Medium / Low"
        ),
        "summary": (
            "Answer:\n<Concise academic summary of key ideas>\n\n"
            "Supporting Pages:\n<Page numbers>\n\nConfidence: High / Medium / Low"
        ),
        "algorithm": (
            "Answer:\n<Algorithm name and purpose>\n"
            "<Step 1: ...>\n<Step 2: ...>\n...\n"
            "<Complexity (if mentioned): ...>\n\n"
            "Supporting Pages:\n<Page numbers>\n\nConfidence: High / Medium / Low"
        ),
        "procedure": (
            "Answer:\n<Step 1: ...>\n<Step 2: ...>\n...\n"
            "<Expected Output (if mentioned): ...>\n\n"
            "Supporting Pages:\n<Page numbers>\n\nConfidence: High / Medium / Low"
        ),
        "code": (
            "Answer:\n```\n<code from document>\n```\n\n"
            "Explanation:\n<What the code does, per the document>\n\n"
            "Supporting Pages:\n<Page numbers>\n\nConfidence: High / Medium / Low"
        ),
        "table": (
            "Answer:\n| Column A | Column B | Column C |\n"
            "|----------|----------|----------|\n"
            "| ...      | ...      | ...      |\n\n"
            "Supporting Pages:\n<Page numbers>\n\nConfidence: High / Medium / Low"
        ),
        "architecture": (
            "Answer:\nSystem Architecture:\n"
            "<Component 1>: <role and description>\n"
            "<Component 2>: <role and description>\n"
            "...\n\n"
            "Interactions:\n<How components connect or communicate>\n\n"
            "Supporting Pages:\n<Page numbers>\n\nConfidence: High / Medium / Low"
        ),
        "methodology": (
            "Answer:\nMethodology / Approach:\n"
            "<Method name and purpose>\n"
            "<Phase 1: ...>\n<Phase 2: ...>\n...\n"
            "<Tools / Technologies: ...>\n\n"
            "Supporting Pages:\n<Page numbers>\n\nConfidence: High / Medium / Low"
        ),
        "research_gap": (
            "Answer:\nResearch Gaps / Open Problems:\n"
            "1. <Gap 1> — <document-supported context>\n"
            "2. <Gap 2> — <document-supported context>\n"
            "...\n\n"
            "Supporting Pages:\n<Page numbers>\n\nConfidence: High / Medium / Low"
        ),
        "literature_survey": (
            "Answer:\nLiterature Survey Summary:\n<Theme-based synthesis>\n\n"
            "| Paper / Source | Method | Dataset | Evaluation | Contribution | Limitation |\n"
            "|----------------|--------|---------|------------|--------------|------------|\n"
            "| ...            | ...    | ...     | ...        | ...          | ...        |\n\n"
            "Research Gaps:\n<Supported gaps>\n\nSupporting Pages:\n<Page numbers>\n\nConfidence: High / Medium / Low"
        ),
        "novelty": (
            "Answer:\nNovel Contributions:\n"
            "1. <Novelty claim> - <document-supported explanation>\n"
            "...\n\n"
            "Novelty Type:\n<Method / architecture / dataset / evaluation / application>\n\n"
            "Supporting Pages:\n<Page numbers>\n\nConfidence: High / Medium / Low"
        ),
        "paper_similarity": (
            "Answer:\n| Aspect | Similarities | Differences | Supporting Pages |\n"
            "|--------|--------------|-------------|------------------|\n"
            "| ...    | ...          | ...         | ...              |\n\n"
            "Overall Similarity:\n<Document-supported summary>\n\nConfidence: High / Medium / Low"
        ),
        "future_work": (
            "Answer:\nFuture Work / Future Scope:\n"
            "1. <Direction 1> — <document-supported explanation>\n"
            "2. <Direction 2> — <document-supported explanation>\n"
            "...\n\n"
            "Supporting Pages:\n<Page numbers>\n\nConfidence: High / Medium / Low"
        ),
        "formula": (
            "Answer:\nFormula / Equation:\n<Exact expression>\n\n"
            "Variables:\n- <Symbol>: <meaning, unit if stated>\n\n"
            "Derivation / Usage:\n<Document-supported steps if present>\n\n"
            "Supporting Pages:\n<Page numbers>\n\nConfidence: High / Medium / Low"
        ),
        "list_extraction": (
            "Answer:\nRequested Items:\n"
            "1. <Item> - <brief document-supported note if available>\n"
            "...\n\n"
            "Supporting Pages:\n<Page numbers>\n\nConfidence: High / Medium / Low"
        ),
        "numerical": (
            "Answer:\n<Value / Statistic: ...> (Unit: ..., Context: ...)\n"
            "<Formula / Method (if described): ...>\n\n"
            "Supporting Pages:\n<Page numbers>\n\nConfidence: High / Medium / Low"
        ),
        "research_question": (
            "Answer:\n<Research Problem: ...>\n<Methodology: ...>\n"
            "<Key Findings: ...>\n<Limitations / Future Work (if any): ...>\n\n"
            "Supporting Pages:\n<Page numbers>\n\nConfidence: High / Medium / Low"
        ),
        "factual": (
            "Answer:\n<Direct, concise factual answer>\n\n"
            "Supporting Pages:\n<Page numbers>\n\nConfidence: High / Medium / Low"
        ),
        "open": (
            "Answer:\n<Academic answer grounded in the document>\n\n"
            "Supporting Pages:\n<Page numbers>\n\nConfidence: High / Medium / Low"
        ),
    }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_prompt(
        self,
        query: str,
        retrieved_chunks: list[dict],
        query_type: str = "open",
        conversation_history: Optional[list[dict]] = None,
    ) -> str:
        """
        Assemble a fully-specified prompt for *query* and *retrieved_chunks*.

        Parameters
        ----------
        query : str
            The user's question.
        retrieved_chunks : list[dict]
            Chunks selected by BSCO.
        query_type : str
            Query category from :class:`QueryClassifier`.
        conversation_history : list[dict], optional
            Recent prior turns (last 3 used).

        Returns
        -------
        str
            A complete prompt string ready for the LLM.
        """
        role = self._ROLES.get(query_type, self._ROLES["open"])
        instructions = self._INSTRUCTIONS.get(query_type, self._INSTRUCTIONS["open"])
        output_format = self._OUTPUT_FORMATS.get(query_type, self._OUTPUT_FORMATS["open"])

        context_block = self._build_context_block(retrieved_chunks)
        history_block = self._format_history(conversation_history or [])

        prompt = (
            f"You are {role}.\n\n"
            "CORE RULE: Answer ONLY using the document context supplied below.\n"
            "Never use your own training knowledge. Never hallucinate or guess.\n"
            "If the document does not contain enough information, reply exactly:\n\n"
            "  The uploaded academic document does not contain enough information "
            "to answer this question.\n\n"
            "====================================================\n"
            "ANSWERING INSTRUCTIONS\n"
            "====================================================\n\n"
            f"{instructions}\n\n"
            "====================================================\n"
            "CONVERSATION MEMORY  (last 3 turns)\n"
            "====================================================\n\n"
            f"{history_block}\n\n"
            "====================================================\n"
            "QUESTION\n"
            "====================================================\n\n"
            f"{query}\n\n"
            "====================================================\n"
            "DOCUMENT CONTEXT\n"
            "====================================================\n\n"
            f"{context_block}\n\n"
            "====================================================\n"
            "REQUIRED OUTPUT FORMAT\n"
            "====================================================\n\n"
            f"{output_format}\n"
        )

        return prompt

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_context_block(retrieved_chunks: list[dict]) -> str:
        """Format retrieved chunks into a numbered context block."""
        if not retrieved_chunks:
            return "No relevant document context found."

        parts: list[str] = []
        for rank, chunk in enumerate(retrieved_chunks, start=1):
            source = chunk.get("source_file", "Uploaded PDF")
            section = chunk.get("section_title") or "Unknown section"
            chunk_type = chunk.get("chunk_type", "paragraph")
            score = chunk.get("score", 0.0)
            rerank_score = chunk.get("rerank_score")
            rerank_info = f" | Rerank: {rerank_score:.4f}" if rerank_score is not None else ""

            parts.append(
                f"--- Chunk {rank} ---\n"
                f"Source : {source}\n"
                f"Page   : {chunk['page_number']}\n"
                f"Section: {section}\n"
                f"Type   : {chunk_type}\n"
                f"Score  : {score:.4f}{rerank_info}\n\n"
                f"{chunk['text']}\n"
            )

        return "\n".join(parts)

    @staticmethod
    def _format_history(conversation_history: list[dict]) -> str:
        """Summarise the last 3 conversation turns."""
        if not conversation_history:
            return "No prior conversation."

        recent = conversation_history[-3:]
        lines: list[str] = []
        for item in recent:
            q = item.get("question", "")
            a = item.get("answer", "")
            lines.append(f"Q: {q}\nA: {a[:400]}")

        return "\n\n".join(lines)
