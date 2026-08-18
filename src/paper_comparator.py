"""
Paper Comparator
----------------
Intelligent comparison of academic papers across multiple dimensions.

Features:
- Method comparison (algorithms, approaches, techniques used)
- Dataset comparison (what datasets were used, sizes, domains)
- Evaluation comparison (metrics, baselines, results)
- Contribution comparison (what each paper claims as novel)
- Limitation comparison (acknowledged weaknesses)
- Literature survey generation (theme-sorted synthesis)
- Automated comparison table generation
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set

# Patterns for identifying comparison-relevant content
METHOD_PATTERNS = [
    r"\b(method|approach|algorithm|technique|framework|architecture)\b",
    r"\b(propose|proposed|introduce|introduced|present|presented)\b",
    r"\b(using|utilizing|employing|leveraging)\b",
    r"\b(model|network|system|pipeline|schema)\b",
]

DATASET_PATTERNS = [
    r"\b(dataset|data set|corpus|benchmark|collection)\b",
    r"\b(data from|collected from|gathered from)\b",
    r"\b(samples?|instances?|examples?|records?)\b",
    r"\b(train|test|validation|evaluat)\b",
]

EVALUATION_PATTERNS = [
    r"\b(accuracy|precision|recall|f1|f-score|mrr|ndcg)\b",
    r"\b(performance|result|experiment|evaluation)\b",
    r"\b(bleu|rouge|perplexity|loss|error)\b",
    r"\b(baseline|state.?of.?the.?art|sota|ablation)\b",
    r"\b(improve|outperform|achieve|achieves|yield|yields)\b",
]

CONTRIBUTION_PATTERNS = [
    r"\b(contribution|contributions|novel|novelty|original)\b",
    r"\b(first|new|unique|state.?of.?the.?art)\b",
    r"\b(different from|unlike|compared to|superior to)\b",
]

LIMITATION_PATTERNS = [
    r"\b(limitation|drawback|weakness|shortcoming)\b",
    r"\b(limited to|not considered|does not address)\b",
    r"\b(future work|future research|needs improvement)\b",
]


class PaperComparator:
    """
    Compares academic papers across multiple dimensions using
    retrieved chunks and metadata.
    """

    def __init__(self) -> None:
        self._method_patterns = [re.compile(p, re.IGNORECASE) for p in METHOD_PATTERNS]
        self._dataset_patterns = [re.compile(p, re.IGNORECASE) for p in DATASET_PATTERNS]
        self._eval_patterns = [re.compile(p, re.IGNORECASE) for p in EVALUATION_PATTERNS]
        self._contrib_patterns = [re.compile(p, re.IGNORECASE) for p in CONTRIBUTION_PATTERNS]
        self._lim_patterns = [re.compile(p, re.IGNORECASE) for p in LIMITATION_PATTERNS]

    def compare_papers(
        self,
        chunks: List[Dict[str, Any]],
        all_chunks: Optional[List[Dict[str, Any]]] = None,
        question: str = "",
    ) -> Dict[str, Any]:
        """
        Compare papers from the provided chunks.

        Parameters
        ----------
        chunks : list[dict]
            Retrieved chunks from one or more papers.
        all_chunks : list[dict], optional
            All chunks from all uploaded papers (for deeper comparison).
        question : str
            User question for targeted comparison.

        Returns
        -------
        dict
            ``paper_count`` — number of unique source papers.
            ``methods`` — per-paper method/approach comparison.
            ``datasets`` — per-paper dataset comparison.
            ``evaluation`` — per-paper evaluation comparison.
            ``contributions`` — per-paper contribution comparison.
            ``similarities`` — shared aspects across papers.
            ``differences`` — key differences between papers.
            ``comparison_table`` — structured markdown table.
            ``survey_summary`` — literature survey style synthesis.
        """
        if not chunks:
            return self._empty_result()

        # Use all_chunks if provided and chunks are insufficient
        if all_chunks and len(self._get_paper_names(chunks)) < 2:
            chunks = all_chunks

        # Group chunks by source file
        paper_chunks: Dict[str, List[Dict]] = defaultdict(list)
        for chunk in chunks:
            source = chunk.get("source_file", "Unknown Paper")
            paper_chunks[source].append(chunk)

        papers = list(paper_chunks.keys())
        paper_data: Dict[str, Dict] = {}

        for paper, paper_chunk_list in paper_chunks.items():
            text = " ".join(c.get("text", "") for c in paper_chunk_list)
            paper_data[paper] = {
                "methods": self._extract_methods(text),
                "datasets": self._extract_datasets(text),
                "evaluation": self._extract_evaluation(text),
                "contributions": self._extract_contributions(text),
                "limitations": self._extract_limitations(text),
                "chunk_count": len(paper_chunk_list),
            }

        # Build comparison
        similarities = self._find_similarities(paper_data)
        differences = self._find_differences(paper_data)
        comparison_table = self._build_comparison_table(paper_data, papers)
        survey_summary = self._generate_survey(paper_data, papers, similarities, differences)

        return {
            "paper_count": len(papers),
            "paper_names": papers,
            "methods": {p: d["methods"] for p, d in paper_data.items()},
            "datasets": {p: d["datasets"] for p, d in paper_data.items()},
            "evaluation": {p: d["evaluation"] for p, d in paper_data.items()},
            "contributions": {p: d["contributions"] for p, d in paper_data.items()},
            "limitations": {p: d["limitations"] for p, d in paper_data.items()},
            "similarities": similarities,
            "differences": differences,
            "comparison_table": comparison_table,
            "survey_summary": survey_summary,
        }

    def _extract_by_patterns(
        self,
        text: str,
        patterns: List[re.Pattern],
        min_sentence_len: int = 10,
    ) -> List[str]:
        """Extract matching sentences from *text* using the given regex patterns.

        Shared helper behind :meth:`_extract_methods`, :meth:`_extract_datasets`,
        :meth:`_extract_evaluation`, :meth:`_extract_contributions` and
        :meth:`_extract_limitations`, which all follow the same sentence-splitting,
        pattern-matching, de-duplication and truncation logic.
        """
        sentences = re.split(r"(?<=[.!?])\s+", text)
        extracted = []
        seen: Set[str] = set()

        for sentence in sentences:
            if len(sentence) < min_sentence_len:
                continue
            normalized = sentence.lower()
            for pattern in patterns:
                if pattern.search(normalized):
                    key = sentence[:80].lower()
                    if key not in seen:
                        seen.add(key)
                        extracted.append(sentence[:200].strip())
                    break

        return extracted[:5]

    def _extract_methods(self, text: str) -> List[str]:
        """Extract method/approach descriptions."""
        return self._extract_by_patterns(text, self._method_patterns, min_sentence_len=15)

    def _extract_datasets(self, text: str) -> List[str]:
        """Extract dataset descriptions."""
        return self._extract_by_patterns(text, self._dataset_patterns)

    def _extract_evaluation(self, text: str) -> List[str]:
        """Extract evaluation results and metrics."""
        return self._extract_by_patterns(text, self._eval_patterns)

    def _extract_contributions(self, text: str) -> List[str]:
        """Extract novelty/contribution claims."""
        return self._extract_by_patterns(text, self._contrib_patterns)

    def _extract_limitations(self, text: str) -> List[str]:
        """Extract limitation statements."""
        return self._extract_by_patterns(text, self._lim_patterns)

    def _find_similarities(
        self,
        paper_data: Dict[str, Dict],
    ) -> List[str]:
        """Find common themes across papers."""
        if len(paper_data) < 2:
            return ["Only one paper available for comparison."]

        # Collect all text
        all_methods: List[str] = []
        all_datasets: List[str] = []
        all_eval: List[str] = []

        for data in paper_data.values():
            all_methods.extend(data["methods"])
            all_datasets.extend(data["datasets"])
            all_eval.extend(data["evaluation"])

        similarities = []

        # Check for shared method keywords
        method_terms = self._common_terms(all_methods)
        if method_terms:
            similarities.append(
                f"Shared method focus: {', '.join(method_terms[:3])}"
            )

        # Check for shared dataset keywords
        dataset_terms = self._common_terms(all_datasets)
        if dataset_terms:
            similarities.append(
                f"Common datasets/tasks: {', '.join(dataset_terms[:3])}"
            )

        # Check for shared evaluation metrics
        eval_terms = self._common_terms(all_eval)
        if eval_terms:
            similarities.append(
                f"Shared evaluation metrics: {', '.join(eval_terms[:3])}"
            )

        if not similarities:
            similarities.append(
                "Papers address distinct aspects with limited overlap."
            )

        return similarities

    def _find_differences(
        self,
        paper_data: Dict[str, Dict],
    ) -> List[str]:
        """Find key differences between papers."""
        if len(paper_data) < 2:
            return ["Only one paper available for comparison."]

        differences = []
        paper_names = list(paper_data.keys())

        for i in range(len(paper_names)):
            for j in range(i + 1, len(paper_names)):
                p1, p2 = paper_names[i], paper_names[j]
                d1, d2 = paper_data[p1], paper_data[p2]

                # Compare method count
                m_diff = abs(len(d1["methods"]) - len(d2["methods"]))
                if m_diff > 1:
                    differences.append(
                        f"{p1} ({len(d1['methods'])} methods) vs "
                        f"{p2} ({len(d2['methods'])} methods): "
                        f"different methodological scope."
                    )

                # Compare dataset mentions
                if d1["datasets"] and not d2["datasets"]:
                    differences.append(
                        f"{p1} uses specific datasets while {p2} does not explicitly mention datasets."
                    )

        if not differences:
            differences.append(
                "Papers share similar scope and structure."
            )

        return differences[:5]

    def _common_terms(self, sentences: List[str]) -> List[str]:
        """Extract commonly occurring technical terms."""
        term_counts: Dict[str, int] = defaultdict(int)
        total = max(len(sentences), 1)

        for sentence in sentences:
            terms = set(re.findall(r"\b[a-z][a-z-]{3,}\b", sentence.lower()))
            for term in terms:
                term_counts[term] += 1

        return sorted(
            [t for t, c in term_counts.items() if c >= max(total * 0.5, 2)],
            key=lambda t: term_counts[t],
            reverse=True,
        )[:5]

    def _build_comparison_table(
        self,
        paper_data: Dict[str, Dict],
        papers: List[str],
    ) -> str:
        """Generate a markdown comparison table."""
        if not papers:
            return "No papers to compare."

        headers = "| Aspect | " + " | ".join(
            p.replace(".pdf", "").replace("_", " ")[:30] for p in papers
        ) + " |"

        separator = "|" + "|".join("---" for _ in range(len(papers) + 1)) + "|"

        rows = []

        # Methods row
        methods_row = "| **Methods** |"
        for paper in papers:
            methods = paper_data[paper]["methods"]
            methods_row += " " + (methods[0][:100] if methods else "—") + " |"
        rows.append(methods_row)

        # Datasets row
        datasets_row = "| **Datasets** |"
        for paper in papers:
            datasets = paper_data[paper]["datasets"]
            datasets_row += " " + (datasets[0][:100] if datasets else "—") + " |"
        rows.append(datasets_row)

        # Evaluation row
        eval_row = "| **Evaluation** |"
        for paper in papers:
            evals = paper_data[paper]["evaluation"]
            eval_row += " " + (evals[0][:100] if evals else "—") + " |"
        rows.append(eval_row)

        # Contributions row
        contrib_row = "| **Contributions** |"
        for paper in papers:
            contribs = paper_data[paper]["contributions"]
            contrib_row += " " + (contribs[0][:100] if contribs else "—") + " |"
        rows.append(contrib_row)

        # Limitations row
        lim_row = "| **Limitations** |"
        for paper in papers:
            lims = paper_data[paper]["limitations"]
            lim_row += " " + (lims[0][:100] if lims else "—") + " |"
        rows.append(lim_row)

        table = "\n".join([headers, separator] + rows)
        return table

    def _generate_survey(
        self,
        paper_data: Dict[str, Dict],
        papers: List[str],
        similarities: List[str],
        differences: List[str],
    ) -> str:
        """Generate a literature survey style summary."""
        if not papers:
            return "No papers available for survey."

        parts = [
            "## Literature Survey Summary",
            "",
            f"This analysis covers **{len(papers)} paper(s)**.",
            "",
        ]

        # Paper overview
        parts.append("### Paper Overview")
        for paper in papers:
            d = paper_data[paper]
            n_methods = len(d["methods"])
            n_datasets = len(d["datasets"])
            n_eval = len(d["evaluation"])
            n_contrib = len(d["contributions"])
            parts.append(
                f"- **{paper.replace('.pdf', '')}**: "
                f"{n_contrib} contribution(s), "
                f"{n_methods} method(s), "
                f"{n_datasets} dataset(s), "
                f"{n_eval} evaluation result(s)."
            )

        parts.append("")

        # Similarities
        parts.append("### Common Themes & Similarities")
        for s in similarities:
            parts.append(f"- {s}")
        parts.append("")

        # Differences
        parts.append("### Key Differences")
        for d in differences:
            parts.append(f"- {d}")
        parts.append("")

        # Research gaps
        parts.append("### Research Gaps & Opportunities")
        parts.append(
            "- Consider combining methods across papers for hybrid approaches."
        )
        parts.append(
            "- Evaluate on shared benchmark datasets for fair comparison."
        )
        parts.append(
            "- Explore underexplored areas identified in limitations sections."
        )

        return "\n".join(parts)

    def generate_literature_survey(
        self,
        chunks: List[Dict[str, Any]],
        all_chunks: Optional[List[Dict[str, Any]]] = None,
        question: str = "",
    ) -> Dict[str, Any]:
        """
        Generate a literature survey style synthesis from chunks.

        Delegates to :meth:`compare_papers` and extracts the survey summary.
        Called by pipeline.py when query_type is ``literature_survey``.

        Parameters
        ----------
        chunks : list[dict]
            Retrieved document chunks.
        all_chunks : list[dict], optional
            All chunks from all uploaded papers.
        question : str
            Original user question.

        Returns
        -------
        dict
            ``survey_summary`` — markdown literature survey.
            ``comparison_table`` — structured comparison table.
            ``paper_count`` — number of papers analyzed.
            ``paper_names`` — list of source filenames.
        """
        comparison = self.compare_papers(
            chunks=chunks,
            all_chunks=all_chunks,
            question=question,
        )
        return {
            "survey_summary": comparison.get("survey_summary", ""),
            "comparison_table": comparison.get("comparison_table", ""),
            "paper_count": comparison.get("paper_count", 0),
            "paper_names": comparison.get("paper_names", []),
        }

    def _get_paper_names(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """Extract unique source file names from chunks."""
        return list(dict.fromkeys(
            c.get("source_file", "Unknown") for c in chunks
        ))

    def _empty_result(self) -> Dict[str, Any]:
        return {
            "paper_count": 0,
            "paper_names": [],
            "methods": {},
            "datasets": {},
            "evaluation": {},
            "contributions": {},
            "limitations": {},
            "similarities": ["No papers available for comparison."],
            "differences": ["No papers available for comparison."],
            "comparison_table": "No papers to compare.",
            "survey_summary": "No papers available for survey generation.",
        }

