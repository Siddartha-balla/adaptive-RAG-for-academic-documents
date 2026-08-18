"""
query_classifier.py
--------------------
Research-level query classifier for academic RAG.

Classifies user questions into 20+ research-oriented types so that
downstream components (retrieval policy, prompt template, BSCO thresholds)
can adapt accordingly.

Query types
-----------
definition       — asks what something is or means
explanation      — asks why/how something works
comparison       — asks to compare or contrast items
advantages       — asks for benefits, merits, or pros
disadvantages    — asks for drawbacks, limitations, or cons
summary          — asks for an overview or key points
algorithm        — asks about algorithmic steps, complexity, pseudocode
procedure        — asks for a workflow, protocol, or ordered process
code             — asks for code, syntax, or implementation
table            — asks for tabular data, matrix, or structured listing
list_extraction  — asks for enumerated items or bullet points
architecture     — asks about system design, components, or diagrams
methodology      — asks about research methods, approaches, or techniques
research_gap     — asks what is missing or unexplored in the research
literature_survey— asks for literature review synthesis across papers
novelty          — asks for novel or original contributions
paper_similarity — asks for overlap or similarity between papers
future_work      — asks about future scope, open problems, or next steps
formula          — asks for mathematical formulas or equations
numerical        — asks for a calculation, quantity, or measure
research_question— asks about study findings, contributions, or conclusions
factual          — asks a narrow who/where/when fact
open             — catch-all for anything else

Research-Level Improvements:
---------------------------
1. Extended query types (20+ categories)
2. Multi-pattern matching for robustness
3. Complexity scoring for adaptive retrieval
4. Follow-up detection for context awareness
5. Token counting for context budget estimation
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class QueryProfile:
    """Structured description of a user question."""

    query_type: str
    complexity: float
    token_count: int
    has_follow_up: bool


class QueryClassifier:
    """
    Rule-based classifier that keeps routing explainable and fully offline.

    Rules are ordered from most-specific to least-specific so that narrower
    patterns take precedence.  Each pattern targets vocabulary that appears
    distinctively in each query category.
    """

    # Pronouns and referential terms that signal a follow-up question.
    FOLLOW_UP_TERMS: frozenset[str] = frozenset({
        "it", "this", "that", "they", "them", "those",
        "above", "previous", "same", "aforementioned",
    })

    # (query_type, compiled_regex) checked in order — first match wins.
    _RULES: list[tuple[str, re.Pattern]] = [
        # ── Formula / mathematical expression ───────────────────────────
        (
            "formula",
            re.compile(
                r"\b(formula|equation|mathematical|derive|derivation|"
                r"expression|calculation|compute|solve for|variable|"
                r"function|integral|derivative|matrix|vector|"
                r"theorem|lemma|proof|prove)\b",
                re.IGNORECASE,
            ),
        ),
        # ── List extraction ───────────────────────────────────────────────
        (
            "list_extraction",
            re.compile(
                r"\b(list|enumerate|itemize|bullet|point|step|"
                r"what are the|name the|identify all|"
                r"give me a list|provide a list|show all|"
                r"what are the various|what are the different)\b",
                re.IGNORECASE,
            ),
        ),
        # ── Highly specific numeric / formula ─────────────────────────
        (
            "numerical",
            re.compile(
                r"\b(calculat|comput|how many|how much|what percentage|what fraction|"
                r"what proportion|quantif|measur|numer|valu|statistic|count|total|"
                r"average|mean|median|standard deviation|probability|ratio|rate)\b",
                re.IGNORECASE,
            ),
        ),
        # ── Code / implementation ──────────────────────────────────────
        (
            "code",
            re.compile(
                r"\b(code|program|function|class|implement|syntax|snippet|"
                r"write a|coding|script|variable|loop|recursion|compile|"
                r"runtime error|debug|output of|what does .* print|pseudocode)\b",
                re.IGNORECASE,
            ),
        ),
        # ── Advantages ────────────────────────────────────────────────
        (
            "advantages",
            re.compile(
                r"\b(advantage|advantages|benefit|benefits|merit|merits|"
                r"strength|strengths|pros?|why use|why is .* useful|"
                r"positive aspect|positive aspects|good about|"
                r"significance of|importance of)\b",
                re.IGNORECASE,
            ),
        ),
        # ── Disadvantages ─────────────────────────────────────────────
        (
            "disadvantages",
            re.compile(
                r"\b(disadvantage|disadvantages|drawback|drawbacks|"
                r"limitation|limitations|weakness|weaknesses|cons?|"
                r"shortcoming|shortcomings|problem with|issue with|"
                r"negative aspect|negative aspects|bad about)\b",
                re.IGNORECASE,
            ),
        ),
        # ── Table / structured data ────────────────────────────────────
        (
            "table",
            re.compile(
                r"\b(table|tabular|tabulate|row|column|matrix|"
                r"structured list|grid|chart|list all|list the|enumerate all|"
                r"give a table|create a table)\b",
                re.IGNORECASE,
            ),
        ),
        # ── Comparison ────────────────────────────────────────────────
        (
            "comparison",
            re.compile(
                r"\b(compare|contrast|difference|differences|similarities|"
                r"versus| vs |better than|worse than|trade.?off|"
                r"pros and cons|distinguish|differentiate)\b",
                re.IGNORECASE,
            ),
        ),
        # ── Algorithm ─────────────────────────────────────────────────
        (
            "algorithm",
            re.compile(
                r"\b(algorithm|time complexity|space complexity|"
                r"big.?o|data structure|sorting|searching|graph|tree|"
                r"dynamic programming|greedy|recursive|iteration|optimiz|"
                r"complexity analysis|asymptotic)\b",
                re.IGNORECASE,
            ),
        ),
        # ── Architecture / design ──────────────────────────────────────
        (
            "architecture",
            re.compile(
                r"\b(architecture|architectural|system design|component|"
                r"diagram|blueprint|module|layer|tier|subsystem|"
                r"block diagram|flowchart|flow chart|data flow|"
                r"uml|class diagram|sequence diagram|design pattern)\b",
                re.IGNORECASE,
            ),
        ),
        # ── Methodology / approach ────────────────────────────────────
        (
            "methodology",
            re.compile(
                r"\b(methodology|methodolog|approach|technique|framework|"
                r"method used|how .* works?|strategy|mechanism|scheme|"
                r"way of|manner of|mode of)\b",
                re.IGNORECASE,
            ),
        ),
        # ── Procedure / steps ─────────────────────────────────────────
        (
            "procedure",
            re.compile(
                r"\b(steps?|procedure|process|workflow|protocol|how to|"
                r"sequence of|phases?|stages?|pipeline|walk.?through|"
                r"implementation|deploy|setup|configure|install)\b",
                re.IGNORECASE,
            ),
        ),
        # ── Research gap ──────────────────────────────────────────────
        (
            "research_gap",
            re.compile(
                r"\b(research gap|gap in|missing|not covered|unexplored|"
                r"open problem|open issue|unresolved|challenge|"
                r"what is lacking|what lacks|not addressed|unsolved)\b",
                re.IGNORECASE,
            ),
        ),
        (
            "literature_survey",
            re.compile(
                r"\b(literature survey|literature review|review paper|"
                r"survey summary|survey of|related work summary|"
                r"synthesize the papers|synthesise the papers)\b",
                re.IGNORECASE,
            ),
        ),
        (
            "novelty",
            re.compile(
                r"\b(novelty|find novelty|novel contribution|"
                r"novel contributions|what is new|what's new|"
                r"unique contribution|original contribution)\b",
                re.IGNORECASE,
            ),
        ),
        (
            "paper_similarity",
            re.compile(
                r"\b(paper similarity|similarity between papers|"
                r"how similar .* papers|similar papers|overlap between papers|"
                r"common themes across papers)\b",
                re.IGNORECASE,
            ),
        ),
        # ── Future work / scope ───────────────────────────────────────
        (
            "future_work",
            re.compile(
                r"\b(future work|future scope|further research|next steps?|"
                r"open question|ongoing|scope for improvement|"
                r"what can be done|what could be|directions)\b",
                re.IGNORECASE,
            ),
        ),
        # ── Summary ───────────────────────────────────────────────────
        (
            "summary",
            re.compile(
                r"\b(summari[sz]e|summary|overview|main idea|key points?|"
                r"abstract|gist|brief|outline|highlight|takeaway|"
                r"key topics|key concepts)\b",
                re.IGNORECASE,
            ),
        ),
        # ── Research question / findings ───────────────────────────────
        (
            "research_question",
            re.compile(
                r"\b(research|hypothesis|finding|result|conclusion|contribution|"
                r"novel|paper|study|experiment|dataset|benchmark|propose|"
                r"state.?of.?the.?art|sota|ablation)\b",
                re.IGNORECASE,
            ),
        ),
        # ── Definition ────────────────────────────────────────────────
        (
            "definition",
            re.compile(
                r"\b(define|definition|what is|what are|meaning of|"
                r"stands? for|refers? to|concept of|notion of|"
                r"expand|full form|abbreviation)\b",
                re.IGNORECASE,
            ),
        ),
        # ── Explanation ───────────────────────────────────────────────
        (
            "explanation",
            re.compile(
                r"\b(why|how|explain|discuss|analy[sz]e|relationship|"
                r"impact|effect|cause|reason|justify|describe|elaborate)\b",
                re.IGNORECASE,
            ),
        ),
        # ── Factual ───────────────────────────────────────────────────
        (
            "factual",
            re.compile(
                r"\b(when|where|who|which|how many|list|name|state|"
                r"mention|identify|enumerate)\b",
                re.IGNORECASE,
            ),
        ),
    ]

    def classify(self, question: str) -> QueryProfile:
        """
        Classify *question* and return a structured :class:`QueryProfile`.

        Parameters
        ----------
        question : str
            Raw user question string.

        Returns
        -------
        QueryProfile
            Immutable dataclass containing ``query_type``, ``complexity``,
            ``token_count``, and ``has_follow_up`` flag.
        """
        normalized = question.lower().strip()
        words = re.findall(r"[a-zA-Z0-9]+", normalized)
        token_count = len(words)

        query_type = "open"
        for candidate_type, pattern in self._RULES:
            if pattern.search(normalized):
                query_type = candidate_type
                break

        complexity = self._compute_complexity(normalized, words, query_type)
        has_follow_up = bool(set(words) & self.FOLLOW_UP_TERMS)

        return QueryProfile(
            query_type=query_type,
            complexity=complexity,
            token_count=token_count,
            has_follow_up=has_follow_up,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_complexity(
        normalized: str, words: list[str], query_type: str
    ) -> float:
        """
        Heuristic complexity score in [0, 1].

        Factors:
        - Base score per query type (multi-faceted types score higher)
        - Scaled by question length (longer ≈ more complex)
        - Boost for multi-entity connectors ("and", "between", "across"…)
        - Boost for negations or conditionals ("not", "unless", "if")
        """
        _TYPE_BASE: dict[str, float] = {
            "comparison":       0.55,
            "architecture":     0.52,
            "summary":          0.50,
            "research_question":0.48,
            "research_gap":     0.48,
            "literature_survey": 0.56,
            "novelty":           0.50,
            "paper_similarity":  0.52,
            "future_work":      0.46,
            "algorithm":        0.45,
            "methodology":      0.44,
            "procedure":        0.42,
            "explanation":      0.40,
            "advantages":       0.38,
            "disadvantages":    0.38,
            "numerical":        0.38,
            "code":             0.36,
            "table":            0.35,
            "definition":       0.25,
            "factual":          0.20,
            "open":             0.30,
        }
        base = _TYPE_BASE.get(query_type, 0.30)
        length_boost = min(len(words) / 50.0, 0.30)
        multi_entity = 0.10 if re.search(
            r"\b(and|or|across|between|multiple|several|various|both)\b",
            normalized,
        ) else 0.0
        conditional = 0.07 if re.search(
            r"\b(if|unless|assuming|given that|when)\b", normalized
        ) else 0.0
        return round(min(base + length_boost + multi_entity + conditional, 1.0), 3)
