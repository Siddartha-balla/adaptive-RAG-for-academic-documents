"""
query_expander.py
-----------------
Offline Academic Query Expansion for Adaptive RAG.

Generates semantically related terms for a user query using:
  1. Abbreviation expansion  — maps acronyms to their full forms
  2. Domain synonym tables   — maps academic terms to related vocabulary
  3. Stemming variants       — adds common morphological forms
  4. Bidirectional lookup    — full-form → abbreviation as well

Fully offline: no internet, no external models, no APIs.

Integration
-----------
Called by :class:`SemanticSearch` before lexical BM25 scoring so that
queries like "What are POs?" also match text containing "Program Outcomes".

Usage
-----
::

    expander = QueryExpander()
    result = expander.expand("What are Program Outcomes?")
    print(result.expanded_query)   # original + expansions joined
    print(result.expansion_terms)  # list of added terms
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class ExpansionResult:
    """Result of query expansion."""
    original_query: str
    expanded_query: str           # original + expansion terms joined
    expansion_terms: list[str]    # terms added by expansion
    abbreviations_found: list[str] = field(default_factory=list)
    synonyms_found: list[str] = field(default_factory=list)


class QueryExpander:
    """
    Rule-based offline query expander tuned for academic engineering documents.

    Covers CS, AI/ML, software engineering, database, networking, and
    general academic/research terminology.
    """

    # ------------------------------------------------------------------
    # Abbreviation → expansion
    # ------------------------------------------------------------------
    ABBREVIATIONS: dict[str, list[str]] = {
        # Academic program / curriculum
        "po":   ["program outcome", "program outcomes", "graduate attribute"],
        "pos":  ["program outcomes", "graduate attributes"],
        "co":   ["course outcome", "course outcomes"],
        "cos":  ["course outcomes"],
        "peo":  ["program educational objective", "program educational objectives"],
        "peos": ["program educational objectives"],
        "pso":  ["program specific outcome", "program specific outcomes"],
        "psos": ["program specific outcomes"],
        "obe":  ["outcome based education"],
        "nba":  ["national board of accreditation"],
        "noc":  ["no objection certificate"],
        "gpa":  ["grade point average"],
        "cgpa": ["cumulative grade point average"],

        # AI / ML / DL
        "ai":   ["artificial intelligence"],
        "ml":   ["machine learning"],
        "dl":   ["deep learning"],
        "nn":   ["neural network", "neural networks"],
        "dnn":  ["deep neural network"],
        "cnn":  ["convolutional neural network"],
        "rnn":  ["recurrent neural network"],
        "lstm": ["long short-term memory"],
        "gru":  ["gated recurrent unit"],
        "gpt":  ["generative pretrained transformer", "generative pre-trained transformer"],
        "llm":  ["large language model", "large language models"],
        "rag":  ["retrieval augmented generation"],
        "nlp":  ["natural language processing"],
        "cv":   ["computer vision"],
        "rl":   ["reinforcement learning"],
        "svm":  ["support vector machine"],
        "knn":  ["k nearest neighbours", "k nearest neighbors"],
        "pca":  ["principal component analysis"],
        "gan":  ["generative adversarial network"],
        "vae":  ["variational autoencoder"],
        "bert": ["bidirectional encoder representations from transformers"],
        "sota": ["state of the art"],

        # Databases
        "dbms": ["database management system"],
        "rdbms":["relational database management system"],
        "sql":  ["structured query language"],
        "nosql":["not only sql", "non relational database"],
        "er":   ["entity relationship"],
        "acid": ["atomicity consistency isolation durability"],
        "crud": ["create read update delete"],

        # OS / Systems
        "os":   ["operating system"],
        "cpu":  ["central processing unit", "processor"],
        "gpu":  ["graphics processing unit"],
        "ram":  ["random access memory", "memory"],
        "io":   ["input output"],
        "ipc":  ["inter process communication"],
        "pcb":  ["process control block"],

        # Networks
        "tcp":  ["transmission control protocol"],
        "ip":   ["internet protocol"],
        "udp":  ["user datagram protocol"],
        "http": ["hypertext transfer protocol"],
        "dns":  ["domain name system"],
        "osi":  ["open systems interconnection"],
        "lan":  ["local area network"],
        "wan":  ["wide area network"],

        # Software Engineering
        "se":   ["software engineering"],
        "sdlc": ["software development life cycle"],
        "uml":  ["unified modeling language"],
        "oop":  ["object oriented programming"],
        "api":  ["application programming interface"],
        "sdk":  ["software development kit"],
        "ci":   ["continuous integration"],
        "cd":   ["continuous deployment", "continuous delivery"],
        "tdd":  ["test driven development"],
        "mvc":  ["model view controller"],

        # Data Structures / Algorithms
        "ds":   ["data structure", "data structures"],
        "dsa":  ["data structures and algorithms"],
        "bfs":  ["breadth first search"],
        "dfs":  ["depth first search"],
        "dp":   ["dynamic programming"],

        # Research
        "rq":   ["research question"],
        "rv":   ["research variable"],
        "lit":  ["literature", "literature review"],
    }

    # ------------------------------------------------------------------
    # Expansion for full phrases → synonyms / related academic terms
    # ------------------------------------------------------------------
    SYNONYMS: dict[str, list[str]] = {
        # Program / learning outcomes
        "program outcome":    ["graduate attribute", "learning objective", "course objective", "competency"],
        "program outcomes":   ["graduate attributes", "learning objectives", "competencies"],
        "course outcome":     ["learning outcome", "course objective", "competency"],
        "learning objective": ["program outcome", "course outcome", "skill", "competency"],

        # General academic
        "define":       ["definition", "meaning", "concept"],
        "definition":   ["meaning", "concept", "description", "explanation"],
        "explain":      ["describe", "elaborate", "discuss", "clarify"],
        "describe":     ["explain", "detail", "outline", "elaborate"],
        "summarize":    ["overview", "summary", "key points", "main ideas"],
        "compare":      ["contrast", "difference", "similarity", "versus"],
        "advantage":    ["benefit", "merit", "strength", "pro"],
        "advantages":   ["benefits", "merits", "strengths", "pros"],
        "disadvantage": ["drawback", "limitation", "weakness", "con"],
        "disadvantages":["drawbacks", "limitations", "weaknesses", "cons"],
        "algorithm":    ["procedure", "method", "technique", "process", "steps"],
        "architecture": ["design", "structure", "framework", "blueprint", "component"],
        "methodology":  ["method", "approach", "technique", "framework"],
        "objective":    ["goal", "aim", "purpose", "target"],
        "result":       ["outcome", "finding", "conclusion", "output"],
        "feature":      ["characteristic", "property", "attribute", "aspect"],
        "features":     ["characteristics", "properties", "attributes", "aspects"],
        "application":  ["use case", "usage", "implementation", "example"],
        "applications": ["use cases", "usages", "implementations", "examples"],
        "limitation":   ["drawback", "constraint", "restriction", "shortcoming"],
        "limitations":  ["drawbacks", "constraints", "restrictions", "shortcomings"],
        "future work":  ["future scope", "open problem", "further research", "ongoing work"],
        "future scope": ["future work", "open problem", "further research"],
        "research gap": ["missing", "not covered", "open problem", "unexplored"],
        "technique":    ["method", "approach", "algorithm", "procedure"],
        "performance":  ["efficiency", "speed", "accuracy", "throughput"],
        "implementation":["development", "coding", "programming", "deployment"],
        "module":       ["component", "unit", "subsystem", "layer"],
        "syllabus":     ["curriculum", "course content", "topics", "subjects"],
        "topics":       ["subjects", "content", "syllabus", "chapters"],
        "units":        ["modules", "chapters", "sections", "topics"],
        "experiment":   ["study", "test", "evaluation", "trial"],
        "dataset":      ["data", "corpus", "benchmark", "collection"],
        "accuracy":     ["precision", "correctness", "performance"],
        "error":        ["mistake", "fault", "bug", "defect"],
        "test":         ["evaluate", "validate", "verify", "check"],
        "model":        ["architecture", "framework", "system", "network"],
    }

    # Terms to exclude from expansion (avoid noise)
    _STOPWORDS: frozenset[str] = frozenset({
        "the", "a", "an", "and", "or", "of", "to", "in", "for", "with",
        "on", "by", "is", "are", "was", "were", "what", "which", "how",
        "why", "from", "this", "that", "into", "does", "do", "be",
        "can", "may", "should", "would", "could", "has", "have", "had",
        "please", "tell", "me", "give", "provide", "show", "list",
    })

    def __init__(self) -> None:
        # Build reverse abbreviation map: full-form → abbreviation
        self._rev: dict[str, str] = {}
        for abbr, expansions in self.ABBREVIATIONS.items():
            for exp in expansions:
                self._rev[exp] = abbr.upper()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def expand(self, query: str) -> ExpansionResult:
        """
        Expand *query* with academically related terms.

        Parameters
        ----------
        query : str
            Raw user question.

        Returns
        -------
        ExpansionResult
            Contains original query, expanded query string, and lists
            of abbreviation expansions and synonyms added.
        """
        original = query.strip()
        normalized = original.lower()

        added_terms: list[str] = []
        abbr_found: list[str] = []
        syn_found: list[str] = []

        # ── 1. Abbreviation expansion ──────────────────────────────────
        words = re.findall(r"[a-zA-Z0-9]+", normalized)
        for word in words:
            if word in self._STOPWORDS:
                continue
            if word in self.ABBREVIATIONS:
                expansions = self.ABBREVIATIONS[word]
                for exp in expansions:
                    if exp not in normalized and exp not in added_terms:
                        added_terms.append(exp)
                        abbr_found.append(exp)

        # ── 2. Synonym expansion over the normalized query ────────────
        # Check multi-word phrases first (longest match wins)
        sorted_phrases = sorted(
            self.SYNONYMS.keys(), key=len, reverse=True
        )
        for phrase in sorted_phrases:
            if phrase in normalized:
                for syn in self.SYNONYMS[phrase]:
                    if syn not in normalized and syn not in added_terms:
                        added_terms.append(syn)
                        syn_found.append(syn)

        # ── 3. Single-word synonym lookup ──────────────────────────────
        for word in words:
            if word in self._STOPWORDS or len(word) < 3:
                continue
            if word in self.SYNONYMS:
                for syn in self.SYNONYMS[word]:
                    if syn not in normalized and syn not in added_terms:
                        added_terms.append(syn)
                        syn_found.append(syn)

        # ── 4. Reverse lookup: full form in query → add abbreviation ──
        for full_form, abbr in self._rev.items():
            if full_form in normalized:
                abbr_lower = abbr.lower()
                if abbr_lower not in normalized and abbr_lower not in added_terms:
                    added_terms.append(abbr_lower)
                    abbr_found.append(abbr_lower)

        # Build the expanded query: original + unique expansion terms
        unique_terms = list(dict.fromkeys(added_terms))  # preserve order, deduplicate
        expanded_query = original
        if unique_terms:
            expanded_query = original + " " + " ".join(unique_terms)

        return ExpansionResult(
            original_query=original,
            expanded_query=expanded_query,
            expansion_terms=unique_terms,
            abbreviations_found=abbr_found,
            synonyms_found=syn_found,
        )

    def get_expansion_terms(self, query: str) -> list[str]:
        """Convenience method — returns only the list of added terms."""
        return self.expand(query).expansion_terms
