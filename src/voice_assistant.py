"""
voice_assistant.py
------------------
Offline multilingual Speech-to-Text (Faster Whisper) + Text-to-Speech (Piper TTS)
for Scholar AI. Supports English and Telugu with three operating modes:

    auto  — Whisper auto-detects; falls back to forced en + te passes when the
            auto-detect language confidence is below 0.80 or the utterance is
            shorter than 5 words, selecting the best transcription by comparing
            average log probability, no-speech probability and compression ratio.
    en    — force English transcription.
    te    — force Telugu transcription.

Why not rely solely on Whisper auto-detection?
----------------------------------------------
Whisper's language tag (``info.language``) is reliable for long, clean
utterances but often wrong for short queries and code-mixed speech
(English + Telugu).  This module therefore runs a *multi-pass fallback*:
when the auto-detect pass is uncertain, it transcribes again with
``language="en"`` and ``language="te"`` and chooses the highest-quality
hypothesis.

Telugu → Hindi mis-transcription fix
-------------------------------------
Whisper frequently renders spoken Telugu as Devanagari (Hindi).  To counter
this:

* A strong Telugu ``initial_prompt`` biases the decoder towards Telugu script.
* ``detect_script_indic()`` distinguishes Telugu (U+0C00–U+0C7F) from
  Devanagari (U+0900–U+097F) so we can *detect* the failure.
* In Telugu mode, hypotheses whose output is predominantly Devanagari receive a
  large script penalty and are re-transcribed with ``condition_on_previous_text=False``
  and the Telugu prompt so the decoder is forced to restart from the Telugu bias.

Code-mixed (English + Telugu) support
-------------------------------------
Most users naturally speak a mix of English and Telugu.  The pipeline:

* preserves English technical terms (AI, ML, DBMS, algorithm, …) during
  Telugu transcription by mapping common Telugu transliterations back to the
  original English term;
* normalizes code-mixed queries before they enter the RAG retriever so that
  retrieval sees a clean query with English keywords intact.

Memory footprint
----------------
The STT model is now ``small`` at ``int8`` (≈460 MB) — a significant accuracy
upgrade over ``base`` while still comfortably fitting in 8 GB RAM alongside the
embedding model, cross-encoder reranker and the running LLM.
"""

from __future__ import annotations

import io
import logging
import os
import re
import tempfile
import urllib.request
import wave
from pathlib import Path
from typing import Any, Generator, Optional, Tuple

from config import (
    PIPER_VOICES_DIR,
    VOICE_LANGUAGE_CONFIDENCE_THRESHOLD,
    VOICE_SHORT_UTTERANCE_WORDS,
    WHISPER_COMPUTE_TYPE,
    WHISPER_MODEL_SIZE,
)

logger = logging.getLogger(__name__)

# ── Model paths ──────────────────────────────────────────────────────────────
VOICES_DIR   = Path(PIPER_VOICES_DIR)
WHISPER_SIZE = WHISPER_MODEL_SIZE      # "small" — improved multilingual accuracy
WHISPER_CT   = WHISPER_COMPUTE_TYPE    # "int8"  — CPU-friendly, fits 8 GB RAM

# Piper voice model definitions (HuggingFace releases)
PIPER_VOICES: dict[str, dict] = {
    "en": {
        "model":  "en_US-ryan-medium.onnx",
        "config": "en_US-ryan-medium.onnx.json",
        "base":   "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ryan/medium/",
    },
    "te": {
        "model":  "te_IN-coqui-high.onnx",
        "config": "te_IN-coqui-high.onnx.json",
        "base":   "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/te/te_IN/coqui/high/",
    },
}

# Short Telugu initial prompt to bias Whisper towards Telugu script.
_TELUGU_INITIAL_PROMPT = (
    "నమస్కారం, దయచేసి తెలుగులో మాట్లాడండి. "
    "కంప్యూటర్ సైన్స్, ఆర్టిఫిషియల్ ఇంటెలిజెన్స్, "
    "మెషీన్ లెర్నింగ్, డేటాబేస్, అల్గోరిథం, "
    "శాస్త్రీయ పరిశోధన, విద్యార్థులు."
)

# Unicode script ranges
_TELUGU_RE    = re.compile(r"[\u0C00-\u0C7F]")
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")

# ── Telugu ↔ English technical term map ───────────────────────────────────────
# Maps common Telugu transliterations (and English acronyms written in Telugu
# script) back to their canonical English form so academic terms survive
# retrieval intact.
TELUGU_TERM_MAP: dict[str, str] = {
    # English acronyms written in Telugu script
    "ఏఐ": "AI", "ఎమెల్": "ML", "డీఎల్": "DL", "ఎన్ఎల్పీ": "NLP",
    "డీబీఎంఎస్": "DBMS", "ఓఎస్": "OS", "సీఎన్": "CN",
    "పీఓ": "PO", "సీఓ": "CO", "పీఈఓ": "PEO",
    # Common transliterated academic words → English
    "కంప్యూటర్": "computer", "కంప్యూటరు": "computer",
    "సైన్స్": "science", "విజ్ఞానం": "science",
    "ఆర్టిఫిషియల్": "artificial", "ఇంటెలిజెన్స్": "intelligence",
    "మెషీన్": "machine", "లెర్నింగ్": "learning",
    "డేటాబేస్": "database", "డేటా": "data", "బేస్": "base",
    "అల్గోరిథం": "algorithm", "అల్గోరిథమ్": "algorithm",
    "సిస్టమ్": "system", "సిస్టం": "system",
    "ప్రోగ్రామ్": "program", "ప్రోగ్రామింగ్": "programming",
    "సాఫ్ట్‌వేర్": "software", "హార్డ్‌వేర్": "hardware",
    "నెట్‌వర్క్": "network", "ఇంటర్నెట్": "internet",
    "వెబ్": "web", "క్లౌడ్": "cloud",
    "సిలబస్": "syllabus", "సిలబస": "syllabus",
    "పరీక్ష": "exam", "విషయం": "subject", "కోర్సు": "course",
    "విద్యార్థి": "student", "ప్రొఫెసర్": "professor",
    "పరిశోధన": "research", "పరిశోధనా": "research",
    "పేపర్": "paper", "ప్రాజెక్ట్": "project",
    "మోడల్": "model", "మెథడ్": "method", "థియరీ": "theory",
    "ఫలితాలు": "results", "ఫలితం": "result",
    "ప్రశ్న": "question", "సమాధానం": "answer",
    "వివరణ": "explanation", "నిర్వచనం": "definition",
    "ఉదాహరణ": "example", "అధ్యాయం": "chapter",
}

# Telugu filler / conversational particles that carry no retrieval value.
_TELUGU_FILLERS: frozenset[str] = frozenset({
    "అంటే", "అండి", "గారు", "ఇప్పుడు", "ఇక్కడ", "అక్కడ",
    "ఏమిటి", "ఏమి", "ఎలా", "ఎందుకు", "చెప్పండి", "చెప్పు",
    "వద్దు", "కాదు", "అవును", "లేదు", "మీరు", "నాకు",
    "కావాలి", "ఉందా", "ఉంది", "చేయండి", "తెలుసా", "తెలియదు",
})

# ── Common English academic abbreviations expanded for TTS ────────────────────
_TTS_ABBREVIATIONS: dict[str, str] = {
    r"\bPOs?\b":    "Program Outcomes",
    r"\bCOs?\b":    "Course Outcomes",
    r"\bPEOs?\b":   "Program Educational Objectives",
    r"\bAI\b":      "Artificial Intelligence",
    r"\bML\b":      "Machine Learning",
    r"\bDL\b":      "Deep Learning",
    r"\bNLP\b":     "Natural Language Processing",
    r"\bDBMS\b":    "Database Management System",
    r"\bOS\b":      "Operating System",
    r"\be\.g\.\b":  "for example",
    r"\bi\.e\.\b":  "that is",
    r"\betc\.\b":   "et cetera",
    r"\bvs\.\b":    "versus",
}


# ══════════════════════════════════════════════════════════════════════════════
# Script helpers
# ══════════════════════════════════════════════════════════════════════════════

def detect_script_indic(text: str) -> str:
    """
    Classify the dominant Indic script in *text*.

    Returns one of ``"telugu"``, ``"devanagari"``, ``"latin"``, ``"other"``.
    This is the key detector for the Telugu → Hindi (Devanagari) fix.
    """
    if not text:
        return "latin"

    telugu = len(_TELUGU_RE.findall(text))
    devanagari = len(_DEVANAGARI_RE.findall(text))
    letters = len(re.findall(r"[A-Za-z]", text))
    total = telugu + devanagari + letters

    if total == 0:
        return "other"
    if telugu > 0 and telugu >= devanagari and telugu >= letters:
        return "telugu"
    if devanagari > 0 and devanagari >= telugu and devanagari > letters:
        return "devanagari"
    if letters > 0:
        return "latin"
    return "other"


def _script_fraction(text: str) -> dict[str, float]:
    """Return fraction of Telugu / Devanagari / Latin characters in *text*."""
    if not text:
        return {"telugu": 0.0, "devanagari": 0.0, "latin": 0.0}
    telugu = len(_TELUGU_RE.findall(text))
    devanagari = len(_DEVANAGARI_RE.findall(text))
    letters = len(re.findall(r"[A-Za-z]", text))
    total = telugu + devanagari + letters or 1
    return {
        "telugu": telugu / total,
        "devanagari": devanagari / total,
        "latin": letters / total,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Code-mixed query normalization
# ══════════════════════════════════════════════════════════════════════════════

def normalize_code_mixed_query(query: str) -> tuple[str, bool]:
    """
    Normalize a code-mixed (English + Telugu) voice query for retrieval.

    Returns ``(normalized_text, is_code_mixed)``.

    Steps
    -----
    1. Detect Telugu script. If none present, just clean whitespace/punctuation
       and return the original (pure English) query untouched.
    2. Replace known Telugu transliterations with their English equivalents
       (e.g. ``ఏఐ`` → ``AI``, ``కంప్యూటర్`` → ``computer``).
    3. Strip Telugu filler / conversational particles.
    4. Collapse whitespace and tidy punctuation so the RAG retriever receives
       a clean query with English academic keywords intact.
    """
    if not query or not query.strip():
        return query, False

    original = query.strip()
    if not _TELUGU_RE.search(original):
        # Pure English (or Latin) — light cleanup only, keep English terms.
        cleaned = re.sub(r"\s+", " ", original).strip()
        return cleaned, False

    text = original
    is_code_mixed = True

    # Map known Telugu terms → English equivalents.
    for te_term, en_term in TELUGU_TERM_MAP.items():
        text = text.replace(te_term, en_term)

    # Tokenise into "words" on any non-letter/digit boundary (keeps Telugu
    # phrases together by using the regex on the whole string).
    words = re.findall(r"[\u0C00-\u0C7F]+|[A-Za-z0-9]+", text)
    kept: list[str] = []
    for w in words:
        if _TELUGU_RE.search(w):
            wl = w.strip()
            if wl and wl not in _TELUGU_FILLERS:
                kept.append(wl)
        else:
            kept.append(w)
    text = " ".join(kept)

    # Collapse whitespace and tidy punctuation.
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)

    return text, is_code_mixed


# ══════════════════════════════════════════════════════════════════════════════
# VoiceAssistant
# ══════════════════════════════════════════════════════════════════════════════

class VoiceAssistant:
    """
    Lazy-initialised STT (Faster Whisper) + TTS (Piper) assistant.

    Both models are loaded on first use and cached in memory for subsequent
    requests, avoiding repeated model loading overhead on 8 GB RAM systems.
    """

    def __init__(self) -> None:
        self._whisper = None
        self._piper: dict = {}

    # ── Whisper loading ───────────────────────────────────────────────────────

    def _load_whisper(self):
        """Load and cache the Faster Whisper model (first call only)."""
        if self._whisper is None:
            from faster_whisper import WhisperModel
            logger.info("Loading Faster-Whisper '%s' (compute_type=%s) …",
                        WHISPER_SIZE, WHISPER_CT)
            self._whisper = WhisperModel(
                WHISPER_SIZE,
                device="cpu",
                compute_type=WHISPER_CT,
            )
        return self._whisper

    # ── Piper loading ─────────────────────────────────────────────────────────

    def _ensure_piper_files(self, lang: str) -> Tuple[Path, Path]:
        """Download Piper ONNX + JSON if not already cached locally."""
        info = PIPER_VOICES.get(lang, PIPER_VOICES["en"])
        VOICES_DIR.mkdir(parents=True, exist_ok=True)

        model_path  = VOICES_DIR / info["model"]
        config_path = VOICES_DIR / info["config"]

        for filename, dest in [(info["model"], model_path), (info["config"], config_path)]:
            if not dest.exists():
                url = info["base"] + filename
                logger.info("Downloading Piper voice: %s …", url)
                try:
                    urllib.request.urlretrieve(url, dest)
                except Exception as exc:
                    logger.error("Failed to download %s: %s", url, exc)
                    raise RuntimeError(
                        f"Could not download Piper voice model for '{lang}'. "
                        "Check your internet connection for first-time setup."
                    ) from exc

        return model_path, config_path

    def _load_piper(self, lang: str):
        """Load and cache a Piper voice model for *lang* (first call only)."""
        if lang not in self._piper:
            from piper.voice import PiperVoice
            model_path, config_path = self._ensure_piper_files(lang)
            logger.info("Loading Piper voice '%s' …", lang)
            self._piper[lang] = PiperVoice.load(
                str(model_path),
                config_path=str(config_path),
                use_cuda=False,
            )
        return self._piper[lang]

    # ── Low-level transcription ───────────────────────────────────────────────

    @staticmethod
    def _write_temp_audio(audio_bytes: bytes) -> str:
        """Write *audio_bytes* to a temp WAV file and return its path."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            return tmp.name

    def _transcribe_once(
        self,
        path: str,
        language: Optional[str] = None,
        *,
        condition_on_previous_text: bool = True,
    ) -> dict[str, Any]:
        """
        Run a single Faster-Whisper transcription pass.

        Parameters
        ----------
        path : str
            Path to the audio file.
        language : str, optional
            ISO-639-1 code to force (``"en"`` / ``"te"``) or None for auto.
        condition_on_previous_text : bool
            Set False for short utterances to avoid hallucinated loops.

        Returns
        -------
        dict
            ``text``, ``language`` (code or tag), ``language_probability``,
            ``avg_logprob``, ``no_speech_prob``, ``compression_ratio``.
        """
        model = self._load_whisper()
        initial_prompt = _TELUGU_INITIAL_PROMPT if language == "te" else None

        segments, info = model.transcribe(
            path,
            beam_size=5,
            language=language,                 # None → auto-detect
            task="transcribe",
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": 500,
                "min_speech_duration_ms":  200,
                "speech_pad_ms":           300,
            },
            initial_prompt=initial_prompt,
            condition_on_previous_text=condition_on_previous_text,
        )

        text = " ".join(seg.text.strip() for seg in segments).strip()

        # Resolve the detected language to a canonical ISO-639-1 code when
        # auto-detect was used.
        detected = info.language or "en"
        lang_code = detected.split("-")[0].split("_")[0].lower()
        if lang_code not in PIPER_VOICES:
            lang_code = "en"

        return {
            "text": text,
            "language": lang_code,             # canonical code (en/te/…)
            "detected_tag": detected,          # raw Whisper tag (hi/te/en/…)
            "language_probability": float(getattr(info, "language_probability", 0.0) or 0.0),
            "avg_logprob": float(getattr(info, "avg_logprob", 0.0) or 0.0),
            "no_speech_prob": float(getattr(info, "no_speech_prob", 1.0) or 1.0),
            "compression_ratio": float(getattr(info, "compression_ratio", 1.0) or 1.0),
        }

    # ── Quality comparison ────────────────────────────────────────────────────

    @staticmethod
    def _quality_score(candidate: dict[str, Any]) -> float:
        """
        Score a transcription candidate for selection.

        Higher is better.  Combines:
            avg_logprob  → higher is better (more likely tokens)
            no_speech_prob → lower is better (speech actually present)
            compression_ratio → Whisper reports ~1.0 for real speech;
                                very low values indicate repeated/hallucinated
                                text, very high values indicate silence/empty.
            devanagari penalty → heavily penalises Hindi script output when
                                 Telugu was the expected mode.
        """
        avg_logprob = candidate.get("avg_logprob", 0.0)
        no_speech = candidate.get("no_speech_prob", 1.0)
        compression = candidate.get("compression_ratio", 1.0)
        text = candidate.get("text", "")

        # Script penalty — Devanagari output for a Telugu request is wrong.
        script = detect_script_indic(text)
        if script == "devanagari":
            script_penalty = 3.0
        elif script == "telugu":
            script_penalty = -0.2   # tiny bonus for correct script
        else:
            script_penalty = 0.0

        # Compression ratio penalty: too-low (<0.5) or too-high (>2.5) are both
        # low-quality signals.  Penalise distance from ~1.2.
        compression_penalty = abs(compression - 1.2) * 0.8

        return (
            avg_logprob
            - 2.0 * no_speech
            - compression_penalty
            - script_penalty
        )

    @staticmethod
    def _select_best(candidates: list[dict[str, Any]]) -> dict[str, Any]:
        """Pick the highest-quality candidate; ties go to higher avg_logprob."""
        scored = sorted(
            candidates,
            key=lambda c: (
                VoiceAssistant._quality_score(c),
                c.get("avg_logprob", 0.0),
            ),
            reverse=True,
        )
        return scored[0]

    # ── Public API ─────────────────────────────────────────────────────────────

    def transcribe(
        self,
        audio_bytes: bytes,
        mode: str = "auto",
        *,
        progress_callback: Optional[callable] = None,
    ) -> dict[str, Any]:
        """
        Transcribe *audio_bytes* to text.

        Parameters
        ----------
        audio_bytes : bytes
            Raw audio data (WAV, WebM, OGG — anything ffmpeg handles).
        mode : str
            ``"auto"`` (default), ``"en"`` (force English) or ``"te"``
            (force Telugu).
        progress_callback : callable, optional
            Optional callable receiving ``(stage, message)`` so the UI can
            show a processing-state stepper.

        Returns
        -------
        dict
            ``text``, ``language``, ``confidence``, ``mode``, ``avg_logprob``,
            ``no_speech_prob``, ``compression_ratio``, ``used_fallback``,
            ``fallback_candidates``, ``script``, ``normalized_query``.
        """
        # Validate/normalise mode.
        mode = (mode or "auto").lower()
        if mode not in ("auto", "en", "te"):
            mode = "auto"

        # Resolve short-utterance and confidence thresholds from config.
        conf_threshold = VOICE_LANGUAGE_CONFIDENCE_THRESHOLD
        short_words = VOICE_SHORT_UTTERANCE_WORDS

        if progress_callback:
            progress_callback("transcribing", "Running speech recognition…")

        path = self._write_temp_audio(audio_bytes)
        try:
            # ── Forced English / Telugu modes ─────────────────────────────
            if mode in ("en", "te"):
                forced = self._transcribe_once(
                    path,
                    language=mode,
                    condition_on_previous_text=True,
                )

                # Telugu fix: if forced Telugu produced Devanagari output,
                # re-run once with condition_on_previous_text=False and the
                # strong Telugu prompt to break the hallucination loop.
                if mode == "te" and detect_script_indic(forced["text"]) == "devanagari":
                    logger.info("Telugu pass returned Devanagari — re-running with reset context")
                    retry = self._transcribe_once(
                        path,
                        language="te",
                        condition_on_previous_text=False,
                    )
                    if detect_script_indic(retry["text"]) != "devanagari":
                        forced = retry

                text = forced["text"]
                lang = mode
                confidence = forced["language_probability"]
                if lang not in PIPER_VOICES:
                    lang = "en"
                candidate = forced
                used_fallback = False
                candidates = [candidate]

            # ── Auto mode ─────────────────────────────────────────────────
            else:
                # Pass 1 — automatic detection.
                auto = self._transcribe_once(path, language=None)
                auto_text = auto["text"]
                word_count = len(re.findall(r"[\u0C00-\u0C7F]+|[A-Za-z0-9]+", auto_text))
                auto_conf = auto["language_probability"]
                auto_script = detect_script_indic(auto_text)

                # Ambiguity triggers:
                #  1. low language confidence
                #  2. very short utterance (< 5 words)
                #  3. Whisper guessed a non-en/te language (e.g. hi/kn/ta) —
                #     common for Telugu mis-detection
                #  4. auto pass produced Devanagari (Hindi) text — classic
                #     Telugu mis-transcription
                ambiguous = (
                    auto_conf < conf_threshold
                    or word_count < short_words
                    or auto["detected_tag"].split("-")[0].split("_")[0].lower() not in ("en", "te")
                    or auto_script == "devanagari"
                )

                if ambiguous:
                    if progress_callback:
                        progress_callback(
                            "transcribing",
                            f"Auto-detect uncertain ({auto_conf:.2f}) — comparing English & Telugu…",
                        )
                    en_candidate = self._transcribe_once(path, language="en")
                    te_candidate = self._transcribe_once(
                        path,
                        language="te",
                        condition_on_previous_text=word_count < short_words,
                    )
                    candidates = [auto, en_candidate, te_candidate]
                    best = self._select_best(candidates)

                    text = best["text"]
                    lang = best["language"]
                    confidence = best["language_probability"]
                    used_fallback = True

                    # If the chosen hypothesis is Devanagari, prefer the
                    # next-best non-Devanagari candidate (Telugu/English).
                    if detect_script_indic(text) == "devanagari":
                        non_dev = [
                            c for c in candidates
                            if detect_script_indic(c.get("text", "")) != "devanagari"
                        ]
                        if non_dev:
                            best = self._select_best(non_dev)
                            text = best["text"]
                            lang = best["language"]
                            confidence = best["language_probability"]
                else:
                    candidates = [auto]
                    best = auto
                    text = auto_text
                    lang = auto["language"]
                    confidence = auto_conf
                    used_fallback = False

                if lang not in PIPER_VOICES:
                    lang = "en"

            # ── Post-processing ───────────────────────────────────────────
            if progress_callback:
                progress_callback("processing", "Normalising transcription…")

            # Restore English technical terms (Telugu transliterations → English).
            restored = self._restore_english_terms(text)

            # Restore punctuation / spacing / academic abbreviations.
            restored = self._restore_punctuation(restored)

            # Normalize code-mixed query for retrieval.
            normalized, is_code_mixed = normalize_code_mixed_query(restored)

            if not restored.strip():
                return {
                    "text": "",
                    "language": lang,
                    "confidence": confidence,
                    "mode": mode,
                    "avg_logprob": best.get("avg_logprob", 0.0),
                    "no_speech_prob": best.get("no_speech_prob", 1.0),
                    "compression_ratio": best.get("compression_ratio", 1.0),
                    "used_fallback": used_fallback,
                    "fallback_candidates": candidates,
                    "script": detect_script_indic(restored),
                    "normalized_query": "",
                    "code_mixed": False,
                }

            logger.info(
                "Transcribed mode=%s lang=%s conf=%.3f fallback=%s text=%r",
                mode, lang, confidence, used_fallback, restored[:80],
            )

            return {
                "text": restored,
                "language": lang,
                "confidence": confidence,
                "mode": mode,
                "avg_logprob": best.get("avg_logprob", 0.0),
                "no_speech_prob": best.get("no_speech_prob", 1.0),
                "compression_ratio": best.get("compression_ratio", 1.0),
                "used_fallback": used_fallback,
                "fallback_candidates": candidates,
                "script": detect_script_indic(restored),
                "normalized_query": normalized,
                "code_mixed": is_code_mixed,
            }

        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    def transcribe_stream(
        self,
        audio_bytes: bytes,
        language_hint: Optional[str] = None,
    ) -> Generator[Tuple[str, str], None, None]:
        """
        Yield transcribed text incrementally, one segment at a time.

        Useful for long recordings where the UI should show partial results
        while Whisper is still processing later segments.

        Parameters
        ----------
        audio_bytes : bytes
            Raw audio data.
        language_hint : str, optional
            ISO-639-1 force-language code (``"en"`` / ``"te"``).

        Yields
        ------
        (segment_text, language_code)
        """
        model = self._load_whisper()

        if language_hint and language_hint not in PIPER_VOICES:
            language_hint = None

        initial_prompt = _TELUGU_INITIAL_PROMPT if language_hint == "te" else None

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            segments, info = model.transcribe(
                tmp_path,
                beam_size=5,
                language=language_hint,
                task="transcribe",
                vad_filter=True,
                vad_parameters={
                    "min_silence_duration_ms": 500,
                    "min_speech_duration_ms":  200,
                    "speech_pad_ms":           300,
                },
                initial_prompt=initial_prompt,
            )

            lang = language_hint or (info.language if info.language else "en")
            if lang not in PIPER_VOICES:
                lang = "en"

            for seg in segments:
                text = self._restore_punctuation(seg.text.strip())
                if text:
                    yield text, lang
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def synthesize(
        self,
        text: str,
        language: str = "en",
    ) -> Optional[bytes]:
        """
        Synthesize *text* to speech using Piper TTS.

        Parameters
        ----------
        text : str
            Text to speak.
        language : str
            ISO-639-1 code (``"en"`` or ``"te"``).

        Returns
        -------
        WAV bytes, or None on synthesis failure.
        """
        if not text.strip():
            return None

        # Normalise language code.
        if language not in PIPER_VOICES:
            language = "en"

        # Preprocess text for better TTS pronunciation.
        text = self._preprocess_for_tts(text, language)

        try:
            voice = self._load_piper(language)
            buf   = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                voice.synthesize(text, wf)
            return buf.getvalue()
        except Exception as exc:
            logger.error("Piper TTS synthesis failed: %s", exc)
            return None

    # ── Language detection helpers ─────────────────────────────────────────────

    @staticmethod
    def detect_script(text: str) -> str:
        """
        Heuristic: return ``"te"`` if >10% of characters are Telugu Unicode,
        else ``"en"``.

        Telugu Unicode block: U+0C00–U+0C7F.
        """
        if not text:
            return "en"
        te_count = sum(1 for c in text if "\u0C00" <= c <= "\u0C7F")
        return "te" if te_count / len(text) > 0.10 else "en"

    # ── English term restoration ───────────────────────────────────────────────

    @staticmethod
    def _restore_english_terms(text: str) -> str:
        """
        Restore English academic terms from Telugu transliterations.

        Replaces every key in :data:`TELUGU_TERM_MAP` found in *text* with its
        canonical English form, so code-mixed queries keep their technical
        keywords intact for retrieval.  Also uppercases known acronyms when
        they appear in Latin script (e.g. ``ai`` → ``AI``) inside a Telugu
        context.
        """
        if not text:
            return text

        for te_term, en_term in TELUGU_TERM_MAP.items():
            text = text.replace(te_term, en_term)

        # Uppercase well-known acronyms when surrounded by non-letters
        # (covers pure-English queries where Whisper returned lowercase).
        text = re.sub(r"(?<![A-Za-z])\b(ai|ml|dl|nlp|dbms|os|cn|po|co|peo)\b(?![A-Za-z])",
                      lambda m: m.group(1).upper(), text)

        return text

    # ── Text post-processing ──────────────────────────────────────────────────

    @staticmethod
    def _restore_punctuation(text: str) -> str:
        """
        Lightweight rule-based punctuation restoration.

        Rules applied:
        1. Collapse multiple spaces into one.
        2. Capitalise the first letter of the entire text.
        3. Capitalise the first letter after a sentence-ending punctuation.
        4. Ensure the text ends with a full stop if it does not already end
           with a punctuation mark.
        5. Remove spaces before punctuation.
        6. Handle academic abbreviations so they are not treated as sentence
           ends (e.g. ``COs.`` should not break into ``COs.`` + capital).

        This is intentionally simple — no ML model — to keep the system
        fully offline and CPU-efficient.
        """
        if not text:
            return text

        # Collapse whitespace.
        text = re.sub(r"\s+", " ", text).strip()

        # Remove space before punctuation.
        text = re.sub(r"\s([.,!?;:])", r"\1", text)

        # Capitalise after sentence-ending punctuation (but not after
        # common abbreviations like "e.g." / "i.e." / "etc.").
        _ABBREV_ENDING = re.compile(
            r"(?:e\.g|i\.e|vs|etc|al|Dr|Mr|Mrs|Prof|Fig|Eq|Sec|Ch|Vol|No|pp|ref|ex)\.?$",
            re.IGNORECASE,
        )

        def _cap_after_punct(m: re.Match) -> str:
            # If the token just before the punctuation is an abbreviation,
            # keep the following letter lowercase (sentence continues).
            if _ABBREV_ENDING.search(text[: m.start(1)]):
                return m.group(1) + " " + m.group(2)
            return m.group(1) + " " + m.group(2).upper()

        text = re.sub(r"([.!?])\s+([a-z])", _cap_after_punct, text)

        # Capitalise first character.
        if text:
            text = text[0].upper() + text[1:]

        # Ensure ends with sentence-terminating punctuation.
        if text and text[-1] not in ".!?":
            text += "."

        return text

    @staticmethod
    def _preprocess_for_tts(text: str, language: str) -> str:
        """
        Prepare *text* for TTS synthesis.

        - Expand common abbreviations for more natural pronunciation (English).
        - Remove markdown formatting (bold, italic, headers) that TTS would
          read literally.
        - Limit text to 1000 characters to keep response time reasonable.
        """
        # Strip markdown.
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)   # bold
        text = re.sub(r"\*(.+?)\*",     r"\1", text)    # italic
        text = re.sub(r"#+\s*",         "",    text)     # headers
        text = re.sub(r"`+([^`]+)`+",   r"\1", text)    # code ticks

        if language == "en":
            for pattern, replacement in _TTS_ABBREVIATIONS.items():
                text = re.sub(pattern, replacement, text)

        # Truncate for TTS latency on CPU.
        if len(text) > 1000:
            truncated = text[:1000]
            last_period = max(
                truncated.rfind("."),
                truncated.rfind("!"),
                truncated.rfind("?"),
            )
            if last_period > 600:
                text = truncated[: last_period + 1]
            else:
                text = truncated.rsplit(" ", 1)[0] + "…"

        return text.strip()

