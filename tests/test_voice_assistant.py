"""
Tests for the multilingual voice pipeline.

Covers the offline (no-model) logic that powers the voice assistant:
Indic script detection (Telugu vs Devanagari), English technical-term
restoration, code-mixed query normalization, punctuation/spacing
post-processing, and the multi-pass quality comparison used to choose
the best transcription candidate.

These tests deliberately avoid loading Faster-Whisper or Piper so they
run fast in CI on machines without the voice models.
"""

from __future__ import annotations

import pytest

from src.voice_assistant import (
    VoiceAssistant,
    detect_script_indic,
    normalize_code_mixed_query,
)

# ── Indic script detection ─────────────────────────────────────────────────────


class TestDetectScriptIndic:
    def test_empty_text_is_latin(self) -> None:
        assert detect_script_indic("") == "latin"

    def test_pure_english_is_latin(self) -> None:
        assert detect_script_indic("What are the Program Outcomes?") == "latin"

    def test_telugu_text_detected(self) -> None:
        assert detect_script_indic("కంప్యూటర్ సైన్స్ అంటే ఏమిటి") == "telugu"

    def test_devanagari_text_detected(self) -> None:
        assert detect_script_indic("कंप्यूटर साइंस क्या है") == "devanagari"

    def test_code_mixed_english_telugu_detected_as_telugu(self) -> None:
        # English acronym + Telugu words → Telugu dominates.
        assert detect_script_indic("ఏఐ అంటే ఏమిటి AI") == "telugu"

    def test_latin_dominant_text(self) -> None:
        assert detect_script_indic("DBMS and OS") == "latin"


# ── English term restoration ──────────────────────────────────────────────────


class TestRestoreEnglishTerms:
    @staticmethod
    def _restore(text: str) -> str:
        return VoiceAssistant._restore_english_terms(text)

    def test_telugu_acronym_restored(self) -> None:
        assert self._restore("ఏఐ అంటే ఏమిటి") == "AI అంటే ఏమిటి"

    def test_dbms_restored(self) -> None:
        assert "DBMS" in self._restore("డీబీఎంఎస్ గురించి చెప్పండి")

    def test_computer_science_restored(self) -> None:
        restored = self._restore("కంప్యూటర్ సైన్స్ విషయం")
        assert "computer" in restored and "science" in restored

    def test_lowercase_acronym_uppercased(self) -> None:
        assert self._restore("what is ai and ml") == "what is AI and ML"

    def test_english_passthrough_unchanged(self) -> None:
        assert self._restore("Explain the syllabus for the course.") == (
            "Explain the syllabus for the course."
        )


# ── Code-mixed query normalization ─────────────────────────────────────────────


class TestNormalizeCodeMixedQuery:
    def test_pure_english_is_not_code_mixed(self) -> None:
        text, is_mixed = normalize_code_mixed_query("What is artificial intelligence?")
        assert is_mixed is False
        assert "artificial" in text.lower()

    def test_telugu_terms_mapped_to_english(self) -> None:
        text, is_mixed = normalize_code_mixed_query("కంప్యూటర్ సైన్స్ అంటే ఏమిటి")
        assert is_mixed is True
        assert "computer" in text.lower()
        assert "science" in text.lower()

    def test_code_mixed_acronyms_survive(self) -> None:
        text, is_mixed = normalize_code_mixed_query("ఏఐ మరియు డీబీఎంఎస్ గురించి చెప్పండి")
        assert is_mixed is True
        assert "AI" in text.upper()
        assert "DBMS" in text.upper()

    def test_empty_query_returns_false(self) -> None:
        text, is_mixed = normalize_code_mixed_query("   ")
        assert is_mixed is False
        assert text.strip() == ""

    def test_whitespace_collapsed(self) -> None:
        text, _ = normalize_code_mixed_query("Explain   the   architecture")
        assert "  " not in text


# ── Punctuation / spacing post-processing ─────────────────────────────────────


class TestPunctuationRestoration:
    @staticmethod
    def _restore(text: str) -> str:
        return VoiceAssistant._restore_punctuation(text)

    def test_capitalises_first_letter(self) -> None:
        assert self._restore("what are the program outcomes").startswith("What")

    def test_ensures_terminal_period(self) -> None:
        assert self._restore("explain the model").endswith(".")

    def test_collapses_whitespace(self) -> None:
        assert "  " not in self._restore("multiple    spaces    here")

    def test_removes_space_before_punctuation(self) -> None:
        assert " ," not in self._restore("hello , world")

    def test_academic_abbreviation_not_broken(self) -> None:
        text = self._restore("what are the COs of DBMS")
        # "COs" should stay intact and the sentence should end with a period.
        assert "COs" in text
        assert text.endswith(".")


# ── Multi-pass quality comparison ─────────────────────────────────────────────


class TestQualitySelection:
    @staticmethod
    def _candidate(
        text: str,
        avg_logprob: float = -0.5,
        no_speech: float = 0.1,
        compression: float = 1.2,
    ) -> dict:
        return {
            "text": text,
            "avg_logprob": avg_logprob,
            "no_speech_prob": no_speech,
            "compression_ratio": compression,
        }

    def test_best_quality_selected(self) -> None:
        candidates = [
            self._candidate("low quality noisy output", avg_logprob=-3.0, no_speech=0.6),
            self._candidate("what is machine learning", avg_logprob=-0.4, no_speech=0.05),
        ]
        best = VoiceAssistant._select_best(candidates)
        assert best["text"] == "what is machine learning"

    def test_devanagari_penalised(self) -> None:
        # Even with slightly better log-prob, Devanagari output for a Telugu
        # request loses to proper Telugu text.
        devanagari = self._candidate("कंप्यूटर साइंस क्या है", avg_logprob=-0.3)
        telugu = self._candidate("కంప్యూటర్ సైన్స్ అంటే ఏమిటి", avg_logprob=-0.6)
        best = VoiceAssistant._select_best([devanagari, telugu])
        assert best["text"] == telugu["text"]

    def test_no_speech_probability_penalised(self) -> None:
        candidates = [
            self._candidate("empty silence", avg_logprob=-0.5, no_speech=0.95),
            self._candidate("real speech here", avg_logprob=-0.6, no_speech=0.05),
        ]
        best = VoiceAssistant._select_best(candidates)
        assert best["text"] == "real speech here"

    def test_compression_ratio_penalised(self) -> None:
        candidates = [
            self._candidate("hallucinated repetition", avg_logprob=-0.4, compression=0.3),
            self._candidate("normal transcription", avg_logprob=-0.5, compression=1.2),
        ]
        best = VoiceAssistant._select_best(candidates)
        assert best["text"] == "normal transcription"

