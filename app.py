"""
Scholar AI — Professional Adaptive RAG Academic Chatbot
=======================================================
Complete UI rewrite: ChatGPT-style chat, light glassmorphism,
streaming responses, voice assistant, and rich research panels.
"""

from __future__ import annotations

import base64
import io
import os
import time
from datetime import datetime
from html import escape
from typing import Any, Dict, List, Optional

import streamlit as st

from config import EMBEDDING_MODEL, LLM_MAX_TOKENS, OLLAMA_MODEL
from src.pipeline import RAGPipeline
from src.utils import validate_file_upload, sanitize_filename, logger

# Path to the stylesheet that holds all component rules. Theme COLORS are
# injected separately as `:root` CSS custom properties by `_theme_tokens()`.
_ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
APP_STYLESHEET = os.path.join(_ASSETS_DIR, "styles.css")

# ── Constants ─────────────────────────────────────────────────────────────────
APP_TITLE       = "Scholar AI"
UPLOAD_DIR      = "data"
PIPELINE_CACHE  = "v5-stream-voice-max"

EXAMPLE_QUESTIONS = [
    "What are the Program Outcomes?",
    "Explain the Course Objectives.",
    "What is the syllabus for AI?",
    "What are the COs of DBMS?",
    "Summarise the key topics covered.",
]

LANG_OPTIONS = {"English": "en", "Telugu": "te", "Auto-detect": "auto"}

# Voice processing-state pipeline shown in the UI stepper.
VOICE_STAGES = [
    ("listening",    "🎙 Listening"),
    ("transcribing", "✍️ Transcribing"),
    ("processing",   "🧠 Processing"),
    ("speaking",     "🔊 Speaking"),
]


def _voice_stepper_html(stage_index: int, stage_msg: str = "") -> str:
    """Render the voice processing-state stepper as inline HTML."""
    html = '<div class="voice-stepper">'
    for i, (_key, label) in enumerate(VOICE_STAGES):
        if i < stage_index:
            cls, icon = "done", "✓"
        elif i == stage_index:
            cls, icon = "active", "●"
        else:
            cls, icon = "idle", "○"
        html += f'<span class="voice-step {cls}">{icon} {label}</span>'
    html += "</div>"
    if stage_msg:
        html += f'<div class="voice-stage-msg">{stage_msg}</div>'
    return html

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"Get help": None, "Report a bug": None, "About": None},
)

# ── Lazy Loaders ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_pipeline(v: str) -> RAGPipeline:
    return RAGPipeline()

def get_pipeline() -> RAGPipeline:
    return load_pipeline(PIPELINE_CACHE)

@st.cache_resource
def load_voice_assistant():
    from src.voice_assistant import VoiceAssistant
    return VoiceAssistant()

def get_va():
    return load_voice_assistant()


# ══════════════════════════════════════════════════════════════════════════════════════════════════════
# CSS  (theme tokens + assets/styles.css)
# ══════════════════════════════════════════════════════════════════════════════════════════════════════

# The app is light-theme only. Theme values are injected as :root CSS custom
# properties (--app-*) so the whole UI stays consistent without a theme toggle.
_LIGHT_TOKENS: dict[str, str] = {
    "bg":            "linear-gradient(135deg,#eef2f7 0%,#e3eaf2 55%,#eef2f7 100%)",
    "surface":       "rgba(255,255,255,0.94)",
    "surface-2":     "rgba(246,249,252,0.95)",
    "sidebar-bg":    "rgba(255,255,255,0.98)",
    "border":        "rgba(15,23,42,0.07)",
    "border-2":      "rgba(15,23,42,0.12)",
    "text":          "#1f2937",
    "muted":         "#5b6472",
    "accent":        "#1a73e8",
    "accent-2":      "#7c3aed",
    "accent-glow":   "rgba(26,115,232,0.18)",
    "user-bubble":   "rgba(214,233,255,0.85)",
    "user-border":   "rgba(26,115,232,0.28)",
    "code-bg":       "rgba(245,247,250,0.98)",
    "input-bg":      "rgba(255,255,255,0.98)",
    "metric-bg":     "rgba(255,255,255,0.9)",
    "tab-active":    "#1a73e8",
    "scrollbar":     "rgba(15,23,42,0.18)",
    "success":       "#15803d",
    "warning":       "#b45309",
    "danger":        "#dc2626",
    "chip-bg":       "rgba(26,115,232,0.08)",
    "chip-border":   "rgba(26,115,232,0.28)",
    "chip-text":     "#1a73e8",
    "hero-text":     "#111827",
    "shadow":        "0 8px 32px rgba(15,23,42,0.10)",
    "shadow-sm":     "0 4px 16px rgba(15,23,42,0.07)",
}


def _theme_tokens() -> str:
    """Return a `:root` block of CSS custom properties for the light theme."""
    lines = [":root {"] + ["    --app-%s: %s;" % (k, v) for k, v in _LIGHT_TOKENS.items()] + ["}"]
    return chr(10).join(lines)


def inject_css() -> None:
    """Inject the light theme tokens plus the shared app stylesheet."""
    try:
        with open(APP_STYLESHEET, "r", encoding="utf-8") as f:
            stylesheet = f.read()
    except OSError:
        stylesheet = ""

    style = (
        "<style>"
        + chr(10)
        + _theme_tokens()
        + chr(10)
        + stylesheet
        + chr(10)
        + "</style>"
    )
    st.markdown(style, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# External scripts  (MathJax + Prism — injected into parent frame)
# ══════════════════════════════════════════════════════════════════════════════

def inject_scripts() -> None:
    """Inject MathJax and Prism.js into the parent document once per session."""
    if st.session_state.get("_scripts_injected"):
        return
    st.session_state._scripts_injected = True

    st.components.v1.html("""
<script>
(function() {
    var p = window.parent.document;

    // MathJax
    if (!p.getElementById('mathjax-cdn')) {
        var mj = p.createElement('script');
        mj.id  = 'mathjax-cdn';
        mj.src = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js';
        mj.async = true;
        p.head.appendChild(mj);
        window.MathJax = {tex:{inlineMath:[['$','$'],['\\\\(','\\\\)']],displayMath:[['$$','$$'],['\\\\[','\\\\]']]}};
    }

    // Prism
    if (!p.getElementById('prism-css')) {
        var lnk  = p.createElement('link');
        lnk.id   = 'prism-css';
        lnk.rel  = 'stylesheet';
        lnk.href = 'https://cdn.jsdelivr.net/npm/prismjs@1.29/themes/prism-tomorrow.min.css';
        p.head.appendChild(lnk);

        var scr  = p.createElement('script');
        scr.id   = 'prism-js';
        scr.src  = 'https://cdn.jsdelivr.net/npm/prismjs@1.29/prism.min.js';
        scr.onload = function() {
            var ac = p.createElement('script');
            ac.src = 'https://cdn.jsdelivr.net/npm/prismjs@1.29/plugins/autoloader/prism-autoloader.min.js';
            p.body.appendChild(ac);
        };
        p.body.appendChild(scr);
    }
})();
</script>
""", height=0)


# ══════════════════════════════════════════════════════════════════════════════
# Session state
# ══════════════════════════════════════════════════════════════════════════════

def init_state() -> None:
    defaults: dict[str, Any] = {
        "database_ready":     False,
        "uploaded_signature": None,
        "build_stats":        {"pages":0,"chunks":0,"embeddings":0,"vectors":0,"files":0,"indexing_time":0.0},
        "processing_logs":    [],
        "chat_history":       [],
        "last_result":        None,
        "preview_page":       None,
        "preview_source":     None,
"pdf_paths":          {},
        "voice_mode":         False,
        "voice_language":     "auto",
        "voice_response_audio": None,
        "fast_mode":          False,
        "max_tokens":         LLM_MAX_TOKENS,
        "_scripts_injected":  False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Auto-detect existing DB on first load
    if not st.session_state.database_ready:
        try:
            db = get_pipeline().get_database_stats()
            if db["vector_count"] > 0:
                st.session_state.database_ready = True
                st.session_state.build_stats = {
                    "pages":        db["page_count"],
                    "chunks":       db["metadata_count"],
                    "embeddings":   db["metadata_count"],
                    "vectors":      db["vector_count"],
                    "files":        0,
                    "indexing_time": 0.0,
                }
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def add_log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.processing_logs.append(f"[{ts}] {msg}")


def uploaded_sig(files: list) -> str:
    return "|".join(f"{f.name}:{f.size}" for f in files)


def save_files(files: list) -> list[str]:
    """Save uploaded files with security validation."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    paths = []
    
    for f in files:
        # Validate file
        is_valid, error_msg = validate_file_upload(f.name, f.size)
        if not is_valid:
            st.error(f"Invalid file '{f.name}': {error_msg}")
            logger.warning(f"File upload rejected: {f.name} - {error_msg}")
            continue
        
        # Sanitize filename
        safe_name = sanitize_filename(f.name)
        p = os.path.join(UPLOAD_DIR, safe_name)
        
        # Save file
        try:
            with open(p, "wb") as out:
                out.write(f.getbuffer())
            paths.append(p)
            logger.info(f"File saved: {safe_name}")
        except Exception as e:
            st.error(f"Failed to save '{f.name}': {e}")
            logger.error(f"Failed to save file {f.name}: {e}")
    
    return paths


def conf_class(level: str) -> str:
    l = level.lower()
    if l == "high":   return "conf-high"
    if l == "medium": return "conf-medium"
    return "conf-low"


def conf_emoji(level: str) -> str:
    l = level.lower()
    if l == "high":   return "🟢"
    if l == "medium": return "🟡"
    return "🔴"


def fmt_bytes(audio_bytes: bytes) -> str:
    return base64.b64encode(audio_bytes).decode()


def answer_txt(r: dict) -> str:
    c = r["confidence"]
    return (
        f"Scholar AI — Answer Report\n{'='*50}\n\n"
        f"Question:\n{r['question']}\n\n"
        f"Answer:\n{r['answer']}\n\n"
        f"Citations:\n{r['citations']['citation_text']}\n\n"
        f"Confidence: {c['level']} ({c['score']:.4f})\n"
        f"Explanation: {c.get('explanation','—')}\n\n"
        f"Model: {r['model']}  |  Time: {r['response_time']} s\n"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar() -> None:
    with st.sidebar:
        # ── Branding ────────────────────────────────────────────────
        st.markdown(
            '<div class="sidebar-logo">🔬</div>'
            '<div class="sidebar-brand">Scholar AI</div>'
            '<div class="sidebar-sub">Adaptive RAG Research Assistant</div>',
            unsafe_allow_html=True,
        )

        # ── Documents ───────────────────────────────────────────────
        st.markdown('<div class="sidebar-section">📄 Documents</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Upload PDFs", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed"
        )
        force = st.checkbox("Force rebuild", value=False)

        if st.button("⚡ Build Database", type="primary", use_container_width=True, disabled=not uploaded):
            sig = uploaded_sig(uploaded) if uploaded else None
            if (
                st.session_state.database_ready
                and sig == st.session_state.uploaded_signature
                and not force
            ):
                st.info("Already indexed. Enable force rebuild to re-index.")
            else:
                st.session_state.processing_logs = []
                try:
                    paths = save_files(uploaded)
                    bar   = st.progress(0)
                    stat  = st.empty()

                    def on_progress(stage, prog, msg):
                        bar.progress(prog)
                        stat.caption(msg)
                        add_log(f"{stage}: {msg}")

                    with st.spinner("Indexing…"):
                        stats = get_pipeline().build_database(paths, on_progress)

                    st.session_state.database_ready     = True
                    st.session_state.uploaded_signature = sig
                    st.session_state.build_stats        = stats
                    add_log(f"Ready — {stats['chunks']} chunks, {stats['indexing_time']}s")
                    st.success("Database ready!")
                except Exception as e:
                    st.session_state.database_ready = False
                    st.error(f"Build failed: {e}")
                    add_log(f"Error: {e}")

        # Status pill
        if st.session_state.database_ready:
            st.markdown('<span class="status-pill pill-ready">● Ready</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-pill pill-waiting">● Awaiting PDFs</span>', unsafe_allow_html=True)

# ── Performance ──────────────────────────────────────────────
        st.markdown('<div class="sidebar-section">⚡ Speed</div>', unsafe_allow_html=True)
        fast_mode = st.toggle(
            "Fast mode (shorter answers)",
            value=st.session_state.fast_mode,
            help="Cap the LLM output length so answers are produced much faster on CPU.",
        )
        st.session_state.fast_mode = fast_mode
        max_tokens = st.slider(
            "Max tokens",
            min_value=100,
            max_value=LLM_MAX_TOKENS,
            value=min(st.session_state.max_tokens, LLM_MAX_TOKENS),
            step=50,
            help="Higher = more detailed answers but slower. Fast mode caps this lower.",
        )
        if fast_mode:
            max_tokens = min(max_tokens, 300)
        st.session_state.max_tokens = max_tokens

        # ── Voice ───────────────────────────────────────────────────
        st.markdown('<div class="sidebar-section">🎤 Voice Assistant</div>', unsafe_allow_html=True)
        st.session_state.voice_mode = st.toggle(
            "Enable voice", value=st.session_state.voice_mode
        )
        if st.session_state.voice_mode:
            lang_choice = st.selectbox(
                "Language", list(LANG_OPTIONS.keys()),
                index=2, label_visibility="collapsed"
            )
            st.session_state.voice_language = LANG_OPTIONS[lang_choice]

        # ── Model info ──────────────────────────────────────────────
        st.markdown('<div class="sidebar-section">⚙️ Model Info</div>', unsafe_allow_html=True)
        st.code(f"Embed : {EMBEDDING_MODEL.split('/')[-1]}\nLLM   : {OLLAMA_MODEL}", language="text")

        # ── DB Stats ────────────────────────────────────────────────
        s = st.session_state.build_stats
        if any(s.values()):
            st.markdown('<div class="sidebar-section">📊 Database</div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            c1.markdown(f'<div class="mini-metric"><span class="mini-metric-val">{s["pages"]}</span><span class="mini-metric-lbl">Pages</span></div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="mini-metric"><span class="mini-metric-val">{s["chunks"]}</span><span class="mini-metric-lbl">Chunks</span></div>', unsafe_allow_html=True)
            c3.markdown(f'<div class="mini-metric"><span class="mini-metric-val">{s["vectors"]}</span><span class="mini-metric-lbl">Vectors</span></div>', unsafe_allow_html=True)

        # ── Logs ────────────────────────────────────────────────────
        with st.expander("📋 Logs", expanded=False):
            if st.session_state.processing_logs:
                for entry in st.session_state.processing_logs[-14:]:
                    st.caption(entry)
            else:
                st.caption("No logs yet.")

        # ── Actions ─────────────────────────────────────────────────
        st.markdown('<div class="sidebar-section">🗑️ Actions</div>', unsafe_allow_html=True)
        if st.button("Clear Database", use_container_width=True):
            get_pipeline().clear_database()
            st.session_state.database_ready     = False
            st.session_state.build_stats        = {"pages":0,"chunks":0,"embeddings":0,"vectors":0,"files":0,"indexing_time":0.0}
            st.session_state.uploaded_signature = None
            st.session_state.last_result        = None
            add_log("Database cleared")
            st.success("Cleared.")

        if st.button("Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.last_result  = None
            st.session_state.voice_response_audio = None
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# Hero / welcome screen
# ══════════════════════════════════════════════════════════════════════════════

def render_hero() -> None:
    st.markdown("""
<div class="scholar-hero">
    <div class="scholar-kicker">🔬 Academic Research Assistant</div>
    <h1 class="scholar-title">Ask your <span>Documents</span> anything</h1>
    <p class="scholar-sub">
        Upload academic PDFs, build the knowledge base, then ask questions in
        natural language — or by voice in English or Telugu.
    </p>
</div>
""", unsafe_allow_html=True)


def render_examples() -> None:
    """Balanced, responsive example chips (3 + 2) so labels never get cramped."""
    # First row: 3 chips
    c1, c2, c3 = st.columns(3)
    for col, q in zip((c1, c2, c3), EXAMPLE_QUESTIONS[:3]):
        if col.button(q, key=f"ex_0_{EXAMPLE_QUESTIONS.index(q)}", use_container_width=True):
            st.session_state._pending_question = q
    # Second row: remaining 2 chips
    c4, c5 = st.columns(2)
    for col, q in zip((c4, c5), EXAMPLE_QUESTIONS[3:]):
        if col.button(q, key=f"ex_1_{EXAMPLE_QUESTIONS.index(q)}", use_container_width=True):
            st.session_state._pending_question = q


# ══════════════════════════════════════════════════════════════════════════════
# Right-side "What can you ask?" question guide
# ══════════════════════════════════════════════════════════════════════════════

GUIDE_GROUPS: list[dict] = [
    {
        "header": "📘 Basic",
        "items": [
            "What is the main topic of this paper?",
            "How does the proposed method work?",
            "How many datasets were used in the experiments?",
        ],
    },
    {
        "header": "💡 Evaluation",
        "items": [
            "What are the advantages of this approach?",
            "What are the disadvantages or limitations?",
            "Compare the proposed method with existing baselines.",
        ],
    },
    {
        "header": "🔬 Research Analysis",
        "items": [
            "What research gaps exist in these papers?",
            "Generate a literature survey on this topic.",
            "What are the novel contributions of this work?",
            "What future work directions are suggested?",
        ],
    },
    {
        "header": "🧮 Technical",
        "items": [
            "What is the formula or equation used?",
            "List all the algorithms mentioned in the paper.",
            "Explain the system architecture.",
            "What methodology was used for evaluation?",
        ],
    },
]


def render_question_guide() -> None:
    """Render a 'What can you ask?' suggestion box on the right side."""
    db_ready = st.session_state.database_ready

    st.markdown(
        '<div class="guide-panel">'
        '<div class="guide-header">❓ What can you ask?</div>'
        '<div class="guide-sub">Click a suggested question to ask it instantly.</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    for gi, group in enumerate(GUIDE_GROUPS):
        st.markdown(
            f'<div class="guide-group">{group["header"]}</div>',
            unsafe_allow_html=True,
        )
        for ii, example in enumerate(group["items"]):
            if st.button(
                example,
                key=f"guide_{gi}_{ii}",
                use_container_width=True,
                disabled=not db_ready,
            ):
                st.session_state._pending_question = example

    if not db_ready:
        st.caption("Upload PDFs and build the database to enable suggestions.")


# ══════════════════════════════════════════════════════════════════════════════
# Voice panel
# ══════════════════════════════════════════════════════════════════════════════

_VOICE_LANG_LABEL = {"en": "🇬🇧 English", "te": "🇮🇳 Telugu"}


def _voice_conf_class(score: float) -> str:
    if score >= 0.80:
        return "conf-high"
    if score >= 0.50:
        return "conf-medium"
    return "conf-low"


def render_voice_panel() -> Optional[str]:
    """Render voice recorder. Returns transcribed text if audio was recorded."""
    mode = st.session_state.get("voice_language", "auto")
    mode_lbl = {"auto": "Auto-detect", "en": "English", "te": "Telugu"}.get(mode, "Auto-detect")

    st.markdown("""
<div class="voice-panel">
    <div class="voice-header">
        <div class="voice-icon">🎤</div>
        <div class="voice-title">Voice Input</div>
    </div>
    <div class="voice-mode-chip">Mode: <b>%s</b></div>
    <div class="waveform">
        <div class="waveform-bar"></div><div class="waveform-bar"></div>
        <div class="waveform-bar"></div><div class="waveform-bar"></div>
        <div class="waveform-bar"></div><div class="waveform-bar"></div>
        <div class="waveform-bar"></div><div class="waveform-bar"></div>
        <div class="waveform-bar"></div>
    </div>
</div>
""" % mode_lbl, unsafe_allow_html=True)

    audio = st.audio_input(
        "🎤 Click to record — speak in English, Telugu, or a mix",
        key="voice_recorder",
    )

    transcribed = None
    if audio:
        stepper = st.empty()
        stepper.markdown(_voice_stepper_html(1, "Listening for speech…"), unsafe_allow_html=True)

        # Use a transient message holder for the current stage.
        stage_msg = st.empty()
        try:
            def on_progress(stage: str, msg: str) -> None:
                stage_idx = {
                    "transcribing": 1,
                    "processing":   2,
                }.get(stage, 1)
                stepper.markdown(_voice_stepper_html(stage_idx, msg), unsafe_allow_html=True)

            # Run the multilingual multi-pass pipeline.
            result = get_va().transcribe(audio.read(), mode=mode, progress_callback=on_progress)
            text = result.get("text", "")
            lang = result.get("language", "en")
            confidence = result.get("confidence", 0.0)
            used_fallback = result.get("used_fallback", False)

            if text:
                # Use the normalized (code-mixed friendly) query for retrieval.
                transcribed = result.get("normalized_query") or text

                # Show detected language + confidence badge.
                lbl = _VOICE_LANG_LABEL.get(lang, lang.upper())
                badge_cls = _voice_conf_class(confidence)
                fallback_note = " ⚡ multi-pass" if used_fallback else ""
                stepper.markdown(
                    f'<span class="confidence-badge {badge_cls}">'
                    f'{lbl} — {confidence:.2f} confidence{fallback_note}</span>',
                    unsafe_allow_html=True,
                )
                st.info(f"**Transcribed:** {text}")

                # Preserve the detected language for TTS response.
                st.session_state.voice_language = lang
            else:
                stepper.markdown(_voice_stepper_html(0, ""), unsafe_allow_html=True)
                st.warning("No speech detected. Please try again.")
        except Exception as e:
            stepper.markdown(_voice_stepper_html(0, ""), unsafe_allow_html=True)
            st.error(f"Transcription failed: {e}")

    # Playback of last TTS answer
    if st.session_state.voice_response_audio:
        st.markdown("**🔊 Voice Answer**")
        st.audio(st.session_state.voice_response_audio, format="audio/wav")
        c1, c2 = st.columns(2)
        c1.download_button(
            "⬇ Download Audio", st.session_state.voice_response_audio,
            file_name="scholar_ai_answer.wav", mime="audio/wav", use_container_width=True
        )
        if c2.button("🗑 Clear Audio", use_container_width=True):
            st.session_state.voice_response_audio = None
            st.rerun()

    return transcribed


# ══════════════════════════════════════════════════════════════════════════════
# Process a question (streaming + optional TTS)
# ══════════════════════════════════════════════════════════════════════════════

def handle_question(question: str, from_voice: bool = False) -> None:
    """Run the RAG pipeline and display the streamed answer."""

    with st.chat_message("user", avatar="👤"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🔬"):
        if not st.session_state.database_ready:
            st.error(
                "⚠️ No database found. Upload PDFs and click **Build Database** in the sidebar."
            )
            return

        placeholder = st.empty()
        placeholder.markdown(
            '<div class="typing-indicator">'
            '<div class="typing-dot"></div>'
            '<div class="typing-dot"></div>'
            '<div class="typing-dot"></div>'
            '</div>',
            unsafe_allow_html=True,
        )

        try:
            # Run pipeline stages → stream LLM tokens
            # Pass the user-selected max_tokens (fast mode caps it lower) so
            # the Ollama call stops generating earlier => lower CPU latency.
            pre_data, stream, llm_start = get_pipeline().ask_stream(
                question,
                max_tokens=st.session_state.max_tokens,
            )

            full_response = ""
            for token in stream:
                full_response += token
                placeholder.markdown(full_response + " ▌")

            placeholder.markdown(full_response)

            # Finalise (citations, confidence, evaluation …)
            with st.spinner("Computing citations & metrics…"):
                result = get_pipeline().finalize_stream_result(
                    full_response, pre_data, llm_start
                )

            # Persist
            st.session_state.last_result = result
            st.session_state.chat_history.append(result)
            if result.get("pdf_paths"):
                st.session_state.pdf_paths = result["pdf_paths"]

            # Confidence badge
            conf  = result["confidence"]
            level = conf["level"]
            st.markdown(
                f'<span class="confidence-badge {conf_class(level)}">'
                f'{conf_emoji(level)} {level} confidence — {conf["score"]:.3f}'
                f'</span>',
                unsafe_allow_html=True,
            )

            # TTS response
            if from_voice or st.session_state.voice_mode:
                lang = st.session_state.get("voice_language", "en")
                if lang == "auto":
                    lang = "en"
                with st.spinner("Synthesising voice response…"):
                    audio_bytes = get_va().synthesize(full_response, lang)
                if audio_bytes:
                    st.session_state.voice_response_audio = audio_bytes
                    st.audio(audio_bytes, format="audio/wav")

            st.rerun()

        except RuntimeError as e:
            placeholder.error(str(e))
        except Exception as e:
            placeholder.error(f"Pipeline error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Result panels (tabs)
# ══════════════════════════════════════════════════════════════════════════════

def render_result_panels(result: dict) -> None:
    """Render Citations / Context / Metrics / PDF panels as tabs."""
    tabs = st.tabs(["📚 Citations", "🔍 Context", "📊 Metrics", "📄 PDF Preview"])

    # ── Tab 1: Citations ─────────────────────────────────────────────
    with tabs[0]:
        citations = result["citations"]
        conf      = result["confidence"]
        verif     = result.get("verification", {})

        cL, cR = st.columns([3, 2])

        with cL:
            st.markdown("#### 📖 Source Pages")
            pages = citations.get("pages", [])
            if pages:
                pcols = st.columns(min(len(pages), 5))
                for i, page in enumerate(pages):
                    src = next(
                        (c.get("source_file") for c in result.get("selected_context", [])
                         if c.get("page_number") == page),
                        None,
                    )
                    if pcols[i % 5].button(f"📄 Page {page}", key=f"cit_{page}_{i}"):
                        st.session_state.preview_page   = page
                        st.session_state.preview_source = src
                        st.rerun()
            else:
                st.info("No page citations extracted.")

            st.caption(citations.get("citation_text", ""))

        with cR:
            st.markdown("#### 🧠 Confidence")
            level = conf["level"]
            st.markdown(
                f'<div class="glass-card">'
                f'<span class="confidence-badge {conf_class(level)}">'
                f'{conf_emoji(level)} {level}</span>'
                f'<div style="font-size:2rem;font-weight:800;margin:0.4rem 0">{conf["score"]:.4f}</div>'
                f'<div class="conf-explanation">{conf.get("explanation","")}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            comps = conf.get("components", {})
            for lbl, val in comps.items():
                st.progress(min(max(float(val), 0.0), 1.0),
                            text=f"{lbl.replace('_',' ').title()}: {float(val):.3f}")

        st.divider()
        st.markdown("#### 🔎 Verification")
        vs1, vs2, vs3 = st.columns(3)
        vs1.metric("Verification Score", f"{verif.get('score', 0.0):.3f}")
        vs2.metric("Supported Statements", verif.get("supported_statements", "—"))
        vs3.metric("Unsupported", len(verif.get("unsupported_statements", [])))

        st.download_button(
            "📥 Download Answer (.txt)", data=answer_txt(result),
            file_name="scholar_ai_answer.txt", mime="text/plain"
        )

    # ── Tab 2: Context ───────────────────────────────────────────────
    with tabs[1]:
        chunks = result.get("selected_context", [])
        policy = result.get("retrieval_policy", {})

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Retrieved",  result.get("retrieved_chunks", 0))
        m2.metric("Selected",   result.get("selected_chunks", 0))
        m3.metric("Query Type", policy.get("query_type", "—").title())
        m4.metric("Top-K Used", policy.get("top_k", "—"))

        st.divider()
        for rank, chunk in enumerate(chunks, 1):
            src   = chunk.get("source_file", "PDF")
            sec   = chunk.get("section_title") or "Unknown section"
            ctype = chunk.get("chunk_type") or "paragraph"
            score = chunk.get("score", 0.0)

            st.markdown(
                f'<div class="chunk-card">'
                f'<div class="chunk-meta">'
                f'<span class="chunk-tag">Rank {rank}</span>'
                f'<span class="chunk-tag">Page {chunk["page_number"]}</span>'
                f'<span class="chunk-tag">Score {score:.4f}</span>'
                f'<span class="chunk-tag">{escape(src)}</span>'
                f'<span class="chunk-tag">{escape(sec)}</span>'
                f'<span class="chunk-tag">{ctype}</span>'
                f'</div>'
                f'<div class="chunk-text">{escape(chunk["text"])}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Tab 3: Metrics ───────────────────────────────────────────────
    with tabs[2]:
        bsco       = result.get("bsco", {})
        evaluation = result.get("evaluation", {})
        stage_t    = result.get("stage_times", {})
        total_t    = result.get("response_time", 0.0)

        # Stage timeline
        st.markdown("#### ⏱ Pipeline Timing")
        max_t = max(max(stage_t.values(), default=0.001), 0.001)
        for stage, label in [
            ("retrieval",  "🔍 Retrieval"),
            ("reranking",  "🔀 Reranking"),
            ("bsco",       "🎯 BSCO"),
            ("llm",        "🤖 LLM Gen"),
        ]:
            t = stage_t.get(stage, 0.0)
            pct = int(t / max_t * 100)
            st.markdown(
                f'<div class="stage-row">'
                f'<div class="stage-name">{label}</div>'
                f'<div class="stage-bar-wrap"><div class="stage-bar" style="width:{pct}%"></div></div>'
                f'<div class="stage-time">{t:.3f}s</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.markdown(f"**Total:** `{total_t}s`")

        st.divider()

        # Context compression visual
        st.markdown("#### 📉 Context Reduction (BSCO)")
        retrieved_n = result.get("retrieved_chunks", 1)
        selected_n  = result.get("selected_chunks", 1)
        reduction   = bsco.get("context_reduction_percent", 0.0)
        if retrieved_n > 0:
            keep_pct = (selected_n / retrieved_n) * 100
        else:
            keep_pct = 100
        st.markdown(
            f'<div class="context-bar-wrap">'
            f'<div class="context-bar-label"><span>Kept ({selected_n}/{retrieved_n} chunks)</span><span>{keep_pct:.1f}%</span></div>'
            f'<div class="context-bar-track"><div class="context-bar-fill" style="width:{keep_pct}%"></div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Token reduction visual
        pt  = evaluation.get("prompt_tokens", 0)
        rt  = evaluation.get("response_tokens", 0)
        tr  = bsco.get("token_reduction", 0)
        max_tok = max(pt + tr, 1)
        tok_pct = int((pt / max_tok) * 100)
        st.markdown(
            f'<div class="context-bar-wrap" style="margin-top:0.6rem">'
            f'<div class="context-bar-label"><span>Token Reduction ({tr} saved)</span><span>{tok_pct}% used</span></div>'
            f'<div class="context-bar-track"><div class="context-bar-fill" style="width:{tok_pct}%"></div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.divider()

        # Numeric metrics grid
        st.markdown("#### 🔢 Token Accounting")
        ta1, ta2, ta3, ta4 = st.columns(4)
        ta1.metric("Prompt Tokens",   pt)
        ta2.metric("Response Tokens", rt)
        ta3.metric("Token Reduction", tr)
        ta4.metric("Response Length", f'{evaluation.get("response_length",0)} ch')

        st.markdown("#### 🎯 Retrieval Quality Proxies")
        rq1, rq2, rq3 = st.columns(3)
        rq1.metric("MRR Proxy",       f'{evaluation.get("mrr_proxy",0.0):.4f}')
        rq2.metric("Precision Proxy", f'{evaluation.get("retrieval_precision_proxy",0.0):.4f}')
        rq3.metric("Recall Proxy",    f'{evaluation.get("retrieval_recall_proxy",0.0):.4f}')

        st.markdown("#### 📐 Similarity Scores")
        ss1, ss2, ss3 = st.columns(3)
        ss1.metric("Avg Similarity", f'{evaluation.get("avg_similarity",0.0):.4f}')
        ss2.metric("Max Similarity", f'{evaluation.get("max_similarity",0.0):.4f}')
        ss3.metric("Min Similarity", f'{evaluation.get("min_similarity",0.0):.4f}')

        st.markdown("#### 📦 BSCO Stats")
        bq1, bq2, bq3, bq4 = st.columns(4)
        bq1.metric("Dedup Removed",  bsco.get("dedup_removed", 0))
        bq2.metric("Term Coverage",  f'{bsco.get("term_coverage",0)*100:.1f}%')
        bq3.metric("Avg Score",      f'{bsco.get("avg_score",0.0):.4f}')
        bq4.metric("Threshold Used", f'{bsco.get("threshold_used",0.0):.2f}')

        with st.expander("🗂 Raw JSON", expanded=False):
            st.json({"bsco": bsco, "evaluation": evaluation,
                     "stage_times": stage_t,
                     "verification": {
                         "score": result.get("verification",{}).get("score",0.0),
                     }})

    # ── Tab 4: PDF Preview ───────────────────────────────────────────
    with tabs[3]:
        if st.session_state.preview_page is not None:
            pp  = st.session_state.preview_page
            src = st.session_state.preview_source
            pdf_paths = result.get("pdf_paths") or st.session_state.get("pdf_paths", {})

            if src and src in pdf_paths:
                with st.spinner(f"Rendering page {pp}…"):
                    img = get_pipeline().pdf_processor.extract_page_as_image(
                        pdf_paths[src], pp
                    )
                if img:
                    st.caption(f"📄 **{src}** — Page {pp}")
                    st.image(img, use_container_width=True)
                else:
                    st.warning("Could not render page image (PyMuPDF required).")
            else:
                # Fallback: text
                fallback = [
                    c for c in result.get("selected_context", [])
                    if c.get("page_number") == pp
                ]
                if fallback:
                    st.caption(f"Page {pp}{' — ' + src if src else ''} (text fallback)")
                    for c in fallback:
                        st.info(c["text"])
                else:
                    st.info("Click a **Page** button in the Citations tab to preview it here.")
        else:
            st.info("Click a **Page** button in the Citations tab to preview a PDF page here.")


# ══════════════════════════════════════════════════════════════════════════════
# Chat history display
# ══════════════════════════════════════════════════════════════════════════════

def render_chat_history() -> None:
    """Replay saved chat_history as styled chat messages."""
    for item in st.session_state.chat_history:
        with st.chat_message("user", avatar="👤"):
            st.markdown(item["question"])

        with st.chat_message("assistant", avatar="🔬"):
            st.markdown(item["answer"])
            conf  = item.get("confidence", {})
            level = conf.get("level", "")
            if level:
                st.markdown(
                    f'<span class="confidence-badge {conf_class(level)}">'
                    f'{conf_emoji(level)} {level} — {conf.get("score",0):.3f}'
                    f'</span>',
                    unsafe_allow_html=True,
                )


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    init_state()
    inject_css()
    inject_scripts()
    render_sidebar()

    # ── Full-width welcome hero (only when no history) ───────────────
    if not st.session_state.chat_history:
        render_hero()
        render_examples()

    # ── Two-column layout: chat (left) + question guide (right) ──────
    chat_col, guide_col = st.columns([3, 1.15], gap="medium")

    with chat_col:
        # ── Chat history ─────────────────────────────────────────────
        render_chat_history()

        # ── Voice panel ──────────────────────────────────────────────
        voice_question: Optional[str] = None
        if st.session_state.voice_mode:
            voice_question = render_voice_panel()

        # ── Result panels (show after at least one answer) ────────────
        if st.session_state.last_result:
            render_result_panels(st.session_state.last_result)

    with guide_col:
        render_question_guide()

    # ── Handle pending question from example chips / guide ────────────
    pending = st.session_state.get("_pending_question")
    if pending:
        del st.session_state["_pending_question"]

    # ── Text chat input (pins to bottom) ─────────────────────────────
    text_question = st.chat_input(
        "Ask a question about your documents…",
        disabled=not st.session_state.database_ready,
    )

    # ── Dispatch ──────────────────────────────────────────────────────
    q = pending or voice_question or text_question
    if q:
        handle_question(str(q).strip(), from_voice=(q is voice_question and voice_question is not None))


if __name__ == "__main__":
    main()
