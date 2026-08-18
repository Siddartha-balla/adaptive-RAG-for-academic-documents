---
name: UI overhaul architecture
description: ChatGPT-style Scholar AI frontend — key patterns and decisions
---

## Pattern: Streaming responses
- `ask_stream(question)` in pipeline.py returns `(pre_data, token_generator, llm_start)`
- App loops generator into `st.empty()` placeholder with ` ▌` cursor
- After exhaustion, calls `finalize_stream_result(answer, pre_data, llm_start)` for citations/confidence/metrics
- **Why:** keeps all retrieval logic in backend; only LLM call streams to UI

## Pattern: Voice pipeline
- `st.audio_input()` (Streamlit ≥1.31) records audio natively; returns UploadedFile bytes
- `VoiceAssistant.transcribe(bytes)` → temp file → faster-whisper → (text, lang)
- `VoiceAssistant.synthesize(text, lang)` → piper-tts ONNX → WAV bytes
- Piper voice models auto-downloaded to `models/voices/` on first use (needs internet once)
- Supported langs: `en` (en_US-ryan-medium) and `te` (te_IN-coqui-high)
- **Why:** all offline after first model download; no cloud API needed

## Pattern: Example chip buttons
- Render as `st.button(q, key=f"ex_{i}")` — sets `st.session_state._pending_question`
- Main loop pops via `st.session_state.get()` + `del` (st.session_state has no .pop())
- **Why:** st.button does NOT support label_visibility param (only input widgets do)

## Layout decisions
- `st.chat_message("user", avatar="👤")` + `st.chat_message("assistant", avatar="🔬")`
- `st.chat_input(disabled=not database_ready)` — auto-pins to page bottom
- Result panels as `st.tabs(["📚 Citations","🔍 Context","📊 Metrics","📄 PDF Preview"])`
- Hero + chips shown only when `chat_history` is empty (home screen)
- Voice panel shown when `st.session_state.voice_mode == True`

## CSS injection
- Single `inject_css(dark: bool)` call; dark/light swap all f-string vars
- Glassmorphism: `backdrop-filter: blur(16-24px)` + semi-transparent `rgba()` backgrounds
- Animations: `@keyframes slideInUp` (messages), `wave` (waveform bars), `typing` (dots), `pulse-ring` (voice icon)
- Chat selectors: `[data-testid="stChatMessage"]`, `[data-testid="chatAvatarIcon-user"]`
- User bubbles offset right via `margin-left: 5rem` on `:has([data-testid="chatAvatarIcon-user"])`
