# Latency Optimization — Task Tracking

## Objective
Reduce answer-generation latency (>90s) on the CPU-based Adaptive RAG pipeline.

## Completed (verified in code)
- [x] `answer_generator.py` — AnswerCache LRU, explicit ollama.Client + timeout, `_chat_options` with num_thread/num_parallel/num_ctx, indentation fixed
- [x] `config.py` — MAX_CONTEXT_TOKENS=900, LLM_MAX_TOKENS=600, full OLLAMA_* knob set
- [x] `Dockerfile` — HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1 offline env vars
- [x] `pipeline.py` — ask_stream(question, max_tokens) passes max_tokens to LLM; EnhancedBSCO(embedder=...) + attach_embeddings reuse + coverage memoization
- [x] `enhanced_bsco.py` — `embedder` param, `_collect_attached_embeddings`, coverage cache

## Remaining
- [x] `app.py` — add "⚡ Fast mode" toggle + max-tokens slider in sidebar
- [x] `app.py` — pass `max_tokens` into `ask_stream` in `handle_question`
- [x] Verify imports/compile are clean — `python -m py_compile` on app.py, config.py, and all touched src modules passed with no syntax errors
