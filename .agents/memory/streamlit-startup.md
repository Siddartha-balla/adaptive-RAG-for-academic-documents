---
name: Streamlit startup blocking
description: How to prevent heavy ML model loads from blocking Streamlit's port opening
---

## Rule
Never call `get_pipeline()` or any `@st.cache_resource` function at module level.
Always wrap in a helper that is called lazily from within the render loop.

## Why
Streamlit opens port 5000 *before* running `main()`. If model loading happens
at import time or at module level, Replit's workflow manager times out waiting
for the port and kills the process before the UI appears.

## How to apply
```python
@st.cache_resource
def load_pipeline(version: str) -> RAGPipeline:
    return RAGPipeline()   # runs once; heavy

def get_pipeline() -> RAGPipeline:
    return load_pipeline(PIPELINE_CACHE)   # called only inside render functions
```
The `.streamlit/config.toml` must also set `gatherUsageStats = false` to suppress
the email prompt that otherwise blocks the process on first run.
