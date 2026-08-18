"""
Answer Generator
----------------
Generates grounded answers using Ollama.

Improvements in this version
-----------------------------
Retry logic
    generate_answer() retries up to 3 times on transient connection errors
    before raising the final RuntimeError.

Timeout configuration
    Uses config.LLM_MAX_TOKENS and a per-call timeout so the system does not
    hang indefinitely when the LLM is slow.

Answer caching
    Repeated prompts (identical question + settings) resolve from an in-memory
    LRU cache in microseconds instead of re-invoking the LLM. This is the
    single biggest latency win on CPU-only machines.
"""

import hashlib
import time
from collections import OrderedDict
from typing import Generator, Optional

import ollama
from ollama import ResponseError

from config import (
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    OLLAMA_CONTEXT_WINDOW,
    OLLAMA_HOST,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_MODEL,
    OLLAMA_NUM_PARALLEL,
    OLLAMA_NUM_THREADS,
    OLLAMA_TIMEOUT,
)


class AnswerCache:
    """
    Small bounded LRU cache mapping a prompt hash to (answer, model, time).

    Repeat questions (or re-renders that re-run the same prompt) resolve in
    microseconds instead of re-invoking the LLM, which is the single biggest
    latency win on CPU-only machines.
    """

    def __init__(self, max_size: int = 64) -> None:
        self._cache: OrderedDict[str, dict] = OrderedDict()
        self._max_size = max_size

    def _key(self, prompt: str, max_tokens: int) -> str:
        raw = f"{max_tokens}|{prompt}"
        return hashlib.md5(raw.encode("utf-8", errors="ignore")).hexdigest()

    def get(self, prompt: str, max_tokens: int) -> Optional[dict]:
        key = self._key(prompt, max_tokens)
        entry = self._cache.get(key)
        if entry is None:
            return None
        # Move to end (most recently used).
        self._cache.move_to_end(key)
        return entry

    def put(self, prompt: str, max_tokens: int, entry: dict) -> None:
        key = self._key(prompt, max_tokens)
        self._cache[key] = entry
        self._cache.move_to_end(key)
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    def clear(self) -> None:
        self._cache.clear()


class AnswerGenerator:

    def __init__(self, cache_size: int = 64):
        self.model = OLLAMA_MODEL
        self._max_retries = 3
        # Module-level ollama.chat() uses the default localhost client. Using an
        # explicit Client honours OLLAMA_HOST and lets us set a real timeout so a
        # stuck server cannot hang the UI indefinitely.
        self._client = ollama.Client(host=OLLAMA_HOST, timeout=OLLAMA_TIMEOUT)
        # LRU cache so repeated prompts don't re-invoke the LLM.
        self._cache = AnswerCache(max_size=cache_size)

    def _chat_options(self, max_tokens: int) -> dict:
        """Common Ollama runtime options shared by non-streaming and streaming calls."""
        options = {
            "num_predict": max_tokens,
            "temperature": LLM_TEMPERATURE,
            "num_ctx": OLLAMA_CONTEXT_WINDOW,
        }
        # Let Ollama use more CPU cores for faster generation. Only add when the
        # user explicitly configured a value (OLLAMA_NUM_THREADS is a string env
        # or None by default so Ollama can autodetect).
        if OLLAMA_NUM_THREADS:
            try:
                options["num_thread"] = int(OLLAMA_NUM_THREADS)
            except (TypeError, ValueError):
                pass
        if OLLAMA_NUM_PARALLEL > 1:
            options["num_parallel"] = OLLAMA_NUM_PARALLEL
        return options

    def generate_answer(self, prompt, max_tokens: int = LLM_MAX_TOKENS):
        """
        Sends prompt to Ollama and returns
        answer, response time and model name.

        Retries up to 3 times on transient errors.
        """

        start_time = time.time()

        # Fast path: identical prompt already answered -> return cached result.
        cached = self._cache.get(prompt, max_tokens)
        if cached is not None:
            cached["cached"] = True
            return cached

        last_error = None
        for attempt in range(1, self._max_retries + 1):
            try:
                response = self._client.chat(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    options=self._chat_options(max_tokens),
                    keep_alive=OLLAMA_KEEP_ALIVE,
                )
                end_time = time.time()
                answer = response["message"]["content"]

                entry = {
                    "answer": answer,
                    "model": self.model,
                    "response_time": round(end_time - start_time, 2),
                    "cached": False,
                }
                self._cache.put(prompt, max_tokens, entry)
                return entry

            except ConnectionError as exc:
                last_error = RuntimeError(
                    "Ollama is not reachable. Make sure it's running (ollama serve)."
                )
                last_error.__cause__ = exc
                if attempt < self._max_retries:
                    time.sleep(2 ** attempt)

            except ResponseError as exc:
                last_error = RuntimeError(
                    f"Ollama error for model {self.model}: {exc}"
                )
                last_error.__cause__ = exc
                if attempt < self._max_retries:
                    time.sleep(1)

            except Exception as exc:
                last_error = RuntimeError(
                    "Answer generation failed. Check that Ollama is running locally."
                )
                last_error.__cause__ = exc
                if attempt < self._max_retries:
                    time.sleep(1)

        if last_error:
            raise last_error
        raise RuntimeError("Answer generation failed after retries.")

    def generate_answer_stream(self, prompt: str, max_tokens: Optional[int] = None) -> Generator[str, None, None]:
        """
        Stream tokens from Ollama. Yields str chunks as they arrive.
        Raises RuntimeError on connection / model errors.
        """
        if max_tokens is None:
            max_tokens = LLM_MAX_TOKENS
        try:
            stream = self._client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                options=self._chat_options(max_tokens),
                keep_alive=OLLAMA_KEEP_ALIVE,
            )
            for chunk in stream:
                token = chunk["message"]["content"]
                if token:
                    yield token
        except ConnectionError as exc:
            raise RuntimeError(
                "Ollama is not reachable. Start Ollama and make sure the model is available."
            ) from exc
        except ResponseError as exc:
            raise RuntimeError(
                f"Ollama returned an error for model {self.model}: {exc}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"Streaming answer generation failed: {exc}"
            ) from exc
