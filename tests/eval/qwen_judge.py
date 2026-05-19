"""Shared Qwen judge model for DeepEval metrics + optimizer.

Why this exists:
  When you pass `model="Qwen/Qwen3.5-9B"` (a string) to GEval / DAGMetric /
  PromptOptimizer, DeepEval auto-picks `OpenAIModel`, which calls
  `client.beta.chat.completions.parse(..., response_format=schema)`. Our Qwen
  vLLM proxy at Runpod does NOT support OpenAI's structured-outputs API
  reliably and returns HTTP 500 (InternalServerError) under that endpoint.

  DeepEval's `LocalModel` instead uses plain `chat.completions.create(...)` +
  post-hoc JSON parsing via `trim_and_load_json`, which works perfectly with
  Qwen.

Usage:
  from tests.eval.qwen_judge import get_qwen_judge
  metric = GEval(..., model=get_qwen_judge())
"""
from __future__ import annotations

import os
from functools import lru_cache

from deepeval.models.llms.local_model import LocalModel


@lru_cache(maxsize=1)
def get_qwen_judge() -> LocalModel:
    base_url = (
        os.getenv("QWEN_BASE_URL")
        or os.getenv("LITELLM_PROXY_URL")
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("OPENAI_API_BASE")
    )
    api_key = (
        os.getenv("QWEN_API_KEY")
        or os.getenv("LITELLM_PROXY_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    model_name = os.getenv("QWEN_MODEL", "Qwen/Qwen3.5-9B")
    if not base_url or not api_key:
        raise RuntimeError(
            "QWEN_BASE_URL and QWEN_API_KEY (or OPENAI_BASE_URL/OPENAI_API_KEY) "
            "must be set for the Qwen judge."
        )
    # Critical generation params:
    #   - max_tokens=4096: Qwen reasoning generates massive chains (5000+ toks).
    #     Without a cap, requests run >100s and trip Cloudflare's 524 timeout.
    #   - extra_body.chat_template_kwargs.enable_thinking=False:
    #     Tells Qwen3 vLLM to skip <think> blocks, returning concise JSON
    #     directly. Massive latency reduction (~50-80%).
    return LocalModel(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=0.0,
        format="json",
        generation_kwargs={
            "max_tokens": 4096,
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": False},
            },
        },
    )
