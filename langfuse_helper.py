"""
Langfuse + Bifrost integration.

Prompts are loaded directly from **Langfuse** (no local-file fallback) and all
LLM calls are routed through the **Bifrost** LLM proxy. Bifrost has no native
Langfuse integration, so every call is wrapped in a manual Langfuse
trace/generation observation here (per the `bifrost-langfuse` skill).

Public API preserved for callers (e.g. `src/heuristics.py`):
  - get_prompts_from_langfuse(...)
  - build_messages(...)
  - call_litellm(...)           # now talks to Bifrost
  - parse_response(...)
  - get_and_call_litellm(...)

Required env vars:
  LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY
  LANGFUSE_PROMPT_LABEL          (optional, default "production")
  BIFROST_PROXY                  (e.g. https://m90-llm-proxy-stage2-...run.app)
  BIFROST_API_KEY
  BIFROST_AUTH_SCHEME            ("Basic" or "Bearer", default "Basic")
  BIFROST_MODEL                  (FALLBACK model — used only when the Langfuse
                                  prompt config has no model declared; the
                                  prompt's model always takes priority)
  BIFROST_TAGS                   (comma-separated default tags)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from src.config import config as app_config

logger = logging.getLogger("autologin.langfuse_helper")


# ---------------------------------------------------------------------------
# Langfuse client (lazy, singleton)
# ---------------------------------------------------------------------------

_LANGFUSE_CLIENT = None
_LANGFUSE_INIT_TRIED = False


def _get_langfuse():
    """Return a Langfuse client, or None if Langfuse is not configured / SDK missing."""
    global _LANGFUSE_CLIENT, _LANGFUSE_INIT_TRIED
    if _LANGFUSE_CLIENT is not None or _LANGFUSE_INIT_TRIED:
        return _LANGFUSE_CLIENT

    _LANGFUSE_INIT_TRIED = True

    if not all(
        os.getenv(k)
        for k in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST")
    ):
        logger.info("Langfuse env vars not set — tracing disabled.")
        return None

    try:
        from langfuse import get_client  # type: ignore

        _LANGFUSE_CLIENT = get_client()
        return _LANGFUSE_CLIENT
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Langfuse client init failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Bifrost configuration
# ---------------------------------------------------------------------------

_BIFROST_PATH = "/v1/chat/completions"


def _bifrost_url() -> str:
    base = app_config.bifrost_proxy
    # Normalise: bifrostProxy may or may not include the /v1 segment
    if base.endswith("/v1"):
        base = base[:-3]
    if not base:
        raise RuntimeError(
            "bifrostProxy must be set in app.config.json or BIFROST_PROXY env var"
        )
    return f"{base}{_BIFROST_PATH}"


def _bifrost_auth_header() -> str:
    scheme = app_config.bifrost_auth_scheme.strip()
    key = (os.getenv("BIFROST_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("BIFROST_API_KEY env var is required")
    return f"{scheme} {key}"


def _bifrost_default_tags() -> list[str]:
    raw = app_config.bifrost_tags
    return [t.strip() for t in raw.split(",") if t.strip()]


# ---------------------------------------------------------------------------
# Model name translation (LiteLLM / prompt-config → Bifrost format)
# ---------------------------------------------------------------------------

LITELLM_TO_BIFROST: dict[str, str] = {
    # OpenAI models
    "gpt-4o":                              "openai/gpt-4o",
    "gpt-4o-mini":                         "openai/gpt-4o-mini",
    "o4-mini":                             "openai/o4-mini",
    # Vertex models — proxy exposes these as vertex/<model>
    "gemini-2.5-pro":                      "vertex/gemini-2.5-pro",
    "gemini-2.5-flash":                    "vertex/gemini-2.5-flash",
    "gemini-2.5-flash-lite":               "vertex/gemini-2.5-flash-lite",
    "gemini-3-flash-preview":              "vertex/gemini-3-flash-preview",
    "gemini-3.1-pro":                      "vertex/gemini-3.1-pro-preview",
    "claude-sonnet-4":                     "vertex/claude-sonnet-4-5",
    "claude-opus-4":                       "vertex/claude-opus-4-5",
    # Old Bifrost openai/vertex/X → vertex/X  (covers saved Langfuse prompt configs)
    "openai/vertex/gemini-2.5-pro":        "vertex/gemini-2.5-pro",
    "openai/vertex/gemini-2.5-flash":      "vertex/gemini-2.5-flash",
    "openai/vertex/gemini-3-flash-preview":"vertex/gemini-3-flash-preview",
    "openai/vertex/gemini-3.1-pro":        "vertex/gemini-3.1-pro-preview",
    # vertex_ai/ → vertex/  (previous attempted fix, keep as fallback)
    "vertex_ai/gemini-2.5-pro":            "vertex/gemini-2.5-pro",
    "vertex_ai/gemini-2.5-flash":          "vertex/gemini-2.5-flash",
    "vertex_ai/gemini-3-flash-preview":    "vertex/gemini-3-flash-preview",
}


def to_bifrost_model(model: str) -> str:
    """Translate a model name to the proxy's model ID format.

    Explicit aliases in LITELLM_TO_BIFROST are applied first.
    Old Bifrost `openai/vertex/X` and `vertex_ai/X` names are rewritten to `vertex/X`.
    Everything else passes through unchanged.
    """
    if not model:
        return "openai/gpt-4o"

    if model in LITELLM_TO_BIFROST:
        return LITELLM_TO_BIFROST[model]

    # Old Bifrost routing prefix → proxy vertex/ format
    if model.startswith("openai/vertex/"):
        return "vertex/" + model[len("openai/vertex/"):]

    # vertex_ai/ (previous attempted name) → vertex/
    if model.startswith("vertex_ai/"):
        return "vertex/" + model[len("vertex_ai/"):]

    # Already correct (vertex/..., openai/..., openrouter/...) or bare name
    return model


def resolve_model_from_prompt_config(
    prompt_config: dict | None,
    fallback_model: str | None = None,
    use_backup: bool = False,
    override_model: str | None = None,
) -> tuple[str, dict]:
    """Resolve the Bifrost model and extra body params from a prompt config.

    Priority (highest first):
      1. ``override_model``                  — explicit programmatic override
      2. ``prompt_config["model"]``          — direct model+provider shape
      3. ``prompt_config["primary_model"]``  — SLA shape (or ``backup_model``
                                                when ``use_backup=True``)
      4. ``fallback_model``                  — env-var fallback for prompts
                                                with no model in config
      5. ``"openai/gpt-4o"`` default

    Note: previously ``BIFROST_MODEL`` was treated as a hard override and would
    win over the Langfuse prompt config. This was inverted so each prompt can
    pin its own model in Langfuse without code/env changes; ``BIFROST_MODEL``
    is now a fallback only.
    """
    if override_model:
        return to_bifrost_model(override_model), {}

    config = prompt_config or {}
    extra_body: dict = {}

    # Shape 2 — direct {model, provider, ...}
    if "model" in config:
        provider = config.get("provider", "openai")
        model_name = config["model"]

        if provider == "openai":
            bifrost_model = f"openai/{model_name}"
        elif provider in ("vertex", "google", "vertex_ai"):
            bifrost_model = f"vertex/{model_name}"
        else:
            bifrost_model = f"openai/{provider}/{model_name}"

        # Bare provider/model paths (e.g. "Qwen/...") use to_bifrost_model
        # so alias maps still apply.
        bifrost_model = to_bifrost_model(model_name) if "/" in model_name else bifrost_model

        known_keys = {"model", "provider"}
        for k, v in config.items():
            if k not in known_keys:
                extra_body[k] = v

        return bifrost_model, extra_body

    # Shape 1 — SLA-style {primary_model, backup_model, ...}
    model_key = "backup_model" if use_backup else "primary_model"
    raw_model = config.get(model_key) or config.get("primary_model")
    if raw_model:
        return to_bifrost_model(raw_model), {}

    if fallback_model:
        return to_bifrost_model(fallback_model), {}

    return "openai/gpt-4o", {}


# ---------------------------------------------------------------------------
# LLM call logger (local JSONL for offline debugging) — unchanged contract
# ---------------------------------------------------------------------------

_LLM_LOG_DIR = Path(__file__).resolve().parent / "output" / "pipeline_logs"
_LLM_LOG_DIR.mkdir(parents=True, exist_ok=True)
_LLM_LOG_FILE = _LLM_LOG_DIR / f"llm_calls_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"


def _log_llm_call(
    call_id: str,
    stage: str,
    model: str,
    api_base: str,
    messages: list,
    response: "BifrostResponse | None" = None,
    error: str = "",
    elapsed_ms: float = 0,
    extra: dict | None = None,
):
    entry: dict[str, Any] = {
        "ts": datetime.now().isoformat(),
        "call_id": call_id,
        "stage": stage,
        "model": model,
        "api_base": api_base,
        "elapsed_ms": round(elapsed_ms, 1),
        "error": error,
        "messages_preview": [
            {
                "role": m.get("role"),
                "content_preview": (
                    m.get("content")
                    if isinstance(m.get("content"), str)
                    else str(m.get("content"))
                )[:1500],
            }
            for m in messages
        ],
    }
    if response is not None and not error:
        try:
            choice = response.choices[0]
            msg = choice.message
            usage = response.usage
            entry["response"] = {
                "content": (msg.content or "")[:2500],
                "reasoning": (getattr(msg, "reasoning", "") or "")[:2500],
                "finish_reason": choice.finish_reason,
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            }
        except Exception as e:
            entry["response"] = {"parse_error": str(e)}
    if extra:
        entry.update(extra)

    try:
        with open(_LLM_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"LLM log write failed: {e}")

    if error:
        logger.warning(
            f"[LLM] {stage} model={model} {elapsed_ms:.0f}ms ERROR={error[:200]}"
        )
    elif response is not None:
        tokens = response.usage.total_tokens
        logger.info(
            f"[LLM] {stage} model={model} {elapsed_ms:.0f}ms tokens={tokens} OK"
        )


# ---------------------------------------------------------------------------
# Prompt loading — Langfuse is the single source of truth
# ---------------------------------------------------------------------------

def _stringify_variables(variables: dict | None) -> dict:
    """Langfuse's prompt.compile(**vars) interpolates raw string values; convert
    lists/dicts to JSON strings so they render cleanly inside the template."""
    if not variables:
        return {}
    out: dict[str, str] = {}
    for k, v in variables.items():
        if v is None:
            out[k] = ""
        elif isinstance(v, (str, int, float, bool)):
            out[k] = str(v)
        else:
            out[k] = json.dumps(v, ensure_ascii=False)
    return out


def get_prompts_from_langfuse(
    prompt_path: str,
    session_id: str,
    variables: dict | None = None,
):
    """
    Fetch a prompt from Langfuse, compile placeholders with ``variables``, and
    return its components.

    Returns:
        (system_prompt, user_prompt, config, prompt_obj)

    ``prompt_obj`` is the Langfuse Prompt SDK object — forwarded into
    :func:`call_litellm` so it can be linked to the generation in Langfuse.

    Raises:
        RuntimeError: if Langfuse is not configured or the prompt cannot be
                      fetched. There is intentionally no local-file fallback
                      so prompt drift between code and Langfuse is impossible.
    """
    langfuse = _get_langfuse()
    if langfuse is None:
        raise RuntimeError(
            "Langfuse is not configured — set LANGFUSE_PUBLIC_KEY, "
            "LANGFUSE_SECRET_KEY and LANGFUSE_HOST."
        )

    # os.getenv allows runtime override (e.g. eval scripts that swap labels on the fly)
    label = os.getenv("LANGFUSE_PROMPT_LABEL") or app_config.langfuse_prompt_label
    try:
        prompt_obj = langfuse.get_prompt(prompt_path, label=label)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to fetch Langfuse prompt '{prompt_path}' (label='{label}'): {exc}"
        ) from exc

    safe_vars = _stringify_variables(variables)
    try:
        compiled = prompt_obj.compile(**safe_vars)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to compile Langfuse prompt '{prompt_path}' (label='{label}'): {exc}"
        ) from exc

    system_prompt: str = ""
    user_prompt: str = ""

    if isinstance(compiled, list):
        for msg in compiled:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", "")).lower()
            content = msg.get("content", "") or ""
            if role == "system":
                system_prompt = content
            elif role == "user":
                user_prompt = content
    elif isinstance(compiled, str):
        user_prompt = compiled

    config: dict = getattr(prompt_obj, "config", {}) or {}
    return system_prompt, user_prompt, config, prompt_obj


# ---------------------------------------------------------------------------
# Message builder (unchanged contract)
# ---------------------------------------------------------------------------

def build_messages(
    system_prompt: str | None = None,
    user_prompt: str | None = None,
    images: list | None = None,
    pdf_file: bytes | None = None,
    additional_text: str | None = None,
) -> list:
    messages: list[dict] = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    user_content: list[dict] = []
    parts: list[str] = []
    if user_prompt:
        parts.append(user_prompt)
    if additional_text:
        parts.append(additional_text)
    if parts:
        user_content.append({"type": "text", "text": "\n".join(parts)})

    if images:
        import base64
        for img in images:
            b64 = base64.b64encode(img).decode("utf-8")
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })

    if pdf_file:
        import base64
        b64 = base64.b64encode(pdf_file).decode("utf-8")
        user_content.append({
            "type": "file",
            "file": {"file_data": f"data:application/pdf;base64,{b64}"},
        })

    if user_content:
        if len(user_content) == 1 and user_content[0]["type"] == "text":
            messages.append({"role": "user", "content": user_content[0]["text"]})
        else:
            messages.append({"role": "user", "content": user_content})

    return messages


# ---------------------------------------------------------------------------
# Bifrost response adapter — exposes the attribute access shape parse_response
# expects (i.e. mirrors the OpenAI Python SDK response).
# ---------------------------------------------------------------------------

class _FCall:
    def __init__(self, raw: dict):
        self.name = raw.get("name")
        self.arguments = raw.get("arguments") or "{}"


class _ToolCall:
    def __init__(self, raw: dict):
        self.id = raw.get("id")
        self.type = raw.get("type")
        self.function = _FCall(raw.get("function") or {})


class _Msg:
    def __init__(self, raw: dict):
        self.content = raw.get("content") or ""
        self.reasoning = raw.get("reasoning") or raw.get("reasoning_content") or ""
        fc = raw.get("function_call")
        self.function_call = _FCall(fc) if fc else None
        tcs = raw.get("tool_calls") or []
        self.tool_calls = [_ToolCall(t) for t in tcs] if tcs else None


class _Choice:
    def __init__(self, raw: dict):
        self.message = _Msg(raw.get("message") or {})
        self.finish_reason = raw.get("finish_reason")
        self.index = raw.get("index", 0)


class _Usage:
    def __init__(self, raw: dict):
        self.prompt_tokens = raw.get("prompt_tokens", 0) or 0
        self.completion_tokens = raw.get("completion_tokens", 0) or 0
        self.total_tokens = raw.get("total_tokens", 0) or (
            self.prompt_tokens + self.completion_tokens
        )


class BifrostResponse:
    """OpenAI-shaped wrapper over the raw Bifrost JSON response."""

    def __init__(self, raw: dict, headers: dict | None = None):
        self._raw = raw
        self.headers = headers or {}
        self.id = raw.get("id")
        self.model = raw.get("model")
        self.choices = [_Choice(c) for c in raw.get("choices", [])]
        self.usage = _Usage(raw.get("usage") or {})


# ---------------------------------------------------------------------------
# Bifrost HTTP caller + Langfuse instrumentation
# ---------------------------------------------------------------------------

_RETRYABLE_TOKENS = (
    "500", "502", "503", "504", "524",
    "timeout", "Timeout", "TimeoutError",
    "rate_limit", "RateLimit",
    "ECONNRESET",
)


async def _post_bifrost(
    payload: dict,
    timeout: float = 300.0,
) -> tuple[dict, dict]:
    url = _bifrost_url()
    headers = {
        "Authorization": _bifrost_auth_header(),
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        res = await client.post(url, headers=headers, json=payload)
        res.raise_for_status()
        return res.json(), dict(res.headers)


async def call_litellm(
    config: dict,
    messages: list,
    session_id: str,
    api_endpoint: str | None = None,
    tag_suffix: str | None = None,
    extra_tags: list[str] | None = None,
    functions: list | None = None,
    tools: list | None = None,
    prompt=None,
):
    """
    Route a chat-completion request through Bifrost and emit a Langfuse trace.

    The function name is preserved (despite no longer using LiteLLM) so existing
    callers in `src/heuristics.py` and elsewhere keep working without changes.

    Args:
        config:        Prompt config dict (may contain ``model``, ``temperature``,
                       ``max_tokens``, ``seed``, etc.).
        messages:      OpenAI-style message list.
        session_id:    Trace session id (groups related calls in Langfuse).
        api_endpoint:  Optional logical endpoint label (e.g. "/check/extractor").
        tag_suffix:    Stage label ("extractor", "matcher", "classifier", ...).
        extra_tags:    Additional tags to attach to the trace.
        functions/tools: OpenAI function-calling / tool-calling payloads.
        prompt:        Raw prompt dict from get_prompts_from_langfuse (forwarded
                       into trace metadata for offline replay).

    Returns:
        BifrostResponse — duck-typed to the OpenAI SDK response shape so
        ``parse_response`` continues to work.
    """
    bifrost_model, extra_body = resolve_model_from_prompt_config(
        prompt_config=config,
        fallback_model=app_config.bifrost_model or None,
    )

    stage_label = (tag_suffix or "").strip() or "unknown"
    call_id = uuid.uuid4().hex[:12]

    payload: dict = {"model": bifrost_model, "messages": messages}
    for k in ("temperature", "max_tokens", "max_completion_tokens", "top_p", "seed"):
        if k in (config or {}):
            payload[k] = config[k]
    for k, v in (extra_body or {}).items():
        payload.setdefault(k, v)

    if functions:
        payload["functions"] = functions
        if (config or {}).get("provider") == "openai":
            payload["function_call"] = {"name": functions[0]["name"]}
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    elif "tool_choice" in payload:
        # Prompt config may carry a default tool_choice; drop it when the
        # caller didn't actually supply tools (otherwise providers 400).
        payload.pop("tool_choice", None)

    tags = list(_bifrost_default_tags())
    if stage_label and stage_label != "unknown":
        tags.append(stage_label)
    for t in extra_tags or []:
        if t and t not in tags:
            tags.append(t)

    prompt_version = getattr(prompt, "version", None) if prompt is not None else None
    prompt_name = getattr(prompt, "name", None) if prompt is not None else None
    prompt_label = getattr(prompt, "label", None) if prompt is not None else None

    trace_metadata: dict[str, Any] = {
        "model": bifrost_model,
        "stage": stage_label,
        "session_id": session_id,
        "api_endpoint": api_endpoint,
        "call_id": call_id,
        "prompt_name": prompt_name,
        "prompt_version": prompt_version,
        "prompt_label": prompt_label,
    }

    langfuse_prompt_obj = prompt if prompt is not None else None

    langfuse = _get_langfuse()
    max_attempts = int(os.getenv("BIFROST_MAX_ATTEMPTS", "5"))
    backoff_base = float(os.getenv("BIFROST_BACKOFF_BASE", "3"))
    backoff_cap = float(os.getenv("BIFROST_BACKOFF_CAP", "60"))

    model_parameters = {
        k: payload[k]
        for k in ("temperature", "max_tokens", "max_completion_tokens", "top_p", "seed")
        if k in payload
    }

    async def _post_with_retries() -> BifrostResponse:
        last_exc: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            start = time.time()
            try:
                raw_json, headers = await _post_bifrost(payload)
                elapsed_ms = (time.time() - start) * 1000
                resp = BifrostResponse(raw_json, headers=headers)

                _log_llm_call(
                    call_id=call_id,
                    stage=stage_label,
                    model=bifrost_model,
                    api_base=_bifrost_url(),
                    messages=messages,
                    response=resp,
                    elapsed_ms=elapsed_ms,
                    extra={
                        "route": "bifrost",
                        "attempt": attempt,
                        "provider": headers.get("x-provider", "unknown"),
                    },
                )
                return resp

            except httpx.HTTPStatusError as e:
                elapsed_ms = (time.time() - start) * 1000
                err_body = ""
                try:
                    err_body = e.response.text[:500]
                except Exception:
                    pass
                err_str = f"HTTP {e.response.status_code}: {err_body}"
                last_exc = e
                retryable = e.response.status_code in (429, 500, 502, 503, 504, 524)
            except Exception as e:
                elapsed_ms = (time.time() - start) * 1000
                err_str = str(e)
                last_exc = e
                retryable = any(tok in err_str for tok in _RETRYABLE_TOKENS)

            _log_llm_call(
                call_id=call_id,
                stage=stage_label,
                model=bifrost_model,
                api_base=_bifrost_url(),
                messages=messages,
                error=err_str,
                elapsed_ms=elapsed_ms,
                extra={"route": "bifrost", "attempt": attempt, "retryable": retryable},
            )

            if not retryable or attempt == max_attempts:
                raise last_exc

            wait_s = min(backoff_cap, backoff_base * (2 ** (attempt - 1)))
            logger.warning(
                f"[LLM] {stage_label} {bifrost_model} attempt {attempt}/{max_attempts} "
                f"failed (retryable), backing off {wait_s:.0f}s: {err_str[:150]}"
            )
            await asyncio.sleep(wait_s)

        # Unreachable but mypy-safe
        raise last_exc  # type: ignore[misc]

    # Fast path: Langfuse not configured → just call Bifrost.
    if langfuse is None:
        return await _post_with_retries()

    # Traced path: open the generation FIRST (this implicitly creates a trace),
    # then update trace-level metadata while we're inside the active span
    # context, then call Bifrost.
    obs_kwargs: dict[str, Any] = {
        "name": f"llm-call:{stage_label}",
        "model": bifrost_model,
        "input": messages,
        "model_parameters": model_parameters,
    }
    if langfuse_prompt_obj is not None:
        obs_kwargs["prompt"] = langfuse_prompt_obj

    try:
        observation_cm = langfuse.start_as_current_generation(**obs_kwargs)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("start_as_current_generation failed: %s", exc)
        return await _post_with_retries()

    with observation_cm as generation:
        try:
            langfuse.update_current_trace(
                name=f"bifrost-{stage_label}",
                session_id=session_id or None,
                tags=tags or None,
                metadata=trace_metadata,
                input=messages,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("langfuse.update_current_trace failed: %s", exc)

        try:
            resp = await _post_with_retries()
        except Exception as exc:
            try:
                generation.update(level="ERROR", status_message=str(exc)[:500])
            except Exception:  # pragma: no cover - defensive
                pass
            try:
                langfuse.flush()
            except Exception:  # pragma: no cover - defensive
                pass
            raise

        try:
            output_msg = resp.choices[0].message
            output_text = output_msg.content or output_msg.reasoning or ""
            generation.update(
                output=output_text,
                model=resp.model or bifrost_model,
                metadata={"provider": resp.headers.get("x-provider", "unknown")},
                usage_details={
                    "input": resp.usage.prompt_tokens,
                    "output": resp.usage.completion_tokens,
                },
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("generation.update failed: %s", exc)

    try:
        langfuse.flush()
    except Exception:  # pragma: no cover - defensive
        pass

    return resp


# ---------------------------------------------------------------------------
# Response parser (unchanged behaviour)
# ---------------------------------------------------------------------------

def _remap_nonascii_outside_strings(source: str, replacement: str) -> str:
    """Walk `source` and replace every non-ASCII character that appears outside
    a JSON string literal with `replacement`.  Characters inside strings are
    left untouched so valid Unicode values are preserved.
    """
    buf: list[str] = []
    in_str = False
    esc = False
    for ch in source:
        if esc:
            buf.append(ch)
            esc = False
            continue
        if in_str:
            if ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            buf.append(ch)
        else:
            if ch == '"':
                in_str = True
                buf.append(ch)
            elif ord(ch) > 127:
                buf.append(replacement)
            else:
                buf.append(ch)
    return "".join(buf)


def _try_recover_json(text: str):
    """Attempt to parse JSON that has stray non-ASCII garbage outside string values.

    Gemini-2.5-flash occasionally inserts CJK or other Unicode characters in
    place of structural JSON tokens (e.g. `縱` replacing `],\\n  `).  We try
    three repair strategies in order:

    1. Replace every non-ASCII outside-string char with `],\\n  `
       — covers the most common case where the char replaced a '],' separator.
    2. Replace with `,\\n  ` — covers a missing comma between values.
    3. Strip entirely — covers pure decoration chars that add no structure.

    Returns the parsed object (dict or list) on success, or None.
    """
    for repl in ("],\n  ", ",\n  ", ""):
        candidate = _remap_nonascii_outside_strings(text, repl)
        try:
            result = json.loads(candidate)
            if isinstance(result, (dict, list)):
                logger.debug(
                    "[parse_response] JSON recovered using non-ASCII replacement %r", repl or "<strip>"
                )
                return result
        except (json.JSONDecodeError, TypeError):
            continue
    return None


def parse_response(response, has_functions: bool = False, has_tools: bool = False):
    message = response.choices[0].message

    if has_functions and getattr(message, "function_call", None):
        return json.loads(message.function_call.arguments)

    if has_tools and getattr(message, "tool_calls", None):
        return json.loads(message.tool_calls[0].function.arguments)

    content = message.content
    reasoning = getattr(message, "reasoning", None)
    text = content or reasoning or ""

    if text:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```[a-zA-Z]*\n?", "", stripped)
            stripped = re.sub(r"\n?```$", "", stripped).strip()
        try:
            return json.loads(stripped)
        except (json.JSONDecodeError, TypeError):
            pass

        # Recovery pass: Gemini sometimes inserts stray non-ASCII characters
        # (e.g. CJK chars) between structural JSON tokens, breaking the parse.
        # Try the three replacement strategies before giving up.
        recovered = _try_recover_json(stripped)
        if recovered is not None:
            logger.warning(
                "[parse_response] recovered JSON after non-ASCII corruption in response"
            )
            return recovered

        # Final fallback: use json-repair to handle structural corruption
        # (e.g. Gemini hallucinating stray words mid-array/object).
        try:
            from json_repair import repair_json  # type: ignore
            repaired = repair_json(stripped, return_objects=True, skip_json_loads=True)
            if isinstance(repaired, (dict, list)):
                logger.warning(
                    "[parse_response] recovered JSON via json-repair (structurally malformed response)"
                )
                return repaired
        except Exception:
            pass

        if reasoning and not content:
            last_open = stripped.rfind("{")
            last_close = stripped.rfind("}")
            if last_open != -1 and last_close > last_open:
                candidate = stripped[last_open:last_close + 1]
                try:
                    return json.loads(candidate)
                except (json.JSONDecodeError, TypeError):
                    pass
            first_open = stripped.find("{")
            if first_open != -1:
                depth = 0
                in_str = False
                escape = False
                for i in range(first_open, len(stripped)):
                    ch = stripped[i]
                    if escape:
                        escape = False
                        continue
                    if ch == "\\":
                        escape = True
                        continue
                    if ch == '"':
                        in_str = not in_str
                        continue
                    if in_str:
                        continue
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            try:
                                return json.loads(stripped[first_open:i + 1])
                            except (json.JSONDecodeError, TypeError):
                                break

        return text

    return text


# ---------------------------------------------------------------------------
# All-in-one convenience (unchanged contract)
# ---------------------------------------------------------------------------

async def get_and_call_litellm(
    prompt_path: str,
    session_id: str,
    api_endpoint: str | None = None,
    tag_suffix: str | None = None,
    variables: dict | None = None,
    pdf_file: bytes | None = None,
    images: list | None = None,
    functions: list | None = None,
    tools: list | None = None,
    additional_text: str | None = None,
):
    system_prompt, user_prompt, config, prompt_obj = get_prompts_from_langfuse(
        prompt_path=prompt_path,
        session_id=session_id,
        variables=variables,
    )
    messages = build_messages(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        pdf_file=pdf_file,
        images=images,
        additional_text=additional_text,
    )
    response = await call_litellm(
        config=config,
        messages=messages,
        session_id=session_id,
        api_endpoint=api_endpoint,
        tag_suffix=tag_suffix,
        functions=functions,
        tools=tools,
        prompt=prompt_obj,
    )
    return parse_response(
        response,
        has_functions=bool(functions),
        has_tools=bool(tools),
    )
