"""
Langfuse + LiteLLM Integration (Python 3.14 compatible)

The Langfuse Python SDK uses pydantic v1 which is broken on Python 3.14.
This module uses the raw Langfuse REST API directly for prompt fetching,
and passes tracing metadata to LiteLLM via the `metadata` param so that
litellm.success_callback = ["langfuse"] in app.py handles all tracing.
"""

import json
import logging
import os
import re
from urllib.parse import quote

import litellm
import requests

logger = logging.getLogger("autologin.langfuse_helper")

DEFAULT_LANGFUSE_LABEL = os.getenv("LANGFUSE_PROMPT_LABEL", "stage2")
LANGFUSE_TIMEOUT_SECONDS = 30
_PLACEHOLDER_PATTERN = re.compile(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}")


# ---------------------------------------------------------------------------
# Prompt fetching via raw REST API (SDK broken on Python 3.14)
# ---------------------------------------------------------------------------

def _fetch_prompt_raw(prompt_path: str) -> dict:
    host = os.getenv("LANGFUSE_HOST")
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")

    if not host or not public_key or not secret_key:
        raise ValueError("LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY must be set")

    encoded = quote(prompt_path, safe="")
    url = f"{host.rstrip('/')}/api/public/v2/prompts/{encoded}"

    resp = requests.get(
        url,
        params={"label": DEFAULT_LANGFUSE_LABEL},
        auth=(public_key, secret_key),
        timeout=LANGFUSE_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    return resp.json()


def _fill_placeholders(template: str, variables: dict | None) -> str:
    """Replace {{key}} placeholders with values from variables dict."""
    if not variables or not template:
        return template

    def replacer(match: re.Match) -> str:
        key = match.group(1)
        val = variables.get(key, match.group(0))
        return val if isinstance(val, str) else json.dumps(val, ensure_ascii=False)

    return _PLACEHOLDER_PATTERN.sub(replacer, template)


def _compile_prompt(raw_prompt, variables: dict | None):
    """Apply variable substitution to raw prompt (string or list of messages)."""
    if isinstance(raw_prompt, str):
        return _fill_placeholders(raw_prompt, variables)

    if isinstance(raw_prompt, list):
        compiled = []
        for msg in raw_prompt:
            if not isinstance(msg, dict):
                continue
            if msg.get("type") == "placeholder":
                name = msg.get("name")
                val = (variables or {}).get(name, "")
                if isinstance(val, list):
                    compiled.extend(val)
                elif val:
                    compiled.append({"role": "user", "content": str(val)})
                continue
            m = dict(msg)
            if isinstance(m.get("content"), str):
                m["content"] = _fill_placeholders(m["content"], variables)
            compiled.append(m)
        return compiled

    return raw_prompt


def get_prompts_from_langfuse(
    prompt_path: str,
    session_id: str,
    variables: dict | None = None,
):
    """
    Fetch prompt from Langfuse REST API and compile variables into it.

    Returns:
        (system_prompt, user_prompt, config, prompt_obj)
        prompt_obj is the raw JSON dict — passed to metadata for LiteLLM tracing.
    """
    prompt_raw = _fetch_prompt_raw(prompt_path)
    compiled = _compile_prompt(prompt_raw.get("prompt"), variables)

    system_prompt: str = ""
    user_prompt: str = ""

    if isinstance(compiled, list):
        for msg in compiled:
            role = msg.get("role", "").lower()
            content = msg.get("content", "")
            if role == "system":
                system_prompt = content
            elif role == "user":
                user_prompt = content
    elif isinstance(compiled, str):
        user_prompt = compiled

    config: dict = prompt_raw.get("config") or {}

    return system_prompt, user_prompt, config, prompt_raw


# ---------------------------------------------------------------------------
# Message builder
# ---------------------------------------------------------------------------

def build_messages(
    system_prompt: str | None = None,
    user_prompt: str | None = None,
    images: list | None = None,
    pdf_file: bytes | None = None,
    additional_text: str | None = None,
) -> list:
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    user_content = []
    parts = []
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
# LiteLLM caller
# ---------------------------------------------------------------------------

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
    Call LiteLLM through the proxy.

    Tracing metadata is passed via the `metadata` kwarg which LiteLLM
    forwards to Langfuse when litellm.success_callback = ["langfuse"].
    """
    model = config.get("model")
    if not model:
        raise ValueError("'model' not specified in Langfuse prompt config")

    provider = config.get("provider", "")

    params: dict = {"model": model, "messages": messages}

    # Route through LiteLLM proxy
    proxy_url = os.getenv("LITELLM_PROXY_URL")
    if proxy_url:
        params["api_base"] = proxy_url
        params["api_key"] = (
            os.getenv("LITELLM_PROXY_API_KEY") or os.getenv("OPENAI_API_KEY")
        )

    # Langfuse tracing metadata — forwarded by LiteLLM to Langfuse callback
    base_tag = "autologin_verification"
    tags = [base_tag]
    if tag_suffix:
        tags.append(tag_suffix)
    if extra_tags:
        tags.extend(extra_tags)

    params["metadata"] = {
        "session_id": session_id or "",
        "tags": tags,
        "api_endpoint": api_endpoint or "",
    }

    # Model params from Langfuse config
    for key in ("temperature", "max_tokens", "max_completion_tokens",
                "seed", "reasoning_effort", "top_p"):
        if key in config:
            params[key] = config[key]

    if functions:
        params["functions"] = functions
        if provider == "openai":
            params["function_call"] = {"name": functions[0]["name"]}

    if tools:
        params["tools"] = tools
        params["tool_choice"] = "auto"

    response = await litellm.acompletion(**params)
    return response


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def parse_response(response, has_functions: bool = False, has_tools: bool = False):
    message = response.choices[0].message

    if has_functions and hasattr(message, "function_call") and message.function_call:
        return json.loads(message.function_call.arguments)

    if has_tools and hasattr(message, "tool_calls") and message.tool_calls:
        return json.loads(message.tool_calls[0].function.arguments)

    content = message.content
    if content:
        stripped = content.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```[a-zA-Z]*\n?", "", stripped)
            stripped = re.sub(r"\n?```$", "", stripped).strip()
        try:
            return json.loads(stripped)
        except (json.JSONDecodeError, TypeError):
            return content

    return content


# ---------------------------------------------------------------------------
# All-in-one convenience
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
