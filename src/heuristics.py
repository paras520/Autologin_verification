"""Heuristics for URL verification checks.

Both checks use the LLM via Langfuse + LiteLLM for intelligent analysis.
Prompt paths are configurable through environment variables so you can manage
and version them in Langfuse independently.

Expected env vars (on top of the shared Langfuse credentials):
  LANGFUSE_DIRECT_LOGIN_PROMPT   — default: test/autologin_Direct_Login_Check
  LANGFUSE_PROVIDER_MATCH_PROMPT — default: test/autologin_service_matcher
"""

from __future__ import annotations

import json
import logging
import os
import re
from uuid import uuid4

logger = logging.getLogger("autologin.heuristics")

DEFAULT_DIRECT_LOGIN_PROMPT = "test/autologin_Direct_Login_Check"
DEFAULT_PROVIDER_MATCH_PROMPT = "test/autologin_service_matcher"

# Cap visible text sent to these LLM checks (separate from the main LLM limit)
_HEURISTIC_VISIBLE_TEXT_LIMIT = 10_000


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _normalize_text(value) -> str:
    if isinstance(value, list):
        value = " ".join(str(item) for item in value if item is not None)
    elif value is None:
        value = ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _langfuse_is_configured() -> bool:
    return all(
        os.getenv(k)
        for k in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST")
    )


def _load_langfuse_helpers():
    from langfuse_helper import (
        build_messages,
        call_litellm,
        get_prompts_from_langfuse,
        parse_response,
    )
    return build_messages, call_litellm, get_prompts_from_langfuse, parse_response


def _build_page_variables(provider: str, service_name: str, page_result: dict) -> dict:
    """Build the template variables dict that will be injected into Langfuse prompts."""
    visible_text = _normalize_text(page_result.get("visible_text"))
    return {
        "provider": provider or "",
        "service_name": service_name or "",
        "url": page_result.get("final_url") or "",
        "page_title": _normalize_text(page_result.get("title")),
        "headings": json.dumps(page_result.get("headings") or [], ensure_ascii=False),
        "buttons": json.dumps(page_result.get("buttons") or [], ensure_ascii=False),
        "login_form_present": str(page_result.get("login_form_present", False)).lower(),
        "visible_text": visible_text[:_HEURISTIC_VISIBLE_TEXT_LIMIT],
    }


def _parse_notes(raw) -> list[str]:
    """Ensure notes is always a flat list of strings."""
    if isinstance(raw, list):
        return [str(n) for n in raw if n]
    if raw:
        return [str(raw)]
    return []


# ---------------------------------------------------------------------------
# Direct login-page check
# ---------------------------------------------------------------------------

async def assess_direct_login_page(
    provider: str,
    service_name: str,
    page_result: dict,
) -> dict:
    """Ask the LLM whether *page_result* represents a login portal.

    Returns:
        {
          "is_login_page": bool,
          "score": int  (0-100),
          "reason": str,
          "notes": list[str],
        }
    """
    prompt_path = os.getenv("LANGFUSE_DIRECT_LOGIN_PROMPT", DEFAULT_DIRECT_LOGIN_PROMPT)

    if not _langfuse_is_configured():
        logger.warning("[direct-login] Langfuse not configured — skipping")
        return {
            "is_login_page": False,
            "score": 0,
            "reason": "Langfuse not configured; direct login check skipped.",
            "notes": ["direct-login LLM check skipped — Langfuse credentials missing"],
        }

    session_id = f"direct-login-{uuid4()}"
    variables = _build_page_variables(provider, service_name, page_result)

    logger.info(
        "[direct-login] Calling LLM  prompt=%s  session=%s  provider=%s  service=%s",
        prompt_path, session_id, provider, service_name,
    )

    try:
        build_messages, call_litellm, get_prompts_from_langfuse, parse_response = (
            _load_langfuse_helpers()
        )

        system_prompt, user_prompt, config, prompt_obj = get_prompts_from_langfuse(
            prompt_path=prompt_path,
            session_id=session_id,
            variables=variables,
        )

        messages = build_messages(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        logger.debug("[direct-login] Sending %d messages to model=%s", len(messages), config.get("model"))

        response = await call_litellm(
            config=config,
            messages=messages,
            session_id=session_id,
            api_endpoint="/check/direct-login",
            prompt=prompt_obj,
        )

        parsed = parse_response(response, has_functions=False, has_tools=False)
        logger.info("[direct-login] Raw LLM response: %s", str(parsed)[:500])

        if isinstance(parsed, dict):
            is_login = bool(parsed.get("is_login_page", False))
            score = max(0, min(int(parsed.get("score", 0)), 100))
            reason = parsed.get("reason") or (
                "Page identified as a login portal."
                if is_login
                else "Page does not appear to be a login portal."
            )
            notes = _parse_notes(parsed.get("notes"))
            notes.append(f"langfuse_session_id={session_id}")

            logger.info(
                "[direct-login] Result: is_login_page=%s  score=%d  reason=%s",
                is_login, score, reason,
            )
            return {
                "is_login_page": is_login,
                "score": score,
                "reason": reason,
                "notes": notes,
            }

        logger.warning("[direct-login] LLM returned unstructured response: %s", str(parsed)[:300])
        return {
            "is_login_page": False,
            "score": 0,
            "reason": f"Direct login LLM returned unstructured response: {str(parsed)[:300]}",
            "notes": [f"langfuse_session_id={session_id}"],
        }

    except Exception as exc:
        logger.error("[direct-login] LLM call failed: %s", exc, exc_info=True)
        return {
            "is_login_page": False,
            "score": 0,
            "reason": f"Direct login LLM check failed: {exc}",
            "notes": [f"llm_error={exc}", f"langfuse_session_id={session_id}"],
        }


# ---------------------------------------------------------------------------
# Provider / service-name page-match check
# ---------------------------------------------------------------------------

async def assess_provider_service_match(
    provider: str,
    service_name: str,
    page_result: dict,
) -> dict:
    """Ask the LLM whether *page_result* belongs to the claimed provider/service.

    Returns:
        {
          "matched": bool,
          "score": int  (0-100),
          "reason": str,
          "notes": list[str],
        }
    """
    if not provider and not service_name:
        logger.info("[provider-match] Skipped — both provider and service_name are empty")
        return {
            "matched": True,
            "score": 0,
            "reason": "No provider or service name provided — match check skipped.",
            "notes": ["provider/service match skipped — both fields empty"],
        }

    prompt_path = os.getenv(
        "LANGFUSE_PROVIDER_MATCH_PROMPT", DEFAULT_PROVIDER_MATCH_PROMPT
    )

    if not _langfuse_is_configured():
        logger.warning("[provider-match] Langfuse not configured — skipping")
        return {
            "matched": True,
            "score": 0,
            "reason": "Langfuse not configured; provider/service match check skipped.",
            "notes": ["provider/service LLM check skipped — Langfuse credentials missing"],
        }

    session_id = f"provider-match-{uuid4()}"
    variables = _build_page_variables(provider, service_name, page_result)

    logger.info(
        "[provider-match] Calling LLM  prompt=%s  session=%s  provider=%s  service=%s",
        prompt_path, session_id, provider, service_name,
    )

    try:
        build_messages, call_litellm, get_prompts_from_langfuse, parse_response = (
            _load_langfuse_helpers()
        )

        system_prompt, user_prompt, config, prompt_obj = get_prompts_from_langfuse(
            prompt_path=prompt_path,
            session_id=session_id,
            variables=variables,
        )

        messages = build_messages(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        logger.debug("[provider-match] Sending %d messages to model=%s", len(messages), config.get("model"))

        response = await call_litellm(
            config=config,
            messages=messages,
            session_id=session_id,
            api_endpoint="/check/provider-match",
            prompt=prompt_obj,
        )

        parsed = parse_response(response, has_functions=False, has_tools=False)
        logger.info("[provider-match] Raw LLM response: %s", str(parsed)[:500])

        if isinstance(parsed, dict):
            matched = bool(parsed.get("matched", False))
            score = max(0, min(int(parsed.get("score", 0)), 100))
            reason = parsed.get("reason") or (
                "Page content matches the claimed provider/service."
                if matched
                else "Page content does NOT match the claimed provider/service."
            )
            notes = _parse_notes(parsed.get("notes"))
            notes.append(f"langfuse_session_id={session_id}")

            logger.info(
                "[provider-match] Result: matched=%s  score=%d  reason=%s",
                matched, score, reason,
            )
            return {
                "matched": matched,
                "score": score,
                "reason": reason,
                "notes": notes,
            }

        logger.warning("[provider-match] LLM returned unstructured response: %s", str(parsed)[:300])
        return {
            "matched": True,
            "score": 0,
            "reason": f"Provider match LLM returned unstructured response: {str(parsed)[:300]}",
            "notes": [f"langfuse_session_id={session_id}"],
        }

    except Exception as exc:
        logger.error("[provider-match] LLM call failed: %s", exc, exc_info=True)
        return {
            "matched": True,
            "score": 0,
            "reason": f"Provider/service match LLM check failed: {exc}",
            "notes": [f"llm_error={exc}", f"langfuse_session_id={session_id}"],
        }
