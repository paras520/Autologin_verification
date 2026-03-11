from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any
from uuid import uuid4

from src.models.internal_models import LLMDecision
from src.models.request_models import CheckRequest
from src.utils.heuristics import (
    assess_direct_login_page,
    assess_provider_service_match,
)
from src.utils.langfuse_helper import (
    build_messages,
    call_litellm,
    get_prompts_from_langfuse,
    parse_response,
)

logger = logging.getLogger("autologin.analysis_service")

DEFAULT_LANGFUSE_PROMPT_PATH = "autologin/login-portal-classifier"
LLM_VISIBLE_TEXT_LIMIT = 40000


def _normalize_proxy_env() -> None:
    if not os.getenv("LITELLM_PROXY_URL") and os.getenv("LITELLM_PROXY"):
        os.environ["LITELLM_PROXY_URL"] = os.getenv("LITELLM_PROXY", "")


def _get_prompt_path() -> str:
    return os.getenv("LANGFUSE_PROMPT_PATH", DEFAULT_LANGFUSE_PROMPT_PATH)


def _langfuse_is_configured() -> bool:
    required_env_vars = [
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_HOST",
    ]
    return all(os.getenv(env_var) for env_var in required_env_vars)


def _coerce_yes_no_flag(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None

    normalized = str(value).strip().lower()
    if normalized in {"yes", "true", "1"}:
        return True
    if normalized in {"no", "false", "0"}:
        return False
    return None


def _build_llm_variables(
    url: str,
    soft_errors: list[str],
    visible_text: str,
    page_result: dict[str, Any],
) -> tuple[dict[str, str], str]:
    llm_visible_text = visible_text[:LLM_VISIBLE_TEXT_LIMIT]
    truncated_for_llm = len(visible_text) > LLM_VISIBLE_TEXT_LIMIT

    page_metadata = {
        "title": page_result.get("title"),
        "headings": page_result.get("headings"),
        "buttons": page_result.get("buttons"),
        "login_form_present": page_result.get("login_form_present"),
        "final_url": page_result.get("final_url"),
        "visible_text_length": page_result.get("visible_text_length"),
        "visible_text_truncated": page_result.get("visible_text_truncated"),
        "visible_text_truncated_for_llm": truncated_for_llm,
    }

    variables = {
        "url": url,
        "soft_errors": json.dumps(soft_errors, ensure_ascii=False),
        "visible_text": llm_visible_text,
        "page_metadata": json.dumps(page_metadata, ensure_ascii=False),
    }

    return variables, llm_visible_text


async def run_main_analysis(
    url: str,
    soft_errors: list[str],
    visible_text: str,
    page_result: dict[str, Any],
) -> LLMDecision:
    prompt_path = _get_prompt_path()

    if not _langfuse_is_configured():
        logger.warning("Langfuse not configured — skipping main LLM analysis")
        return LLMDecision(
            inactive_flagged=False,
            reason=None,
            raw_output=None,
            error="Langfuse credentials are not configured.",
            session_id=None,
            prompt_path=prompt_path,
        )

    _normalize_proxy_env()
    session_id = f"url-check-{uuid4()}"
    variables, llm_visible_text = _build_llm_variables(
        url, soft_errors, visible_text, page_result
    )

    logger.info(
        "[LLM-main] Calling Langfuse prompt=%s  session=%s  url=%s",
        prompt_path,
        session_id,
        url,
    )

    try:
        system_prompt, user_prompt, config, prompt_obj = get_prompts_from_langfuse(
            prompt_path=prompt_path,
            session_id=session_id,
            variables=variables,
        )

        messages = build_messages(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        logger.debug(
            "[LLM-main] Sending %d messages to model=%s",
            len(messages),
            config.get("model"),
        )

        response = await call_litellm(
            config=config,
            messages=messages,
            session_id=session_id,
            api_endpoint="/check",
            tag_suffix="1",
            prompt=prompt_obj,
        )

        parsed_output = parse_response(response, has_functions=False, has_tools=False)
        logger.info("[LLM-main] Raw response: %s", str(parsed_output)[:500])

        decision_value = None
        decision_reason = None

        if isinstance(parsed_output, dict):
            decision_value = _coerce_yes_no_flag(
                parsed_output.get("inactive_flagged")
            )
            decision_reason = parsed_output.get("reason")

        if decision_value is None:
            logger.warning(
                "[LLM-main] Could not extract inactive_flagged from response"
            )
            return LLMDecision(
                inactive_flagged=False,
                reason=None,
                raw_output=parsed_output,
                error=(
                    "LLM response did not contain a valid inactive_flagged "
                    "yes/no value."
                ),
                session_id=session_id,
                prompt_path=prompt_path,
            )

        logger.info(
            "[LLM-main] Decision: inactive_flagged=%s  reason=%s",
            decision_value,
            decision_reason,
        )

        return LLMDecision(
            inactive_flagged=decision_value,
            reason=decision_reason,
            raw_output=parsed_output,
            error=None,
            session_id=session_id,
            prompt_path=prompt_path,
        )
    except Exception as exc:
        logger.error("[LLM-main] LLM call failed: %s", exc, exc_info=True)
        return LLMDecision(
            inactive_flagged=False,
            reason=None,
            raw_output={
                "soft_errors": soft_errors,
                "visible_text_preview": llm_visible_text[:500],
            },
            error=str(exc),
            session_id=session_id,
            prompt_path=prompt_path,
        )


async def run_parallel_checks(
    payload: CheckRequest,
    url: str,
    soft_errors: list[str],
    visible_text: str,
    page_result: dict[str, Any],
) -> tuple[LLMDecision, dict | None, dict | None]:
    login_type = payload.login_type.strip().lower()
    needs_login_check = login_type in ("direct", "navigation")
    has_provider_or_service = bool(payload.provider or payload.service_name)

    llm_coro = run_main_analysis(url, soft_errors, visible_text, page_result)
    direct_coro = (
        assess_direct_login_page(
            provider=payload.provider,
            service_name=payload.service_name,
            page_result=page_result,
        )
        if needs_login_check
        else asyncio.sleep(0)
    )
    provider_coro = (
        assess_provider_service_match(
            provider=payload.provider,
            service_name=payload.service_name,
            page_result=page_result,
        )
        if has_provider_or_service
        else asyncio.sleep(0)
    )

    llm_decision, direct_result, provider_result = await asyncio.gather(
        llm_coro,
        direct_coro,
        provider_coro,
    )

    return (
        llm_decision,
        direct_result if needs_login_check else None,
        provider_result if has_provider_or_service else None,
    )
