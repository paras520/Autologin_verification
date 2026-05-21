from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "app.config.json"
APP_ENV: str = os.getenv("APP_ENV", "local")


@dataclass(frozen=True)
class AppConfig:
    app_env: str
    port: int
    uvicorn_host: str
    m103_base_url: str
    temporal_uri: str
    temporal_namespace: str
    temporal_enabled: bool
    langfuse_prompt_label: str
    langfuse_extractor_prompt: str
    langfuse_provider_match_prompt: str
    langfuse_customer_facing_prompt: str
    bifrost_proxy: str
    bifrost_auth_scheme: str
    bifrost_model: str
    bifrost_tags: str
    log_level: str


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().upper() in ("ON", "TRUE", "1", "YES")
    return bool(value)


def _build_config() -> AppConfig:
    raw: dict = {}
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH, encoding="utf-8") as f:
            raw = json.load(f)

    merged: dict = {
        **raw.get("defaults", {}),
        **raw.get("environments", {}).get(APP_ENV, {}),
    }

    def _get(config_key: str, env_var: str | None = None, default: object = None) -> object:
        if env_var is not None:
            val = os.getenv(env_var)
            if val is not None:
                return val
        return merged.get(config_key, default)

    return AppConfig(
        app_env=APP_ENV,
        port=int(_get("port", "UVICORN_PORT", 5000)),
        uvicorn_host=str(_get("uvicornHost", "UVICORN_HOST", "127.0.0.1")),
        m103_base_url=str(_get("m103BaseUrl", "M103_BASE_URL", "http://127.0.0.1:8001")).rstrip("/"),
        temporal_uri=str(_get("temporalUri", "TEMPORAL_URI", "localhost:7233")),
        temporal_namespace=str(_get("temporalNamespace", "TEMPORAL_NAMESPACE", "default")),
        temporal_enabled=_parse_bool(_get("temporalEnabled", "TEMPORAL_STATE", False)),
        langfuse_prompt_label=str(_get("langfusePromptLabel", "LANGFUSE_PROMPT_LABEL", "production")),
        langfuse_extractor_prompt=str(_get("langfuseExtractorPrompt", "LANGFUSE_EXTRACTOR_PROMPT", "autologinQA/identifier_extractor")),
        langfuse_provider_match_prompt=str(_get("langfuseProviderMatchPrompt", "LANGFUSE_PROVIDER_MATCH_PROMPT", "autologinQA/service_matcher")),
        langfuse_customer_facing_prompt=str(_get("langfuseCustomerFacingPrompt", "LANGFUSE_CUSTOMER_FACING_PROMPT", "autologinQA/customer_facing_classifier")),
        bifrost_proxy=str(_get("bifrostProxy", "BIFROST_PROXY", "")).rstrip("/"),
        bifrost_auth_scheme=str(_get("bifrostAuthScheme", "BIFROST_AUTH_SCHEME", "Basic")),
        bifrost_model=str(_get("bifrostModel", "BIFROST_MODEL", "vertex/gemini-2.5-flash")),
        bifrost_tags=str(_get("bifrostTags", "BIFROST_TAGS", "")),
        log_level=str(_get("logLevel", "LOG_LEVEL", "INFO")).upper(),
    )


config: AppConfig = _build_config()
