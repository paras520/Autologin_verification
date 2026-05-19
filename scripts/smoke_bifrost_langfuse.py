"""Quick end-to-end smoke test for the Bifrost + Langfuse integration.

Makes one real chat completion against Bifrost (with Langfuse tracing) using
the env vars from .env. Use this after configuring credentials to verify the
integration is wired up correctly.

Run:
    python scripts/smoke_bifrost_langfuse.py
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

# Make the repo root importable when invoked from anywhere
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from langfuse_helper import build_messages, call_litellm, parse_response  # noqa: E402


async def main() -> None:
    messages = build_messages(
        system_prompt="You are a JSON-only assistant. Reply with valid JSON only.",
        user_prompt='Reply with the JSON {"ok": true, "msg": "hello from bifrost"}.',
    )
    response = await call_litellm(
        config={"temperature": 0.0, "max_tokens": 100},
        messages=messages,
        session_id="bifrost-smoke-test",
        api_endpoint="/scripts/smoke_bifrost_langfuse",
        tag_suffix="smoke",
        extra_tags=["integration-test"],
    )
    parsed = parse_response(response)
    print("PARSED:", json.dumps(parsed, indent=2) if isinstance(parsed, dict) else parsed)
    print("MODEL:", response.model)
    print("USAGE:", {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    })


if __name__ == "__main__":
    asyncio.run(main())
