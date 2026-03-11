from typing import Any

from pydantic import BaseModel


class LLMDecision(BaseModel):
    inactive_flagged: bool
    reason: str | None
    raw_output: dict[str, Any] | str | None
    error: str | None
    session_id: str | None
    prompt_path: str
