"""Activity: fetch and deduplicate rows for a cb_link_id.

Used by VerificationQueueWorkflow so the queue workflow can do DB I/O without
violating Temporal's determinism requirements.
"""
from __future__ import annotations

from dataclasses import dataclass

from temporalio import activity

from src.db import fetch_rows
from src.services.duplicate_service import detect_duplicates


@dataclass
class FetchRowsInput:
    cb_link_id: str
    include_inactive: bool = False


@activity.defn
async def fetch_rows_activity(inp: FetchRowsInput) -> list[dict]:
    """Fetch login_service rows for a cb_link_id and mark duplicates."""
    rows = await fetch_rows(inp.cb_link_id, include_inactive=inp.include_inactive)
    return detect_duplicates(rows)
