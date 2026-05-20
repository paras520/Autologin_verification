"""Activities: persist verification results to DB and ingest to m103.

Three activities cover the full lifecycle of a queue-workflow run item:
  start_run_activity        — mark activity_run as 'running'
  persist_item_activity     — map + save one cb_link_id's results to DB
  finalize_run_activity     — set final run status + POST to m103 QA ingest
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from temporalio import activity

from src.db import (
    finalize_activity_run,
    start_activity_run,
    upsert_run_item_result,
)
from src.models.response_models import BatchCheckResponse, CbLinkResult, RowResult
from src.services.duplicate_service import resolve_parent_child, resolve_same_name_duplicates
from src.services.m103_ingest import ingest_to_m103

logger = logging.getLogger("autologin.persist_activity")


# ---------------------------------------------------------------------------
# Input types
# ---------------------------------------------------------------------------

@dataclass
class PersistItemInput:
    run_id: str
    cb_link_id: str
    rows: list[dict]           # raw DB rows (from fetch_rows_activity)
    result_rows: list[dict]    # [{url, result: {health_check, ...} | None, error}]


@dataclass
class FinalizeRunInput:
    run_id: str
    triggered_by: str
    success_items: int
    failed_items: int
    all_results: list[dict]    # list of CbLinkResult dicts from persist_item_activity


# ---------------------------------------------------------------------------
# Activities
# ---------------------------------------------------------------------------

@activity.defn
async def start_run_activity(run_id: str) -> None:
    """Transition the activity_run record to status=running."""
    await start_activity_run(run_id)


@activity.defn
async def persist_item_activity(inp: PersistItemInput) -> dict:
    """Map verification output + raw DB rows → CbLinkResult, persist to DB.

    Returns the CbLinkResult as a plain dict so the queue workflow can
    accumulate results for the final m103 ingest.
    """
    result_by_url = {r["url"]: r.get("result") for r in inp.result_rows}

    row_dicts: list[dict] = []
    for row in inp.rows:
        url = row.get("login_url") or ""
        t = result_by_url.get(url)  # VerificationResult dict or None

        d: dict = {
            "cb_link_id": str(row["cb_link_id"]),
            "service_id": str(row["id"]) if row.get("id") else None,
            "login_service": row.get("login_service") or "",
            "url": url,
            "health_check": t["health_check"] if t else None,
            "page_match_score": t["page_match_score"] if t else None,
            "direct_match_score": t["direct_match_score"] if t else None,
            "display_name_score": None,
            "notes": t["notes"] if t else None,
            "inactive_flagged": t["inactive_flagged"] if t else None,
            "marked_for_deletion": t["marked_for_deletion"] if t else None,
            "marked_for_human_review": t["marked_for_human_review"] if t else None,
            "is_duplicate": row["is_duplicate"],
            "duplicate_of_url": row.get("duplicate_of_url"),
            "duplicate_of_id": row.get("duplicate_of_id"),
            "dedupe_action": row.get("dedupe_action"),
            "dedupe_reason": row.get("dedupe_reason"),
            "canonical_display_name": row.get("canonical_display_name"),
            "reason": t["reason"] if t else None,
            "status": row.get("status") or "",
            # temp fields for duplicate resolution (stripped before final model)
            "id": str(row.get("id") or ""),
            "sorting_order": row.get("sorting_order"),
        }
        row_dicts.append(d)

    row_dicts = resolve_same_name_duplicates(row_dicts)
    row_dicts = resolve_parent_child(row_dicts)

    final_rows: list[RowResult] = []
    for d in row_dicts:
        d.pop("id", None)
        d.pop("sorting_order", None)
        final_rows.append(RowResult(**d))

    cb_link_result = CbLinkResult(
        cb_link_id=inp.cb_link_id,
        total_rows=len(inp.rows),
        rows=final_rows,
    )
    metrics = cb_link_result.model_dump()

    succeeded = any(r.health_check is not None for r in final_rows)
    await upsert_run_item_result(inp.run_id, inp.cb_link_id, metrics, succeeded=succeeded)

    logger.info("[persist] saved result for cb_link_id=%s run_id=%s", inp.cb_link_id, inp.run_id)
    return metrics


@activity.defn
async def finalize_run_activity(inp: FinalizeRunInput) -> None:
    """Set final run status and POST all results to m103 QA ingest endpoint."""
    await finalize_activity_run(inp.run_id, inp.success_items, inp.failed_items)

    cb_results: list[CbLinkResult] = []
    for r in inp.all_results:
        rows = [RowResult(**row) for row in r.get("rows", [])]
        cb_results.append(CbLinkResult(
            cb_link_id=r["cb_link_id"],
            total_rows=r["total_rows"],
            rows=rows,
        ))

    response = BatchCheckResponse(results=cb_results)
    await ingest_to_m103(inp.run_id, response, inp.triggered_by)
    logger.info("[persist] finalized run_id=%s and ingested to m103", inp.run_id)
