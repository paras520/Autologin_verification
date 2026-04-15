from __future__ import annotations

import asyncio
import logging

from fastapi import HTTPException

from src.db import fetch_rows
from src.models.request_models import BatchCheckRequest, CheckRequest
from src.models.response_models import (
    BatchCheckResponse,
    CbLinkResult,
    RowResult,
)
from src.services.duplicate_service import detect_duplicates
from src.services.verification_service import verify_row, verify_url

logger = logging.getLogger("autologin.verification_controller")


class VerificationController:
    async def handle_request(self, payload: CheckRequest) -> ReturnResponse:
        """Single-URL check (legacy endpoint kept for backwards compat)."""
        try:
            return await verify_url(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("Unexpected controller error: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=500,
                detail="Internal server error while processing /check request.",
            ) from exc

    async def handle_batch(self, payload: BatchCheckRequest) -> BatchCheckResponse:
        """Batch check: for each cb_link_id, fetch all rows, dupe-check, then verify each."""
        try:
            cb_link_results = await asyncio.gather(
                *[
                    self._process_cb_link(cb_link_id, payload.include_inactive)
                    for cb_link_id in payload.cb_link_ids
                ]
            )
            return BatchCheckResponse(results=list(cb_link_results))
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("Unexpected batch controller error: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=500,
                detail="Internal server error while processing /check/batch request.",
            ) from exc

    async def _process_cb_link(
        self, cb_link_id: str, include_inactive: bool
    ) -> CbLinkResult:
        logger.info("[batch] processing cb_link_id=%s", cb_link_id)

        rows = await fetch_rows(cb_link_id, include_inactive=include_inactive)
        rows = detect_duplicates(rows)

        row_results = await asyncio.gather(
            *[self._process_row(row) for row in rows]
        )

        return CbLinkResult(
            cb_link_id=cb_link_id,
            total_rows=len(rows),
            rows=list(row_results),
        )

    async def _process_row(self, row: dict) -> RowResult:
        verification: ReturnResponse | None = None

        if row["is_duplicate"]:
            logger.info(
                "[batch] skipping verification for duplicate row %s (duplicate of %s)",
                row.get("id"), row.get("duplicate_of_id"),
            )
        else:
            try:
                verification = await verify_row(row)
            except Exception as exc:
                logger.error(
                    "[batch] verification failed for row %s: %s", row.get("id"), exc
                )

        return RowResult(
            cb_link_id=str(row["cb_link_id"]),
            login_service=row.get("login_service") or "",
            url=row.get("login_url") or "",
            health_check=verification.health_check if verification else None,
            page_match_score=verification.page_match_score if verification else None,
            direct_match_score=verification.direct_match_score if verification else None,
            display_name_score=None,
            notes=verification.notes if verification else None,
            inactive_flagged=verification.inactive_flagged if verification else None,
            marked_for_deletion=verification.marked_for_deletion if verification else None,
            marked_for_human_review=verification.marked_for_human_review if verification else None,
            is_duplicate=row["is_duplicate"],
            duplicate_of_url=row.get("duplicate_of_url"),
            reason=verification.reason if verification else None,
            status=row.get("status") or "",
        )
