from __future__ import annotations

import asyncio
import logging
import uuid

# Cap concurrent Playwright browser instances across the whole process.
# Each browser uses ~150-300 MB RAM; 5 concurrent is safe on a 2-4 GB container.
_URL_CHECK_SEMAPHORE = asyncio.Semaphore(5)

from fastapi import BackgroundTasks, HTTPException

from src.db import (
    create_activity_run,
    fetch_rows,
    finalize_activity_run,
    start_activity_run,
    upsert_run_item_result,
)
from src.models.request_models import BatchCheckRequest, CheckRequest
from src.models.response_models import (
    AsyncBatchResponse,
    BatchCheckResponse,
    CbLinkResult,
    ReturnResponse,
    RowResult,
)
from src.services.duplicate_service import (
    detect_duplicates,
    resolve_parent_child,
    resolve_same_name_duplicates,
)
from src.services.m103_ingest import ingest_to_m103
from src.services.verification_service import verify_row, verify_url

logger = logging.getLogger("autologin.verification_controller")


def _temporal_enabled() -> bool:
    """Read kill switch at call time so tests can toggle it without restart."""
    from temporal.config.settings import TEMPORAL_ENABLED  # noqa: PLC0415
    return TEMPORAL_ENABLED


class VerificationController:
    async def handle_request(self, payload: CheckRequest) -> ReturnResponse:
        """Single-URL check (legacy endpoint — always runs inline, no Temporal)."""
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
        """Batch check — routes to Temporal when TEMPORAL_STATE=ON, inline otherwise."""
        try:
            if _temporal_enabled():
                return await self._handle_batch_temporal(payload)
            return await self._handle_batch_inline(payload)
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("Unexpected batch controller error: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=500,
                detail="Internal server error while processing /check/batch request.",
            ) from exc

    # ------------------------------------------------------------------
    # Temporal path
    # ------------------------------------------------------------------

    async def _handle_batch_temporal(self, payload: BatchCheckRequest) -> BatchCheckResponse:
        """Submit one BatchVerificationWorkflow per cb_link_id and await results."""
        from temporal.client.client import get_temporal_client  # noqa: PLC0415
        from temporal.config.settings import (  # noqa: PLC0415
            BATCH_TASK_QUEUE,
            BATCH_WORKFLOW_TIMEOUT,
        )
        from temporal.workflows.batch_workflow import (  # noqa: PLC0415
            BatchVerificationInput,
            BatchVerificationOutput,
            BatchVerificationWorkflow,
        )
        from temporal.workflows.verification_workflow import VerificationInput  # noqa: PLC0415

        client = await get_temporal_client()

        async def _run_one_cb_link(cb_link_id: str) -> CbLinkResult:
            rows = await fetch_rows(cb_link_id, include_inactive=payload.include_inactive)
            rows = detect_duplicates(rows)

            # Build VerificationInput objects only for non-duplicate rows.
            verification_inputs = [
                VerificationInput(
                    url=row.get("login_url") or "",
                    provider=row.get("cb_link_id") or "",
                    service_name=row.get("login_service") or "",
                    login_type="direct",
                    country="india",
                    cb_link_id=str(row.get("cb_link_id") or ""),
                )
                for row in rows
                if not row["is_duplicate"]
            ]

            workflow_id = f"batch-{cb_link_id}-{uuid.uuid4().hex[:8]}"
            output: BatchVerificationOutput = await client.execute_workflow(
                BatchVerificationWorkflow.run,
                BatchVerificationInput(rows=verification_inputs, cb_link_id=cb_link_id),
                id=workflow_id,
                task_queue=BATCH_TASK_QUEUE,
                execution_timeout=BATCH_WORKFLOW_TIMEOUT,
            )

            # Build a url→result lookup from the workflow output.
            result_by_url: dict[str, object] = {
                r.url: r.result for r in output.rows
            }

            row_dicts: list[dict] = []
            for row in rows:
                url = row.get("login_url") or ""
                t_result = result_by_url.get(url)
                row_result = self._temporal_result_to_row_result(row, t_result)
                d = row_result.model_dump()
                d["id"] = str(row.get("id") or "")
                d["sorting_order"] = row.get("sorting_order")
                row_dicts.append(d)

            row_dicts = resolve_same_name_duplicates(row_dicts)
            row_dicts = resolve_parent_child(row_dicts)

            final_rows = []
            for d in row_dicts:
                d.pop("id", None)
                d.pop("sorting_order", None)
                final_rows.append(RowResult(**d))

            return CbLinkResult(
                cb_link_id=cb_link_id,
                total_rows=len(rows),
                rows=final_rows,
            )

        cb_link_results = await asyncio.gather(
            *[_run_one_cb_link(cb_id) for cb_id in payload.cb_link_ids]
        )
        return BatchCheckResponse(results=list(cb_link_results))

    @staticmethod
    def _temporal_result_to_row_result(row: dict, t_result: object | None) -> RowResult:
        """Map a VerificationResult (or None) + DB row dict → RowResult."""
        return RowResult(
            cb_link_id=str(row["cb_link_id"]),
            service_id=str(row["id"]) if row.get("id") else None,
            login_service=row.get("login_service") or "",
            url=row.get("login_url") or "",
            health_check=t_result.health_check if t_result else None,
            page_match_score=t_result.page_match_score if t_result else None,
            direct_match_score=t_result.direct_match_score if t_result else None,
            display_name_score=None,
            notes=t_result.notes if t_result else None,
            inactive_flagged=t_result.inactive_flagged if t_result else None,
            marked_for_deletion=t_result.marked_for_deletion if t_result else None,
            marked_for_human_review=t_result.marked_for_human_review if t_result else None,
            is_duplicate=row["is_duplicate"],
            duplicate_of_url=row.get("duplicate_of_url"),
            duplicate_of_id=row.get("duplicate_of_id"),
            dedupe_action=row.get("dedupe_action"),
            dedupe_reason=row.get("dedupe_reason"),
            canonical_display_name=row.get("canonical_display_name"),
            reason=t_result.reason if t_result else None,
            status=row.get("status") or "",
        )

    # ------------------------------------------------------------------
    # Inline (original) path — unchanged logic, used when TEMPORAL_STATE=OFF
    # ------------------------------------------------------------------

    async def _handle_batch_inline(self, payload: BatchCheckRequest) -> BatchCheckResponse:
        cb_link_results = await asyncio.gather(
            *[
                self._process_cb_link(cb_link_id, payload.include_inactive)
                for cb_link_id in payload.cb_link_ids
            ]
        )
        return BatchCheckResponse(results=list(cb_link_results))

    async def _process_cb_link(
        self, cb_link_id: str, include_inactive: bool
    ) -> CbLinkResult:
        logger.info("[batch] processing cb_link_id=%s", cb_link_id)

        rows = await fetch_rows(cb_link_id, include_inactive=include_inactive)
        rows = detect_duplicates(rows)  # Phase 1: Cases 1 + 3 (pre-verification)

        row_results = await asyncio.gather(
            *[self._process_row(row) for row in rows]
        )

        row_dicts: list[dict] = []
        for row, result in zip(rows, row_results):
            d = result.model_dump()
            d["id"] = str(row.get("id") or "")
            d["sorting_order"] = row.get("sorting_order")
            row_dicts.append(d)

        row_dicts = resolve_same_name_duplicates(row_dicts)  # Phase 2: Case 2
        row_dicts = resolve_parent_child(row_dicts)          # Phase 3: Case 4 stub

        final_rows = []
        for d in row_dicts:
            d.pop("id", None)
            d.pop("sorting_order", None)
            final_rows.append(RowResult(**d))

        return CbLinkResult(
            cb_link_id=cb_link_id,
            total_rows=len(rows),
            rows=final_rows,
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
                async with _URL_CHECK_SEMAPHORE:
                    verification = await verify_row(row)
            except Exception as exc:
                logger.error(
                    "[batch] verification failed for row %s: %s", row.get("id"), exc
                )

        return RowResult(
            cb_link_id=str(row["cb_link_id"]),
            service_id=str(row["id"]) if row.get("id") else None,
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
            duplicate_of_id=row.get("duplicate_of_id"),
            dedupe_action=row.get("dedupe_action"),
            dedupe_reason=row.get("dedupe_reason"),
            canonical_display_name=row.get("canonical_display_name"),
            reason=verification.reason if verification else None,
            status=row.get("status") or "",
        )

    # ------------------------------------------------------------------
    # Async path — returns run_id immediately, verification runs in background
    # ------------------------------------------------------------------

    async def handle_batch_async(
        self,
        payload: BatchCheckRequest,
        background_tasks: BackgroundTasks,
    ) -> AsyncBatchResponse:
        """Create an activity_run record and queue background verification."""
        try:
            run_id = await create_activity_run(
                cb_link_ids=payload.cb_link_ids,
                triggered_by=payload.triggered_by or "m114",
            )
        except Exception as exc:
            logger.error("Failed to create activity_run: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=500,
                detail="Could not initialise verification run in database.",
            ) from exc

        background_tasks.add_task(self._run_verification_background, run_id, payload)
        return AsyncBatchResponse(run_id=run_id, total_links=len(payload.cb_link_ids))

    async def _run_verification_background(
        self,
        run_id: str,
        payload: BatchCheckRequest,
    ) -> None:
        """Background task: run full inline verification and persist results."""
        try:
            await start_activity_run(run_id)
            results = await self._handle_batch_inline(payload)
        except Exception as exc:
            logger.error(
                "[bg] verification run %s failed during execution: %s",
                run_id, exc, exc_info=True,
            )
            try:
                await finalize_activity_run(run_id, 0, len(payload.cb_link_ids))
            except Exception:
                pass
            return

        success, failed = 0, 0
        for cb_result in results.results:
            try:
                metrics = cb_result.model_dump()
                await upsert_run_item_result(run_id, cb_result.cb_link_id, metrics, succeeded=True)
                success += 1
            except Exception as exc:
                logger.error(
                    "[bg] failed to persist result for %s: %s",
                    cb_result.cb_link_id, exc, exc_info=True,
                )
                try:
                    await upsert_run_item_result(run_id, cb_result.cb_link_id, {}, succeeded=False)
                except Exception:
                    pass
                failed += 1

        try:
            await finalize_activity_run(run_id, success, failed)
        except Exception as exc:
            logger.error("[bg] failed to finalize run %s: %s", run_id, exc, exc_info=True)

        await ingest_to_m103(run_id, results, payload.triggered_by)
