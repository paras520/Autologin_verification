"""Workflow: singleton verification queue.

A long-running workflow that accepts enqueue_signal calls and processes
cb_link_ids sequentially.  Follows the same queue-workflow pattern used in
m103 (displaySort) and m112 (discovery).

Lifecycle:
  - Started once by the worker on boot (workflowId = QUEUE_WORKFLOW_ID).
  - Runs forever, sleeping via wait_condition until signals arrive.
  - After QUEUE_ITEM_THRESHOLD items are processed, calls continue_as_new
    to keep Temporal history bounded while preserving remaining queue state.

Signals:
  enqueue_signal  — add a cb_link_id to the queue (deduplicates silently).

Queries:
  get_status_query — returns live queue state for monitoring / API endpoints.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from temporal.activities.fetch_activity import FetchRowsInput, fetch_rows_activity
    from temporal.activities.persist_activity import (
        FinalizeRunInput,
        PersistItemInput,
        finalize_run_activity,
        persist_item_activity,
        start_run_activity,
    )
    from temporal.config.settings import (
        BATCH_TASK_QUEUE,
        BATCH_WORKFLOW_TIMEOUT,
        FETCH_ACTIVITY_TIMEOUT,
        FETCH_RETRY_POLICY,
        QUEUE_ITEM_THRESHOLD,
    )
    from temporal.workflows.batch_workflow import (
        BatchVerificationInput,
        BatchVerificationWorkflow,
    )
    from temporal.workflows.verification_workflow import VerificationInput

_PERSIST_RETRY = RetryPolicy(initial_interval=timedelta(seconds=2), maximum_attempts=3)
_PERSIST_TIMEOUT = timedelta(seconds=60)
_FINALIZE_TIMEOUT = timedelta(seconds=120)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class QueueItem:
    cb_link_id: str
    include_inactive: bool = False
    queued_at: str = ""
    run_id: str | None = None
    run_item_id: str | None = None
    triggered_by: str = "m114"
    total_in_run: int = 1


@dataclass
class VerificationQueueInput:
    """Passed to continue_as_new to carry forward surviving state."""
    initial_queue: list[QueueItem] = field(default_factory=list)
    initial_completed: list[dict] = field(default_factory=list)
    initial_failed: list[dict] = field(default_factory=list)
    initial_run_progress: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------

@workflow.defn
class VerificationQueueWorkflow:
    """Singleton queue — one instance per Temporal namespace."""

    def __init__(self) -> None:
        self._queue: list[QueueItem] = []
        self._processing: QueueItem | None = None
        self._completed: list[dict] = []
        self._failed: list[dict] = []
        self._items_processed: int = 0
        # Tracks per-run progress so we know when to finalize + ingest.
        # { run_id: {total, processed, success, failed, results: [CbLinkResult dict]} }
        self._run_progress: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Signal
    # ------------------------------------------------------------------

    @workflow.signal
    async def enqueue_signal(self, item: QueueItem) -> None:
        """Add cb_link_id to the queue; silently drops duplicates."""
        if self._processing and self._processing.cb_link_id == item.cb_link_id:
            workflow.logger.info(
                "Skipping duplicate cb_link_id=%s (already processing)", item.cb_link_id
            )
            return
        if any(q.cb_link_id == item.cb_link_id for q in self._queue):
            workflow.logger.info(
                "Skipping duplicate cb_link_id=%s (already queued)", item.cb_link_id
            )
            return
        self._queue.append(item)
        workflow.logger.info("Enqueued cb_link_id=%s (queue depth=%d)", item.cb_link_id, len(self._queue))

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    @workflow.query
    def get_status_query(self) -> dict:
        return {
            "processing": self._processing.cb_link_id if self._processing else None,
            "queued": [q.cb_link_id for q in self._queue],
            "completed_count": len(self._completed),
            "failed_count": len(self._failed),
            "items_processed": self._items_processed,
        }

    # ------------------------------------------------------------------
    # Run loop
    # ------------------------------------------------------------------

    @workflow.run
    async def run(self, inp: VerificationQueueInput | None = None) -> None:
        if inp:
            self._queue = list(inp.initial_queue or [])
            self._completed = list(inp.initial_completed or [])
            self._failed = list(inp.initial_failed or [])
            self._run_progress = dict(inp.initial_run_progress or {})

        while True:
            if self._items_processed >= QUEUE_ITEM_THRESHOLD:
                workflow.continue_as_new(
                    VerificationQueueInput(
                        initial_queue=self._queue,
                        initial_completed=self._completed[-50:],
                        initial_failed=self._failed[-50:],
                        initial_run_progress=self._run_progress,
                    )
                )

            await workflow.wait_condition(lambda: len(self._queue) > 0)

            item = self._queue.pop(0)
            self._processing = item
            workflow.logger.info("Processing cb_link_id=%s", item.cb_link_id)

            try:
                await self._process_item(item)
            except Exception as exc:
                workflow.logger.error(
                    "Queue item failed cb_link_id=%s: %s", item.cb_link_id, exc
                )
                self._failed.append({"cb_link_id": item.cb_link_id, "error": str(exc)})
                await self._record_run_failure(item)
            finally:
                self._processing = None
                self._items_processed += 1

    # ------------------------------------------------------------------
    # Item processing
    # ------------------------------------------------------------------

    async def _process_item(self, item: QueueItem) -> None:
        # workflow.patched() guards the new start_run_activity call so existing
        # workflow histories (which didn't have this step) replay without a
        # nondeterminism error.
        if item.run_id and workflow.patched("start-run-v1"):
            await workflow.execute_activity(
                start_run_activity,
                item.run_id,
                start_to_close_timeout=_PERSIST_TIMEOUT,
                retry_policy=_PERSIST_RETRY,
            )

        # Fetch rows + deduplication.
        rows: list[dict] = await workflow.execute_activity(
            fetch_rows_activity,
            FetchRowsInput(
                cb_link_id=item.cb_link_id,
                include_inactive=item.include_inactive,
            ),
            start_to_close_timeout=FETCH_ACTIVITY_TIMEOUT,
            retry_policy=FETCH_RETRY_POLICY,
        )

        # Build verification inputs — pure transform, safe in workflow.
        verification_inputs = [
            VerificationInput(
                url=row.get("login_url") or "",
                provider=row.get("display_name") or row.get("login_service") or row.get("cb_link_id") or "",
                service_name=row.get("login_service") or "",
                login_type="direct",
                country="india",
                cb_link_id=str(row.get("cb_link_id") or ""),
            )
            for row in rows
            if not row.get("is_duplicate")
        ]

        if not verification_inputs:
            workflow.logger.info("No non-duplicate rows for cb_link_id=%s", item.cb_link_id)
            empty_result = {"cb_link_id": item.cb_link_id, "total_rows": len(rows), "rows": []}
            self._completed.append({"cb_link_id": item.cb_link_id, "total": 0, "succeeded": 0, "failed": 0})
            await self._record_run_success(item, empty_result)
            return

        # Run child BatchVerificationWorkflow.
        child_id = f"batch-{item.cb_link_id}-{str(workflow.uuid4()).replace('-', '')[:8]}"
        output = await workflow.execute_child_workflow(
            BatchVerificationWorkflow.run,
            BatchVerificationInput(rows=verification_inputs, cb_link_id=item.cb_link_id),
            id=child_id,
            task_queue=BATCH_TASK_QUEUE,
            execution_timeout=BATCH_WORKFLOW_TIMEOUT,
            retry_policy=RetryPolicy(maximum_attempts=1),
        )

        # Serialize output rows to plain dicts — safe cross-activity boundary.
        result_rows = []
        for r in output.rows:
            entry: dict = {"url": r.url, "error": r.error, "result": None}
            if r.result:
                entry["result"] = {
                    "url": r.result.url,
                    "inactive_flagged": r.result.inactive_flagged,
                    "reason": r.result.reason,
                    "health_check": r.result.health_check,
                    "page_match_score": r.result.page_match_score,
                    "direct_match_score": r.result.direct_match_score,
                    "notes": r.result.notes,
                    "updated_name": r.result.updated_name,
                    "marked_for_human_review": r.result.marked_for_human_review,
                    "marked_for_deletion": r.result.marked_for_deletion,
                    "errors": r.result.errors,
                    "time": r.result.time,
                }
            result_rows.append(entry)

        # Persist this item's results to DB and get back the CbLinkResult dict.
        cb_link_result: dict = {"cb_link_id": item.cb_link_id, "total_rows": len(rows), "rows": []}
        if item.run_id:
            cb_link_result = await workflow.execute_activity(
                persist_item_activity,
                PersistItemInput(
                    run_id=item.run_id,
                    cb_link_id=item.cb_link_id,
                    rows=rows,
                    result_rows=result_rows,
                ),
                start_to_close_timeout=_PERSIST_TIMEOUT,
                retry_policy=_PERSIST_RETRY,
            )

        self._completed.append({
            "cb_link_id": item.cb_link_id,
            "total": output.total,
            "succeeded": output.succeeded,
            "failed": output.failed,
        })

        await self._record_run_success(item, cb_link_result, output.succeeded, output.failed)

        workflow.logger.info(
            "Completed cb_link_id=%s total=%d succeeded=%d failed=%d",
            item.cb_link_id, output.total, output.succeeded, output.failed,
        )

    # ------------------------------------------------------------------
    # Run-level tracking helpers
    # ------------------------------------------------------------------

    async def _record_run_success(
        self,
        item: QueueItem,
        cb_link_result: dict,
        item_succeeded: int = 0,
        item_failed: int = 0,
    ) -> None:
        if not item.run_id:
            return
        prog = self._ensure_run_progress(item)
        prog["processed"] += 1
        prog["success"] += (1 if item_failed == 0 else 0)
        prog["failed"] += (1 if item_failed > 0 else 0)
        prog["results"].append(cb_link_result)
        await self._maybe_finalize_run(item.run_id)

    async def _record_run_failure(self, item: QueueItem) -> None:
        if not item.run_id:
            return
        prog = self._ensure_run_progress(item)
        prog["processed"] += 1
        prog["failed"] += 1
        await self._maybe_finalize_run(item.run_id)

    def _ensure_run_progress(self, item: QueueItem) -> dict:
        if item.run_id not in self._run_progress:
            self._run_progress[item.run_id] = {
                "total": item.total_in_run,
                "processed": 0,
                "success": 0,
                "failed": 0,
                "triggered_by": item.triggered_by,
                "results": [],
            }
        return self._run_progress[item.run_id]

    async def _maybe_finalize_run(self, run_id: str) -> None:
        prog = self._run_progress.get(run_id)
        if not prog:
            return
        if prog["processed"] < prog["total"]:
            return

        # All items for this run are done — finalize + ingest to m103.
        await workflow.execute_activity(
            finalize_run_activity,
            FinalizeRunInput(
                run_id=run_id,
                triggered_by=prog["triggered_by"],
                success_items=prog["success"],
                failed_items=prog["failed"],
                all_results=prog["results"],
            ),
            start_to_close_timeout=_FINALIZE_TIMEOUT,
            retry_policy=_PERSIST_RETRY,
        )
        del self._run_progress[run_id]
        workflow.logger.info("Finalized and ingested run_id=%s", run_id)
