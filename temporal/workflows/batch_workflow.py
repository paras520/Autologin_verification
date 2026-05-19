"""Workflow: batch URL verification.

Spawns one child VerificationWorkflow per URL.  Uses asyncio.gather inside
the workflow so all child workflows execute in parallel.  Partial failures
are collected (not raised) so one bad URL cannot block the rest.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from temporal.config.settings import (
        CHILD_WORKFLOW_EXECUTION_TIMEOUT,
        VERIFICATION_TASK_QUEUE,
    )
    from temporal.workflows.verification_workflow import (
        VerificationInput,
        VerificationResult,
        VerificationWorkflow,
    )


@dataclass
class BatchVerificationInput:
    """One entry per DB row that should be verified."""
    rows: list[VerificationInput] = field(default_factory=list)
    cb_link_id: str = ""


@dataclass
class BatchRowResult:
    cb_link_id: str
    url: str
    result: VerificationResult | None
    error: str | None = None


@dataclass
class BatchVerificationOutput:
    cb_link_id: str
    total: int
    succeeded: int
    failed: int
    rows: list[BatchRowResult] = field(default_factory=list)


@workflow.defn
class BatchVerificationWorkflow:
    """Durable batch verification — one child workflow per URL."""

    @workflow.run
    async def run(self, inp: BatchVerificationInput) -> BatchVerificationOutput:
        tasks = [self._verify_one(row) for row in inp.rows]
        row_results: list[BatchRowResult] = await asyncio.gather(*tasks)

        succeeded = sum(1 for r in row_results if r.error is None)
        failed = len(row_results) - succeeded

        return BatchVerificationOutput(
            cb_link_id=inp.cb_link_id,
            total=len(inp.rows),
            succeeded=succeeded,
            failed=failed,
            rows=row_results,
        )

    async def _verify_one(self, row: VerificationInput) -> BatchRowResult:
        """Start a child VerificationWorkflow and return its result (or error)."""
        try:
            result: VerificationResult = await workflow.execute_child_workflow(
                VerificationWorkflow.run,
                row,
                id=f"verify-{row.cb_link_id}-{_url_slug(row.url)}",
                task_queue=VERIFICATION_TASK_QUEUE,
                execution_timeout=CHILD_WORKFLOW_EXECUTION_TIMEOUT,
                # Child workflows inherit the parent's retry semantics via
                # activity-level retries; don't add workflow-level retries here.
                retry_policy=RetryPolicy(maximum_attempts=1),
            )
            return BatchRowResult(
                cb_link_id=row.cb_link_id,
                url=row.url,
                result=result,
            )
        except Exception as exc:
            workflow.logger.warning(
                "Child workflow failed for cb_link_id=%s url=%s: %s",
                row.cb_link_id,
                row.url,
                exc,
            )
            return BatchRowResult(
                cb_link_id=row.cb_link_id,
                url=row.url,
                result=None,
                error=str(exc),
            )


def _url_slug(url: str, max_len: int = 40) -> str:
    """Create a short, workflow-ID-safe slug from a URL."""
    slug = url.replace("https://", "").replace("http://", "").replace("/", "-")
    return slug[:max_len]
