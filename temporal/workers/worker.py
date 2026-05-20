"""Worker entry point.

Start this process alongside the FastAPI server to enable Temporal execution:

    python -m temporal.workers.worker

Two workers run in the same process:
  - verification-queue    → VerificationWorkflow + health/llm/country activities
  - batch-verification-queue → BatchVerificationWorkflow, VerificationQueueWorkflow
                               + fetch_rows, start_run, persist_item, finalize_run activities

On startup, ensures the singleton VerificationQueueWorkflow is running.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import sys
from pathlib import Path

# Ensure the repo root is on sys.path when run as a module from any CWD.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# Playwright on Windows needs SelectorEventLoop, same as app.py.
if sys.platform == "win32":
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from temporalio.client import Client
from temporalio.common import WorkflowIDReusePolicy
from temporalio.exceptions import WorkflowAlreadyStartedError
from temporalio.worker import Worker

from temporal.activities.country_activity import country_match_activity
from temporal.activities.fetch_activity import fetch_rows_activity
from temporal.activities.persist_activity import (
    finalize_run_activity,
    persist_item_activity,
    start_run_activity,
)
from temporal.activities.health_activity import health_check_activity
from temporal.activities.llm_activities import audience_classify_activity, full_match_activity
from temporal.client.client import get_temporal_client
from temporal.config.settings import (
    BATCH_TASK_QUEUE,
    QUEUE_TASK_QUEUE,
    QUEUE_WORKFLOW_ID,
    VERIFICATION_TASK_QUEUE,
)
from temporal.workflows.batch_workflow import BatchVerificationWorkflow
from temporal.workflows.queue_workflow import VerificationQueueInput, VerificationQueueWorkflow
from temporal.workflows.verification_workflow import VerificationWorkflow

logger = logging.getLogger("temporal.worker")
logging.basicConfig(level=logging.INFO)


async def ensure_queue_workflow(client: Client) -> None:
    """Start the singleton VerificationQueueWorkflow if it isn't already running.

    Uses ALLOW_DUPLICATE so a fresh execution is started whenever the previous
    one was terminated or completed.  WorkflowAlreadyStartedError is only raised
    for a currently-open (running) execution, which is the happy path.
    """
    try:
        await client.start_workflow(
            VerificationQueueWorkflow.run,
            VerificationQueueInput(),
            id=QUEUE_WORKFLOW_ID,
            task_queue=QUEUE_TASK_QUEUE,
            id_reuse_policy=WorkflowIDReusePolicy.ALLOW_DUPLICATE,
        )
        logger.info("Started singleton queue workflow id=%s", QUEUE_WORKFLOW_ID)
    except WorkflowAlreadyStartedError:
        logger.info("Queue workflow already running id=%s", QUEUE_WORKFLOW_ID)


async def start_worker() -> None:
    """Start both Temporal workers and the singleton queue workflow.

    Safe to call as an asyncio background task from the FastAPI lifespan
    (fire-and-forget).  Also runnable standalone via `python -m temporal.workers.worker`.
    """
    client = await get_temporal_client()

    await ensure_queue_workflow(client)

    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as activity_executor:
        # Worker 1: single-URL verification workflows + their activities.
        worker_verification = Worker(
            client,
            task_queue=VERIFICATION_TASK_QUEUE,
            workflows=[VerificationWorkflow],
            activities=[
                health_check_activity,
                audience_classify_activity,
                full_match_activity,
                country_match_activity,
            ],
            activity_executor=activity_executor,
        )

        # Worker 2: batch + queue workflows + persistence activities.
        worker_batch = Worker(
            client,
            task_queue=BATCH_TASK_QUEUE,
            workflows=[BatchVerificationWorkflow, VerificationQueueWorkflow],
            activities=[
                fetch_rows_activity,
                start_run_activity,
                persist_item_activity,
                finalize_run_activity,
            ],
            activity_executor=activity_executor,
        )

        logger.info(
            "Workers started — queues: %s, %s",
            VERIFICATION_TASK_QUEUE,
            BATCH_TASK_QUEUE,
        )
        await asyncio.gather(worker_verification.run(), worker_batch.run())


if __name__ == "__main__":
    asyncio.run(start_worker())
