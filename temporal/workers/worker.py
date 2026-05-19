"""Worker entry point.

Start this process alongside the FastAPI server to enable Temporal execution:

    python -m temporal.workers.worker

The worker polls both the verification queue (single-URL workflows) and the
batch queue (batch workflows) so a single process handles all work types.
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

from temporalio.worker import Worker

from temporal.activities.country_activity import country_match_activity
from temporal.activities.health_activity import health_check_activity
from temporal.activities.llm_activities import audience_classify_activity, full_match_activity
from temporal.client.client import get_temporal_client
from temporal.config.settings import BATCH_TASK_QUEUE, VERIFICATION_TASK_QUEUE
from temporal.workflows.batch_workflow import BatchVerificationWorkflow
from temporal.workflows.verification_workflow import VerificationWorkflow

logger = logging.getLogger("temporal.worker")
logging.basicConfig(level=logging.INFO)


async def main() -> None:
    client = await get_temporal_client()

    # Sync activities (country_match_activity is a plain def) need a thread executor.
    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as activity_executor:
        worker = Worker(
            client,
            # Poll both queues from the same process.
            task_queue=VERIFICATION_TASK_QUEUE,
            workflows=[VerificationWorkflow, BatchVerificationWorkflow],
            activities=[
                health_check_activity,
                audience_classify_activity,
                full_match_activity,
                country_match_activity,
            ],
            activity_executor=activity_executor,
        )

        logger.info(
            "Worker started — queues: %s, %s",
            VERIFICATION_TASK_QUEUE,
            BATCH_TASK_QUEUE,
        )
        await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
