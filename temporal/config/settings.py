from __future__ import annotations

import os
from datetime import timedelta

from temporalio.common import RetryPolicy

# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------
TEMPORAL_HOST: str = os.getenv("TEMPORAL_URI", "localhost:7233")
TEMPORAL_NAMESPACE: str = os.getenv("TEMPORAL_NAMESPACE", "default")
TEMPORAL_API_KEY: str | None = os.getenv("TEMPORAL_API_KEY")  # None → no TLS/auth (local dev)

# ---------------------------------------------------------------------------
# Kill switch — set TEMPORAL_STATE=OFF to bypass Temporal and run inline
# ---------------------------------------------------------------------------
TEMPORAL_STATE: str = os.getenv("TEMPORAL_STATE", "ON").upper()
TEMPORAL_ENABLED: bool = TEMPORAL_STATE == "ON"

# ---------------------------------------------------------------------------
# Task queues
# ---------------------------------------------------------------------------
VERIFICATION_TASK_QUEUE = "verification-queue"
BATCH_TASK_QUEUE = "batch-verification-queue"

# ---------------------------------------------------------------------------
# Retry policies
# ---------------------------------------------------------------------------

# LLM calls: up to 5 attempts, exponential 2 s → 30 s cap
LLM_RETRY_POLICY = RetryPolicy(
    initial_interval=timedelta(seconds=2),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(seconds=30),
    maximum_attempts=5,
)

# URL health / page extraction: up to 3 attempts, quick retries
HEALTH_RETRY_POLICY = RetryPolicy(
    initial_interval=timedelta(seconds=1),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(seconds=10),
    maximum_attempts=3,
)

# Country match is deterministic — retry once only (guard against transient blips)
COUNTRY_RETRY_POLICY = RetryPolicy(
    initial_interval=timedelta(seconds=1),
    maximum_attempts=2,
)

# ---------------------------------------------------------------------------
# Activity timeouts
# ---------------------------------------------------------------------------
HEALTH_ACTIVITY_TIMEOUT = timedelta(seconds=90)   # Playwright can be slow
LLM_ACTIVITY_TIMEOUT = timedelta(seconds=60)
COUNTRY_ACTIVITY_TIMEOUT = timedelta(seconds=10)

# ---------------------------------------------------------------------------
# Queue workflow
# ---------------------------------------------------------------------------
QUEUE_WORKFLOW_ID = "batch-verification-queue"  # singleton workflow ID matches task queue (m103/m112 convention)
QUEUE_TASK_QUEUE = "batch-verification-queue"
QUEUE_ITEM_THRESHOLD = 100                      # continue_as_new after this many items

FETCH_RETRY_POLICY = RetryPolicy(
    initial_interval=timedelta(seconds=1),
    maximum_attempts=3,
)
FETCH_ACTIVITY_TIMEOUT = timedelta(seconds=30)

# ---------------------------------------------------------------------------
# Workflow timeouts
# ---------------------------------------------------------------------------
VERIFICATION_WORKFLOW_TIMEOUT = timedelta(minutes=5)
BATCH_WORKFLOW_TIMEOUT = timedelta(minutes=30)
CHILD_WORKFLOW_EXECUTION_TIMEOUT = timedelta(minutes=5)
