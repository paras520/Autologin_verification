"""Push completed verification results to m103's QA ingest endpoint.

m103 owns the qa_check_runs / qa_check_results tables that m114 reads for the
QA Review tab.  After every batch run we POST the full result set here so the
dashboard reflects the latest verdicts.

Required env var:
  M103_BASE_URL — base URL of the m103 service (default: http://127.0.0.1:8001)
"""

from __future__ import annotations

import logging

import httpx

from src.config import config

from src.models.response_models import BatchCheckResponse

logger = logging.getLogger("autologin.m103_ingest")

_INGEST_PATH = "/whitelistautologin/qa-results/ingest"


async def ingest_to_m103(
    run_id: str,
    results: BatchCheckResponse,
    triggered_by: str | None,
) -> None:
    """POST verification results to m103.  Non-fatal — logs on failure."""
    endpoint = config.m103_base_url + _INGEST_PATH

    payload = {
        "results": [
            {
                "cb_link_id": cb_result.cb_link_id,
                "total_rows": cb_result.total_rows,
                "rows": [row.model_dump() for row in cb_result.rows],
            }
            for cb_result in results.results
        ],
        "run_mode": "batch",
        "triggered_by": triggered_by or "autologin_verification",
        "meta": {"activity_run_id": run_id},
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(endpoint, json=payload)
            resp.raise_for_status()
            data = resp.json().get("data", {})
            logger.info(
                "[ingest] m103 accepted results — qa_run_id=%s inserted=%s flagged=%s",
                data.get("run_id"), data.get("inserted"), data.get("flagged"),
            )
    except httpx.HTTPStatusError as exc:
        logger.error(
            "[ingest] m103 returned %s for run %s: %s",
            exc.response.status_code, run_id, exc.response.text,
        )
    except Exception as exc:
        logger.error("[ingest] failed to push results to m103 for run %s: %s", run_id, exc)
