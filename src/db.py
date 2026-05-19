"""Database access layer.

Mirrors the fetchRows activity in displaySort.activities.js:
  - connects via DATABASE_URL env var (postgres DSN)
  - fetches all login_services rows for a given cb_link_id
  - optionally filters to active-only rows

Expected env var:
  DATABASE_URL — e.g. postgresql://user:pass@host:5432/dbname?sslmode=require
"""

from __future__ import annotations

import json
import logging
import os
import ssl
import time
import uuid
from urllib.parse import urlparse, parse_qs
from typing import Any

import asyncpg

logger = logging.getLogger("autologin.db")


def _build_connect_kwargs(dsn: str) -> dict:
    """Parse DSN and extract asyncpg-compatible connect kwargs.

    asyncpg does not support ?sslmode=require in the DSN string directly —
    it must be passed as ssl=True in the connect() call.
    """
    parsed = urlparse(dsn)
    qs = parse_qs(parsed.query)
    sslmode = qs.get("sslmode", [None])[0]

    # Strip query string from DSN so asyncpg doesn't choke on unknown params
    clean_dsn = parsed._replace(query="").geturl()

    kwargs: dict = {"dsn": clean_dsn}
    if sslmode == "require":
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        kwargs["ssl"] = ctx

    return kwargs


async def fetch_rows(cb_link_id: str, include_inactive: bool = True) -> list[dict[str, Any]]:
    """Fetch all login_services rows for a cb_link_id.

    Mirrors the Prisma fetchRows activity in displaySort.activities.js.
    Returns a list of dicts with keys:
      id, cb_link_id, login_service, login_url, display_name, sorting_order, status
    """
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise RuntimeError("DATABASE_URL environment variable is not set.")

    query = """
        SELECT
            id,
            cb_link_id,
            login_service,
            login_url,
            display_name,
            sorting_order,
            status
        FROM login_services
        WHERE cb_link_id = $1
    """
    params: list[Any] = [cb_link_id]

    if not include_inactive:
        query += " AND status = $2"
        params.append("active")

    query += " ORDER BY created_at DESC"

    connect_kwargs = _build_connect_kwargs(dsn)
    conn = await asyncpg.connect(**connect_kwargs)
    try:
        rows = await conn.fetch(query, *params)
        logger.info("[db] fetched %d rows for %s", len(rows), cb_link_id)
        return [dict(row) for row in rows]
    finally:
        await conn.close()


async def create_activity_run(
    cb_link_ids: list[str],
    triggered_by: str = "system",
) -> str:
    """Insert one activity_runs row (status=queued) plus one activity_run_items row per
    cb_link_id into the shared whitelist-loging-services DB.  Returns the run UUID."""
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise RuntimeError("DATABASE_URL environment variable is not set.")

    run_id = uuid.uuid4()
    now_ms = int(time.time() * 1000)

    connect_kwargs = _build_connect_kwargs(dsn)
    conn = await asyncpg.connect(**connect_kwargs)
    try:
        async with conn.transaction():
            await conn.execute(
                """
                INSERT INTO activity_runs
                    (id, activity_type, run_mode, source_module, triggered_by,
                     triggered_at, status, total_items, success_items, failed_items,
                     created_at, updated_at)
                VALUES ($1, 'autologin_verification', 'adhoc'::run_mode_enum,
                        'autologin_verification', $2, $3,
                        'queued'::run_status_enum, $4, 0, 0, $3, $3)
                """,
                run_id, triggered_by, now_ms, len(cb_link_ids),
            )
            for cb_link_id in cb_link_ids:
                await conn.execute(
                    """
                    INSERT INTO activity_run_items
                        (id, run_id, entity_type, entity_id, queued_at, status,
                         attempt_count, created_at, updated_at)
                    VALUES ($1, $2, 'cb_link_id', $3, $4,
                            'queued'::item_status_enum, 1, $4, $4)
                    """,
                    uuid.uuid4(), run_id, cb_link_id, now_ms,
                )
        logger.info("[db] created activity_run %s for %d cb_links", run_id, len(cb_link_ids))
        return str(run_id)
    finally:
        await conn.close()


async def start_activity_run(run_id: str) -> None:
    """Transition activity_runs status → running and set started_at."""
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise RuntimeError("DATABASE_URL environment variable is not set.")

    now_ms = int(time.time() * 1000)
    connect_kwargs = _build_connect_kwargs(dsn)
    conn = await asyncpg.connect(**connect_kwargs)
    try:
        await conn.execute(
            """
            UPDATE activity_runs
            SET status = 'running'::run_status_enum, started_at = $1, updated_at = $1
            WHERE id = $2::uuid
            """,
            now_ms, run_id,
        )
    finally:
        await conn.close()


async def upsert_run_item_result(
    run_id: str,
    cb_link_id: str,
    metrics: dict,
    succeeded: bool,
) -> None:
    """Write verification metrics onto the matching activity_run_items row."""
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise RuntimeError("DATABASE_URL environment variable is not set.")

    now_ms = int(time.time() * 1000)
    status = "completed" if succeeded else "failed"
    connect_kwargs = _build_connect_kwargs(dsn)
    conn = await asyncpg.connect(**connect_kwargs)
    try:
        await conn.execute(
            """
            UPDATE activity_run_items
            SET status = $1::item_status_enum,
                ended_at = $2,
                metrics  = $3::jsonb,
                updated_at = $2
            WHERE run_id = $4::uuid AND entity_id = $5
            """,
            status, now_ms, json.dumps(metrics), run_id, cb_link_id,
        )
    finally:
        await conn.close()


async def finalize_activity_run(
    run_id: str,
    success_items: int,
    failed_items: int,
) -> None:
    """Set final status, end timestamp, and item counts on the activity_runs row."""
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise RuntimeError("DATABASE_URL environment variable is not set.")

    now_ms = int(time.time() * 1000)
    if failed_items == 0:
        final_status = "completed"
    elif success_items == 0:
        final_status = "failed"
    else:
        final_status = "partial"

    connect_kwargs = _build_connect_kwargs(dsn)
    conn = await asyncpg.connect(**connect_kwargs)
    try:
        await conn.execute(
            """
            UPDATE activity_runs
            SET status        = $1::run_status_enum,
                ended_at      = $2,
                success_items = $3,
                failed_items  = $4,
                updated_at    = $2
            WHERE id = $5::uuid
            """,
            final_status, now_ms, success_items, failed_items, run_id,
        )
        logger.info(
            "[db] finalized activity_run %s → %s (ok=%d fail=%d)",
            run_id, final_status, success_items, failed_items,
        )
    finally:
        await conn.close()
