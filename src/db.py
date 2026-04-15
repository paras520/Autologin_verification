"""Database access layer.

Mirrors the fetchRows activity in displaySort.activities.js:
  - connects via DATABASE_URL env var (postgres DSN)
  - fetches all login_services rows for a given cb_link_id
  - optionally filters to active-only rows

Expected env var:
  DATABASE_URL — e.g. postgresql://user:pass@host:5432/dbname?sslmode=require
"""

from __future__ import annotations

import logging
import os
import ssl
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
