"""Cancel all queued/running activity runs in the database."""
from __future__ import annotations

import asyncio
import os
import ssl
from urllib.parse import urlparse, parse_qs

import asyncpg
from dotenv import load_dotenv

load_dotenv()


def _build_connect_kwargs(dsn: str) -> dict:
    parsed = urlparse(dsn)
    qs = parse_qs(parsed.query)
    sslmode = qs.get("sslmode", [None])[0]
    clean_dsn = parsed._replace(query="").geturl()
    kwargs: dict = {"dsn": clean_dsn}
    if sslmode == "require":
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        kwargs["ssl"] = ctx
    return kwargs


async def main() -> None:
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise RuntimeError("DATABASE_URL not set")

    conn = await asyncpg.connect(**_build_connect_kwargs(dsn))
    try:
        now_ms = int(__import__("time").time() * 1000)

        # Find all open runs
        runs = await conn.fetch(
            "SELECT id, status, total_items FROM activity_runs WHERE status IN ('queued', 'running')"
        )

        if not runs:
            print("No queued/running activity runs found.")
            return

        print(f"Found {len(runs)} open run(s):")
        for r in runs:
            print(f"  {r['id']}  status={r['status']}  total={r['total_items']}")

        async with conn.transaction():
            # Cancel all queued items within those runs
            item_count = await conn.fetchval(
                """
                UPDATE activity_run_items
                SET status = 'failed'::item_status_enum,
                    ended_at = $1,
                    updated_at = $1
                WHERE run_id = ANY($2::uuid[])
                  AND status = 'queued'
                """,
                now_ms,
                [r["id"] for r in runs],
            )

            # Cancel the runs themselves
            await conn.execute(
                """
                UPDATE activity_runs
                SET status    = 'failed'::run_status_enum,
                    ended_at  = $1,
                    updated_at = $1
                WHERE id = ANY($2::uuid[])
                """,
                now_ms,
                [r["id"] for r in runs],
            )

        print(f"\nDone — cancelled {len(runs)} run(s), {item_count or 0} queued item(s) marked failed.")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
