"""Quickly dump all DB rows for cb_link_ids from Excel to a local JSON file."""
import asyncio
import json
import sys
from pathlib import Path

import anyio

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

from src.db import fetch_rows


async def dump():
    import openpyxl
    wb = openpyxl.load_workbook(
        str(Path("C:/Users/user/Documents/bank_cb_link_ids.xlsx"))
    )
    ws = wb.active
    ids = [str(r[0]).strip() for r in ws.iter_rows(min_row=1, values_only=True) if r[0]]
    print(f"Fetching rows for {len(ids)} cb_link_ids...")

    all_rows = {}
    for cid in ids:
        try:
            rows = await fetch_rows(cid, include_inactive=True)
            all_rows[cid] = rows
            print(f"  {cid}: {len(rows)} rows")
        except Exception as e:
            print(f"  {cid}: ERROR {e}")
            all_rows[cid] = []

    total = sum(len(v) for v in all_rows.values())
    out = REPO_ROOT / "output" / "db_rows_dump.json"
    out.parent.mkdir(exist_ok=True)
    async with await anyio.open_file(out, "w", encoding="utf-8") as f:
        await f.write(json.dumps(all_rows, ensure_ascii=False, indent=2, default=str))
    print(f"\nDONE: {total} total rows saved to {out}")


if __name__ == "__main__":
    asyncio.run(dump())
