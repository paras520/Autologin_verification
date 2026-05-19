"""Smoke test for the Phase 1.5 customer-facing audience classifier.

Reads pre-scraped pages from output/scrape_only_<ts>.jsonl, runs
classify_customer_facing on each in parallel, and prints a verdict table.

Usage:
    venv\\Scripts\\python scripts\\smoke_audience_classifier.py
    venv\\Scripts\\python scripts\\smoke_audience_classifier.py --limit 20
    venv\\Scripts\\python scripts\\smoke_audience_classifier.py --input output/scrape_only_20260417_040949.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

os.environ.setdefault("LOG_LEVEL", "WARNING")
from src.utils.logging_utils import configure_logging
configure_logging()

from src.heuristics import classify_customer_facing

DEFAULT_INPUT = REPO_ROOT / "output" / "scrape_only_20260417_040949.jsonl"
DEFAULT_CONCURRENCY = 6
DEFAULT_LIMIT = 30


def load_scrape_records(path: Path, limit: int | None) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            scraping = rec.get("scraping") or {}
            if scraping.get("extraction_quality") in {"unreachable", "empty", "empty_server_response"}:
                continue
            rows.append(rec)
            if limit and len(rows) >= limit:
                break
    return rows


async def classify_one(sem: asyncio.Semaphore, rec: dict) -> dict:
    async with sem:
        scraping = rec.get("scraping") or {}
        page_result = {
            "title": scraping.get("title"),
            "headings": scraping.get("headings") or [],
            "buttons": scraping.get("buttons") or [],
            "login_form_present": bool(scraping.get("login_form_present", False)),
            "visible_text": scraping.get("visible_text") or "",
            "final_url": scraping.get("final_url") or rec.get("url"),
        }
        try:
            verdict = await classify_customer_facing(
                provider=rec.get("cb_link_id") or "",
                service_name=rec.get("login_service") or "",
                url=rec.get("url") or "",
                page_result=page_result,
                session_id=f"smoke-{rec.get('cb_link_id', '')}-{rec.get('login_service', '')}",
            )
        except Exception as exc:
            verdict = {
                "is_customer_facing": None,
                "confidence": 0,
                "category": "error",
                "reason": f"exception: {exc}",
            }
        return {
            "cb_link_id": rec.get("cb_link_id"),
            "login_service": rec.get("login_service"),
            "url": rec.get("url"),
            "verdict": verdict,
        }


def render_row(item: dict) -> str:
    v = item["verdict"] or {}
    is_cf = v.get("is_customer_facing")
    cf_str = "YES" if is_cf is True else ("NO " if is_cf is False else "?? ")
    conf = v.get("confidence", 0)
    cat = (v.get("category") or "?")[:18].ljust(18)
    svc = (item.get("login_service") or "")[:24].ljust(24)
    url = (item.get("url") or "")[:55].ljust(55)
    return f"  {cf_str} conf={conf:>3}  {cat}  {svc}  {url}"


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=str(DEFAULT_INPUT), help="Path to scrape_only_<ts>.jsonl")
    ap.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="Max records to classify")
    ap.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="Concurrent LLM calls")
    args = ap.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    records = load_scrape_records(input_path, limit=args.limit)
    if not records:
        print("ERROR: no usable scrape records found in input", file=sys.stderr)
        sys.exit(1)

    print(f"=== Audience classifier smoke test ===")
    print(f"Input:        {input_path}")
    print(f"Records:      {len(records)}")
    print(f"Concurrency:  {args.concurrency}")
    print()

    sem = asyncio.Semaphore(args.concurrency)
    tasks = [classify_one(sem, rec) for rec in records]
    results = await asyncio.gather(*tasks)

    print("  flag conf  category            login_service             url")
    print("  ---- ----  ------------------  ------------------------  -------------------------------------------------------")
    cf_count = nc_high = nc_low = unknown = err = 0
    for item in results:
        v = item["verdict"] or {}
        is_cf = v.get("is_customer_facing")
        conf = int(v.get("confidence", 0) or 0)
        cat = v.get("category")
        if cat == "error":
            err += 1
        elif is_cf is True:
            cf_count += 1
        elif is_cf is False and conf >= 70:
            nc_high += 1
        elif is_cf is False:
            nc_low += 1
        else:
            unknown += 1
        print(render_row(item))

    print()
    print("=== Summary ===")
    print(f"  customer-facing:                  {cf_count}")
    print(f"  non-customer-facing (conf>=70):   {nc_high}  -> would be marked_for_deletion")
    print(f"  non-customer-facing (conf<70):    {nc_low}   -> would be marked_for_human_review")
    print(f"  unknown / fail-open:              {unknown}")
    print(f"  errors:                           {err}")


if __name__ == "__main__":
    asyncio.run(main())
