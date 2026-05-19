"""Scrape-only runner: fast iteration on page extraction quality.

Reads URLs from output/db_rows_dump.json, runs the stealth Playwright
extraction on each in parallel (no LLM calls, no DB hits), and writes:

  output/scrape_only_<ts>.jsonl   one line per URL with full extraction result
  output/scrape_quality_<ts>.txt  human-readable quality summary

Usage:
    venv\\Scripts\\python scripts\\run_scrape_only.py
    venv\\Scripts\\python scripts\\run_scrape_only.py --concurrency 8
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

import os
os.environ.setdefault("LOG_LEVEL", "WARNING")
from src.utils.logging_utils import configure_logging
configure_logging()

from src.url_health import check_url_health

logger = logging.getLogger("autologin.scrape_only")

DB_DUMP_PATH = REPO_ROOT / "output" / "db_rows_dump.json"
OUTPUT_DIR = REPO_ROOT / "output"
URL_TIMEOUT = 75


def load_urls() -> list[dict]:
    """Return a de-duplicated list of {cb_link_id, login_service, url} rows."""
    with open(DB_DUMP_PATH, encoding="utf-8") as f:
        dump = json.load(f)

    seen = set()
    rows = []
    for cb_link_id, cb_rows in dump.items():
        for r in cb_rows:
            url = (r.get("login_url") or "").strip()
            if not url or url in seen:
                continue
            seen.add(url)
            rows.append({
                "cb_link_id": cb_link_id,
                "login_service": r.get("login_service") or "",
                "url": url,
            })
    return rows


async def scrape_one(row: dict, sem: asyncio.Semaphore, counter: dict):
    async with sem:
        entry = {**row, "scraping": None, "error": None}
        try:
            health = await asyncio.wait_for(
                check_url_health(row["url"]), timeout=URL_TIMEOUT,
            )
            page_result = health.get("page_result") or {}
            entry["scraping"] = {
                "health": health.get("health"),
                "http_status": page_result.get("http_status") or health.get("status"),
                "network_error": page_result.get("network_error"),
                "reason": health.get("reason"),
                "final_url": page_result.get("final_url"),
                "title": page_result.get("title"),
                "visible_text": page_result.get("visible_text"),
                "visible_text_length": page_result.get("visible_text_length"),
                "headings": page_result.get("headings"),
                "buttons": page_result.get("buttons"),
                "login_form_present": page_result.get("login_form_present"),
                "extraction_quality": page_result.get("extraction_quality"),
                "extraction_attempts": page_result.get("extraction_attempts"),
                "used_html_fallback": page_result.get("used_html_fallback"),
                "load_error": page_result.get("load_error"),
            }
        except asyncio.TimeoutError:
            entry["error"] = "timeout"
            # Treat outer timeout the same as unreachable — site is too slow
            # or blocking us; there's nothing more we can extract.
            entry["scraping"] = {
                "health": "INACTIVE",
                "http_status": None,
                "network_error": "TIMEOUT",
                "reason": "TIMEOUT",
                "extraction_quality": "unreachable",
                "visible_text_length": 0,
                "visible_text": None,
                "headings": [],
                "buttons": [],
                "login_form_present": False,
                "used_html_fallback": False,
                "extraction_attempts": 0,
            }
        except Exception as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
            entry["scraping"] = {
                "health": "INACTIVE",
                "http_status": None,
                "network_error": "EXCEPTION",
                "reason": str(exc)[:120],
                "extraction_quality": "unreachable",
                "visible_text_length": 0,
                "visible_text": None,
                "headings": [],
                "buttons": [],
                "login_form_present": False,
                "used_html_fallback": False,
                "extraction_attempts": 0,
            }

        counter["done"] += 1
        if counter["done"] % 10 == 0 or counter["done"] == counter["total"]:
            elapsed = time.time() - counter["start"]
            print(f"[progress] {counter['done']}/{counter['total']} "
                  f"({elapsed:.0f}s)")
        return entry


def summarize(entries: list[dict]) -> str:
    lines = []
    quality = Counter()
    health = Counter()
    attempts = Counter()
    fallback_used = 0
    failed_entirely = 0
    lengths = []
    bad = []  # (quality, url, text_len, http_status)

    for e in entries:
        s = e.get("scraping") or {}
        q = s.get("extraction_quality") or "unknown"
        # Hard exception without a scraping dict still counts toward failed
        if e.get("error") and not s:
            failed_entirely += 1
            bad.append(("ERROR", e["url"], 0, e.get("error")))
            continue
        quality[q] += 1
        health[s.get("health") or "?"] += 1
        a = s.get("extraction_attempts")
        if a is not None:
            attempts[a] += 1
        if s.get("used_html_fallback"):
            fallback_used += 1
        tl = s.get("visible_text_length") or 0
        if tl:
            lengths.append(tl)
        if q in ("empty", "low", "empty_server_response"):
            bad.append((q, e["url"], tl, s.get("http_status") or s.get("network_error")))

    total = len(entries)
    # Success = we extracted SOMETHING useful (any quality >= low) OR the site
    # is genuinely unreachable/http-error/server-empty (server-side issue,
    # not a scraping bug).
    reachable_success = sum(quality.get(k, 0) for k in ("high", "medium", "low"))
    unreachable = quality.get("unreachable", 0)
    empty_server = quality.get("empty_server_response", 0)
    true_failures = quality.get("empty", 0) + quality.get("unknown", 0) + failed_entirely
    scrape_success_rate = (
        (reachable_success + unreachable + empty_server) / total * 100
        if total else 0
    )

    lines.append(f"=== SCRAPING QUALITY SUMMARY ===")
    lines.append(f"Total URLs: {total}")
    lines.append(
        f"Scrape accuracy: {scrape_success_rate:.1f}%  "
        f"(extracted content, unreachable, or server-empty — i.e. not a scraper bug)"
    )
    lines.append(f"True scrape failures (empty/unknown/hard): {true_failures}")
    lines.append("")
    lines.append("Extraction quality:")
    for k in ("high", "medium", "low", "empty",
              "empty_server_response", "unreachable", "unknown"):
        n = quality.get(k, 0)
        pct = (n / total * 100) if total else 0
        lines.append(f"  {k:22s}  {n:4d}  ({pct:5.1f}%)")
    lines.append("")
    lines.append("Health status:")
    for k, v in health.most_common():
        lines.append(f"  {k:10s} {v}")
    lines.append("")
    lines.append("Attempts distribution:")
    for k, v in sorted(attempts.items()):
        lines.append(f"  {k} attempt(s): {v}")
    lines.append(f"HTML fallback used: {fallback_used}")
    if lengths:
        lengths.sort()
        n = len(lengths)
        p50 = lengths[n // 2]
        p10 = lengths[n // 10]
        lines.append(f"Text length median={p50} p10={p10} max={lengths[-1]}")
    lines.append("")
    lines.append(f"=== FAILING URLs ({len(bad)}) ===")
    for q, url, tl, extra in bad:
        lines.append(f"  [{q:8s}] len={tl:4d} {url[:80]}  extra={extra}")

    return "\n".join(lines)


async def main_async(concurrency: int, limit: int | None):
    rows = load_urls()
    if limit:
        rows = rows[:limit]

    print(f"[scrape] {len(rows)} unique URLs, concurrency={concurrency}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    counter = {"done": 0, "total": len(rows), "start": time.time()}
    sem = asyncio.Semaphore(concurrency)

    entries = await asyncio.gather(*[
        scrape_one(r, sem, counter) for r in rows
    ], return_exceptions=False)

    elapsed = time.time() - counter["start"]
    print(f"\n[scrape] done in {elapsed:.1f}s")

    raw_path = OUTPUT_DIR / f"scrape_only_{ts}.jsonl"
    summary_path = OUTPUT_DIR / f"scrape_quality_{ts}.txt"

    with open(raw_path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False, default=str) + "\n")

    summary = summarize(entries)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)

    print(f"\n{summary}")
    print(f"\n[out] raw:     {raw_path}")
    print(f"[out] summary: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Scrape-only runner")
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit to first N URLs (for quick iteration)")
    args = parser.parse_args()
    asyncio.run(main_async(args.concurrency, args.limit))


if __name__ == "__main__":
    main()
