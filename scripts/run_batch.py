"""Batch runner: process cb_link_ids from Excel, log raw scraping + final results.

Outputs two JSONL files in output/:
  - raw_scraping_<timestamp>.jsonl   one line per URL with raw page extraction data
  - results_<timestamp>.jsonl        one line per URL with final verification result

Usage:
    cd <repo root>
    venv\\Scripts\\python scripts\\run_batch.py
    venv\\Scripts\\python scripts\\run_batch.py --concurrency 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

from src.utils.logging_utils import configure_logging
configure_logging()

from src.models.request_models import CheckRequest

# Lazy import so offline mode works without asyncpg installed
fetch_rows = None  # type: ignore

def _lazy_fetch_rows(*args, **kwargs):
    global fetch_rows
    if fetch_rows is None:
        from src.db import fetch_rows as _fr
        fetch_rows = _fr
    return fetch_rows(*args, **kwargs)
from src.services.analysis_service import run_checks
from src.services.duplicate_service import (
    ACTION_KEEP,
    detect_duplicates,
    resolve_parent_child,
    resolve_same_name_duplicates,
)
from src.utils.heuristics import assess_country_match
from src.utils.url_health import check_url_health

logger = logging.getLogger("autologin.batch_runner")

EXCEL_PATH = os.path.join("C:", os.sep, "Users", "user", "Documents", "bank_cb_link_ids.xlsx")
OUTPUT_DIR = REPO_ROOT / "output"
DB_DUMP_PATH = REPO_ROOT / "output" / "db_rows_dump.json"
URL_TIMEOUT = 120  # seconds max per single URL (health + LLM)


def read_cb_link_ids(path: str) -> list[str]:
    import openpyxl
    wb = openpyxl.load_workbook(path)
    ws = wb.active
    ids = []
    for row in ws.iter_rows(min_row=1, values_only=True):
        val = row[0]
        if val:
            ids.append(str(val).strip())
    return ids


def _is_valid_url(url: str) -> bool:
    from urllib.parse import urlparse
    parsed = urlparse(url)
    return bool(parsed.scheme and parsed.netloc)


async def verify_and_capture(
    row: dict[str, Any],
    scraping_log: list[dict],
    results_log: list[dict],
    sem: asyncio.Semaphore,
    counter: dict,
):
    async with sem:
        cb_link_id = row.get("cb_link_id") or ""
        service_name = row.get("login_service") or ""
        url = (row.get("login_url") or "").strip()
        status = row.get("status") or ""

        row_meta = {
            "cb_link_id": cb_link_id,
            "login_service": service_name,
            "url": url,
            "status": status,
            "row_id": str(row.get("id") or ""),
            "sorting_order": row.get("sorting_order"),
        }

        if not url or not _is_valid_url(url):
            scraping_log.append({**row_meta, "scraping": None, "error": "invalid_url"})
            results_log.append({**row_meta, "result": None, "error": "invalid_url"})
            counter["done"] += 1
            _print_progress(counter)
            return

        try:
            await asyncio.wait_for(
                _do_verify(row_meta, cb_link_id, service_name, url, scraping_log, results_log),
                timeout=URL_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.warning("[timeout] %s %s after %ds", cb_link_id, url[:60], URL_TIMEOUT)
            scraping_log.append({**row_meta, "scraping": None, "error": "timeout"})
            results_log.append({**row_meta, "result": None, "error": "timeout"})
        except Exception as exc:
            logger.error("[error] %s %s: %s", cb_link_id, url[:60], exc)
            scraping_log.append({**row_meta, "scraping": None, "error": str(exc)})
            results_log.append({**row_meta, "result": None, "error": str(exc)})

        counter["done"] += 1
        _print_progress(counter)


def _print_progress(counter: dict):
    done = counter["done"]
    total = counter["total"]
    if done % 5 == 0 or done == total:
        elapsed = time.time() - counter["start"]
        print(f"[progress] {done}/{total} URLs processed ({elapsed:.0f}s elapsed)")


async def _do_verify(
    row_meta: dict,
    cb_link_id: str,
    service_name: str,
    url: str,
    scraping_log: list[dict],
    results_log: list[dict],
):
    health_result = await check_url_health(url)

    page_result = health_result.get("page_result") or {}
    health_ok = health_result.get("health") in {"OK", "REDIRECT"}
    token_detected = health_result.get("token_detected")

    scraping_log.append({
        **row_meta,
        "scraping": {
            "health": health_result.get("health"),
            "status_code": health_result.get("status"),
            "reason": health_result.get("reason"),
            "load_time_ms": health_result.get("load_time_ms"),
            "final_url": page_result.get("final_url"),
            "title": page_result.get("title"),
            "headings": page_result.get("headings"),
            "buttons": page_result.get("buttons"),
            "login_form_present": page_result.get("login_form_present"),
            "visible_text": page_result.get("visible_text"),
            "soft_errors": health_result.get("soft_errors"),
            "token_detected": token_detected,
        },
    })

    if not health_ok:
        reason = health_result.get("reason") or "URL unreachable"
        results_log.append({
            **row_meta,
            "result": {
                "health_check": False,
                "inactive_flagged": True,
                "marked_for_deletion": True,
                "marked_for_human_review": True,
                "reason": reason,
                "page_match_score": None,
                "bank_matched": None,
                "service_matched": None,
            },
        })
        return

    payload = CheckRequest(
        provider=cb_link_id,
        service_name=service_name,
        login_type="direct",
        url=url,
        country="india",
        cb_link_id=cb_link_id,
    )

    match_result = None
    try:
        match_result = await run_checks(
            payload=payload, url=url,
            page_result=page_result, session_id=cb_link_id,
        )
    except Exception as exc:
        logger.error("[batch] LLM failed %s: %s", url[:60], exc)

    bank_match_failed = bool(match_result) and not match_result.get("bank_matched", True)
    service_match_failed = bool(match_result) and not match_result.get("service_matched", True)
    provider_match_failed = bank_match_failed or service_match_failed

    prior_ok = health_ok and not provider_match_failed
    country_check = None
    if prior_ok:
        country_check = assess_country_match(expected_country="india", page_result=page_result)

    country_mismatch = bool(country_check) and country_check["matched"] is False

    inactive_flagged = provider_match_failed or country_mismatch or bool(token_detected)
    marked_for_deletion = bool(token_detected) or bank_match_failed or country_mismatch
    needs_human_review = provider_match_failed or country_mismatch or bool(token_detected)

    if token_detected:
        final_reason = f"Token in URL: {token_detected.get('summary', '')}"
    elif bank_match_failed:
        final_reason = match_result.get("reason", "Bank mismatch")
    elif service_match_failed:
        final_reason = match_result.get("reason", "Service mismatch")
    elif country_mismatch:
        final_reason = country_check["reason"]
    else:
        final_reason = (match_result.get("reason") if match_result else None) or "OK"

    results_log.append({
        **row_meta,
        "result": {
            "health_check": True,
            "inactive_flagged": inactive_flagged,
            "marked_for_deletion": marked_for_deletion,
            "marked_for_human_review": needs_human_review,
            "reason": final_reason,
            "page_match_score": match_result.get("confidence_score") if match_result else None,
            "url_confidence_score": match_result.get("url_confidence_score") if match_result else None,
            "bank_matched": match_result.get("bank_matched") if match_result else None,
            "service_matched": match_result.get("service_matched") if match_result else None,
            "login_type": match_result.get("login_type") if match_result else None,
            "notes": match_result.get("notes") if match_result else None,
        },
    })


async def process_cb_link(
    cb_link_id: str,
    scraping_log: list[dict],
    results_log: list[dict],
    sem: asyncio.Semaphore,
    counter: dict,
    local_db: dict | None = None,
):
    if local_db is not None and cb_link_id in local_db:
        rows = local_db[cb_link_id]
    else:
        try:
            rows = await _lazy_fetch_rows(cb_link_id, include_inactive=True)
        except Exception as exc:
            logger.error("[batch] DB fetch failed %s: %s", cb_link_id, exc)
            results_log.append({
                "cb_link_id": cb_link_id, "error": f"db_fetch_failed: {exc}",
            })
            return

    rows = detect_duplicates(rows)  # Phase 1: Cases 1 + 3
    print(f"[{cb_link_id}] {len(rows)} rows")

    # Keep track of the entries we add to results_log for this cb_link_id so
    # Phase 2 only re-scores its own rows rather than scanning the shared log.
    cb_entries: list[dict] = []

    tasks = []
    for row in rows:
        if row.get("is_duplicate"):
            entry = {
                "cb_link_id": cb_link_id,
                "login_service": row.get("login_service", ""),
                "url": row.get("login_url", ""),
                "status": row.get("status", ""),
                "row_id": str(row.get("id") or ""),
                "sorting_order": row.get("sorting_order"),
                "is_duplicate": True,
                "dedupe_action": row.get("dedupe_action"),
                "dedupe_reason": row.get("dedupe_reason"),
                "duplicate_of_id": row.get("duplicate_of_id"),
                "duplicate_of_url": row.get("duplicate_of_url"),
                "canonical_display_name": row.get("canonical_display_name"),
                "result": {
                    "is_duplicate": True,
                    "duplicate_of_url": row.get("duplicate_of_url"),
                },
            }
            results_log.append(entry)
            cb_entries.append(entry)
            continue
        tasks.append(verify_and_capture(row, scraping_log, results_log, sem, counter))

    counter["total"] += len(tasks)

    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)

    # After verification completes, pick up the entries this cb_link_id owns
    # (results_log is shared across concurrent cb_link_ids, so filter by id).
    already_tracked = {id(e) for e in cb_entries}
    for entry in results_log:
        if (
            entry.get("cb_link_id") == cb_link_id
            and id(entry) not in already_tracked
        ):
            cb_entries.append(entry)

    # ----- Phase 2 (Case 2: same name, different URLs) -----
    # Build flat views the dedupe helper expects; mutate originals via shared refs.
    views: list[dict] = []
    for entry in cb_entries:
        result_block = entry.get("result") or {}
        view = {
            "login_service": entry.get("login_service") or "",
            "url": entry.get("url") or "",
            "id": entry.get("row_id") or "",
            "sorting_order": entry.get("sorting_order"),
            "health_check": result_block.get("health_check"),
            "page_match_score": result_block.get("page_match_score"),
            "marked_for_deletion": result_block.get("marked_for_deletion"),
            "dedupe_action": entry.get("dedupe_action") or ACTION_KEEP,
            "is_duplicate": bool(entry.get("is_duplicate")),
            # carryover to preserve through Phase 2/3 helpers
            "duplicate_of_id": entry.get("duplicate_of_id"),
            "duplicate_of_url": entry.get("duplicate_of_url"),
            "dedupe_reason": entry.get("dedupe_reason"),
            "canonical_display_name": entry.get("canonical_display_name"),
        }
        views.append(view)

    resolve_same_name_duplicates(views)  # Phase 2
    resolve_parent_child(views)          # Phase 3 stub

    # Write any Phase 2 verdict changes back onto the result entries
    for entry, view in zip(cb_entries, views):
        entry["dedupe_action"] = view.get("dedupe_action")
        entry["dedupe_reason"] = view.get("dedupe_reason")
        entry["is_duplicate"] = bool(view.get("is_duplicate"))
        entry["duplicate_of_id"] = view.get("duplicate_of_id")
        entry["duplicate_of_url"] = view.get("duplicate_of_url")


def write_jsonl(path: Path, data: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    print(f"[output] wrote {len(data)} entries -> {path}")


async def main_async(concurrency: int, offline: bool):
    ids = read_cb_link_ids(EXCEL_PATH)

    local_db = None
    if offline:
        if not DB_DUMP_PATH.exists():
            print(f"[error] offline mode but dump file missing: {DB_DUMP_PATH}")
            return
        with open(DB_DUMP_PATH, encoding="utf-8") as f:
            local_db = json.load(f)
        print(f"[batch] OFFLINE mode — loaded {sum(len(v) for v in local_db.values())} rows from dump")
    
    print(f"[batch] {len(ids)} cb_link_ids from Excel")
    print(f"[batch] concurrency: {concurrency} simultaneous URLs")
    print(f"[batch] per-URL timeout: {URL_TIMEOUT}s")

    OUTPUT_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    scraping_log: list[dict] = []
    results_log: list[dict] = []
    counter = {"done": 0, "total": 0, "start": time.time()}

    sem = asyncio.Semaphore(concurrency)

    await asyncio.gather(*[
        process_cb_link(cid, scraping_log, results_log, sem, counter, local_db)
        for cid in ids
    ], return_exceptions=True)

    elapsed = time.time() - counter["start"]
    print(f"\n[batch] DONE in {elapsed:.1f}s")
    print(f"[batch] scraped: {len(scraping_log)}  results: {len(results_log)}")

    scraping_path = OUTPUT_DIR / f"raw_scraping_{ts}.jsonl"
    results_path = OUTPUT_DIR / f"results_{ts}.jsonl"
    write_jsonl(scraping_path, scraping_log)
    write_jsonl(results_path, results_log)


def main():
    parser = argparse.ArgumentParser(description="Batch verify cb_link_ids from Excel")
    parser.add_argument(
        "--concurrency", type=int, default=5,
        help="Max simultaneous URL verifications (default: 5)",
    )
    parser.add_argument(
        "--offline", action="store_true",
        help="Use local db_rows_dump.json instead of live DB",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args.concurrency, args.offline))


if __name__ == "__main__":
    main()
