"""Per-phase checkpoint diagnostic.

Runs a sample of URLs through every phase of the verification pipeline
explicitly (not via verify_url) and writes ONE jsonl file per phase so
each checkpoint can be inspected in isolation:

    output/checkpoints_<ts>/
        01_health.jsonl
        02_audience.jsonl
        03_extractor.jsonl
        04_matcher.jsonl
        05_country.jsonl
        06_final.jsonl
        summary.txt

Each record includes the cb_link_id + url identity and the full phase
output. Records are cleaned (None-stripped, visible_text truncated in
the inspection files; full text kept in 01_health only) so humans can
actually read the files.

Usage:
    venv\\Scripts\\python scripts\\pipeline_checkpoints.py
    venv\\Scripts\\python scripts\\pipeline_checkpoints.py --limit 10
    venv\\Scripts\\python scripts\\pipeline_checkpoints.py --urls https://x https://y
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import anyio

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

os.environ.setdefault("LOG_LEVEL", "WARNING")
from src.utils.logging_utils import configure_logging
configure_logging()

from src.heuristics import (
    assess_country_match,
    assess_match_with_identifiers,
    classify_customer_facing,
    extract_and_score,
)
from src.url_health import check_url_health

DEFAULT_DB_DUMP = REPO_ROOT / "output" / "db_rows_dump.json"
DEFAULT_LIMIT = 8
DEFAULT_CONCURRENCY = 4
AUDIENCE_DELETE_THRESHOLD = 70
VISIBLE_TEXT_TRUNC = 400


def _serializable(value: Any) -> Any:
    """Recursively make a value JSON-serializable and strip None-empty noise."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        cleaned: dict = {}
        for k, v in value.items():
            cleaned[k] = _serializable(v)
        return cleaned
    if isinstance(value, (list, tuple, set)):
        return [_serializable(v) for v in value]
    return str(value)


def _truncate_visible_text(page_result: dict) -> dict:
    """Return a shallow copy of page_result with visible_text truncated."""
    out = dict(page_result)
    vt = out.get("visible_text")
    if isinstance(vt, str) and len(vt) > VISIBLE_TEXT_TRUNC:
        out["visible_text"] = vt[:VISIBLE_TEXT_TRUNC] + f" ...[truncated, total {len(vt)} chars]"
    return out


async def run_one(
    target: dict,
    ckpts_dir: Path,
    lock: asyncio.Lock,
) -> dict:
    """Run one URL through each phase, append to each checkpoint file."""
    cb_link_id = target.get("cb_link_id") or ""
    service_name = target.get("login_service") or ""
    url = (target.get("login_url") or target.get("url") or "").strip()
    ident = {
        "cb_link_id": cb_link_id,
        "login_service": service_name,
        "url": url,
    }
    row_result: dict[str, Any] = {
        "ident": ident,
        "phases": {},
        "errors": [],
    }

    # ---------- Phase 1: health + page extraction ----------
    try:
        health = await check_url_health(url)
    except Exception as exc:
        health = {"error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()}
        row_result["errors"].append(f"phase1_health: {exc}")

    async with lock:
        async with await anyio.open_file(ckpts_dir / "01_health.jsonl", "a", encoding="utf-8") as f:
            await f.write(json.dumps({**ident, "health": _serializable(health)}, ensure_ascii=False) + "\n")

    page_result = (health.get("page_result") if isinstance(health, dict) else None) or {}
    health_ok = isinstance(health, dict) and health.get("health") in {"OK", "REDIRECT"}
    row_result["phases"]["health"] = {
        "ok": health_ok,
        "status": health.get("status") if isinstance(health, dict) else None,
        "reason": health.get("reason") if isinstance(health, dict) else None,
        "final_url": (page_result.get("final_url") or url),
        "extraction_quality": page_result.get("extraction_quality"),
    }

    if not health_ok:
        row_result["final_decision"] = {
            "inactive_flagged": True,
            "reason": (health.get("reason") if isinstance(health, dict) else "health failed"),
            "marked_for_deletion": (health.get("reason") != "EMPTY_SERVER_RESPONSE") if isinstance(health, dict) else True,
            "marked_for_human_review": True,
            "exited_at_phase": "1_health",
        }
        async with lock:
            async with await anyio.open_file(ckpts_dir / "06_final.jsonl", "a", encoding="utf-8") as f:
                await f.write(json.dumps({**ident, "final": row_result["final_decision"]}, ensure_ascii=False) + "\n")
        return row_result

    # ---------- Phase 1.5: audience classifier ----------
    try:
        audience = await classify_customer_facing(
            provider=cb_link_id,
            service_name=service_name,
            url=url,
            page_result=page_result,
            session_id=f"ckpt-{cb_link_id}",
        )
    except Exception as exc:
        audience = {"error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()}
        row_result["errors"].append(f"phase1.5_audience: {exc}")

    audience_record = {
        **ident,
        "audience": _serializable(audience),
        "page_inputs": _serializable(_truncate_visible_text({
            "title": page_result.get("title"),
            "headings": page_result.get("headings"),
            "buttons": page_result.get("buttons"),
            "login_form_present": page_result.get("login_form_present"),
            "visible_text": page_result.get("visible_text"),
        })),
    }
    async with lock:
        async with await anyio.open_file(ckpts_dir / "02_audience.jsonl", "a", encoding="utf-8") as f:
            await f.write(json.dumps(audience_record, ensure_ascii=False) + "\n")

    audience_is_non_customer = bool(audience) and audience.get("is_customer_facing") is False
    audience_confidence = int(audience.get("confidence", 0)) if audience else 0
    audience_high_del = audience_is_non_customer and audience_confidence >= AUDIENCE_DELETE_THRESHOLD
    audience_uncertain = audience_is_non_customer and audience_confidence < AUDIENCE_DELETE_THRESHOLD

    row_result["phases"]["audience"] = {
        "is_customer_facing": audience.get("is_customer_facing") if audience else None,
        "confidence": audience_confidence,
        "category": audience.get("category") if audience else None,
        "short_circuit_delete": audience_high_del,
        "route_to_human_review": audience_uncertain,
    }

    if audience_high_del:
        row_result["final_decision"] = {
            "inactive_flagged": True,
            "reason": f"non-customer-facing ({audience.get('category')}): {audience.get('reason', '')}",
            "marked_for_deletion": True,
            "marked_for_human_review": False,
            "exited_at_phase": "1.5_audience",
        }
        async with lock:
            async with await anyio.open_file(ckpts_dir / "06_final.jsonl", "a", encoding="utf-8") as f:
                await f.write(json.dumps({**ident, "final": row_result["final_decision"]}, ensure_ascii=False) + "\n")
        return row_result

    # ---------- Phase 2a: cheap extractor ----------
    try:
        extractor = await extract_and_score(
            provider=cb_link_id,
            service_name=service_name,
            url=url,
            page_result=page_result,
            session_id=f"ckpt-{cb_link_id}",
        )
    except Exception as exc:
        extractor = {"error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()}
        row_result["errors"].append(f"phase2a_extractor: {exc}")

    async with lock:
        async with await anyio.open_file(ckpts_dir / "03_extractor.jsonl", "a", encoding="utf-8") as f:
            await f.write(json.dumps({**ident, "extractor": _serializable(extractor)}, ensure_ascii=False) + "\n")

    row_result["phases"]["extractor"] = {
        "bank_identifier_count": len(extractor.get("bank_identifiers", []) or []),
        "section_count": len(extractor.get("relevant_page_sections", []) or []),
        "is_login_page": extractor.get("is_login_page"),
        "login_type_suggestion": extractor.get("login_type_suggestion"),
    }

    # ---------- Phase 2b: smart matcher ----------
    try:
        matcher = await assess_match_with_identifiers(
            provider=cb_link_id,
            service_name=service_name,
            url=url,
            extractor_result=extractor,
            session_id=f"ckpt-{cb_link_id}",
        )
    except Exception as exc:
        matcher = {"error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()}
        row_result["errors"].append(f"phase2b_matcher: {exc}")

    async with lock:
        async with await anyio.open_file(ckpts_dir / "04_matcher.jsonl", "a", encoding="utf-8") as f:
            await f.write(json.dumps({**ident, "matcher": _serializable(matcher)}, ensure_ascii=False) + "\n")

    row_result["phases"]["matcher"] = {
        "bank_matched": matcher.get("bank_matched"),
        "service_matched": matcher.get("service_matched"),
        "confidence_score": matcher.get("confidence_score"),
        "url_confidence_score": matcher.get("url_confidence_score"),
        "login_type": matcher.get("login_type"),
    }

    # ---------- Phase 3: country match (deterministic) ----------
    bank_failed = matcher.get("bank_matched") is False
    service_failed = matcher.get("service_matched") is False
    prior_passed = not (bank_failed or service_failed)

    country = None
    if prior_passed:
        try:
            country = assess_country_match(expected_country="india", page_result=page_result)
        except Exception as exc:
            country = {"error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()}
            row_result["errors"].append(f"phase3_country: {exc}")

    async with lock:
        async with await anyio.open_file(ckpts_dir / "05_country.jsonl", "a", encoding="utf-8") as f:
            await f.write(json.dumps({**ident, "country": _serializable(country)}, ensure_ascii=False) + "\n")

    row_result["phases"]["country"] = {
        "matched": country.get("matched") if country else "skipped",
        "detected_country": country.get("detected_country") if country else None,
    } if country else {"matched": "skipped"}

    # ---------- Phase 4: final assembly ----------
    country_mismatch = bool(country) and country.get("matched") is False
    country_uncertain = bool(country) and country.get("matched") is None

    final = {
        "inactive_flagged": bool(bank_failed or service_failed or country_mismatch or audience_uncertain),
        "marked_for_deletion": bool(bank_failed or country_mismatch),
        "marked_for_human_review": bool(
            bank_failed or service_failed or country_mismatch or country_uncertain or audience_uncertain
        ),
        "page_match_score": matcher.get("confidence_score"),
        "reason": matcher.get("reason"),
        "exited_at_phase": "4_final",
    }
    row_result["final_decision"] = final
    async with lock:
        async with await anyio.open_file(ckpts_dir / "06_final.jsonl", "a", encoding="utf-8") as f:
            await f.write(json.dumps({**ident, "final": final}, ensure_ascii=False) + "\n")

    return row_result


def load_targets(args) -> list[dict]:
    if args.urls:
        return [{"cb_link_id": f"adhoc-{i}", "login_service": "", "login_url": u}
                for i, u in enumerate(args.urls)]

    if not DEFAULT_DB_DUMP.exists():
        print(f"ERROR: {DEFAULT_DB_DUMP} not found and no --urls given", file=sys.stderr)
        sys.exit(1)

    with open(DEFAULT_DB_DUMP, encoding="utf-8") as f:
        dump = json.load(f)

    rows = []
    seen = set()
    for cb_link_id, cb_rows in dump.items():
        for r in cb_rows:
            url = (r.get("login_url") or "").strip()
            if not url or url in seen:
                continue
            seen.add(url)
            rows.append({
                "cb_link_id": cb_link_id,
                "login_service": r.get("login_service") or "",
                "login_url": url,
            })
            if len(rows) >= args.limit:
                return rows
    return rows


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    ap.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    ap.add_argument("--urls", nargs="+", help="Explicit URLs to check (skips db_rows_dump)")
    args = ap.parse_args()

    targets = load_targets(args)
    if not targets:
        print("ERROR: no targets", file=sys.stderr)
        sys.exit(1)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpts_dir = REPO_ROOT / "output" / f"checkpoints_{ts}"
    ckpts_dir.mkdir(parents=True, exist_ok=True)
    for name in ["01_health", "02_audience", "03_extractor", "04_matcher", "05_country", "06_final"]:
        (ckpts_dir / f"{name}.jsonl").write_text("", encoding="utf-8")

    print(f"=== Pipeline checkpoint diagnostic ===")
    print(f"Targets:     {len(targets)}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Output dir:  {ckpts_dir}")
    print()

    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()

    async def bound(target):
        async with sem:
            try:
                return await run_one(target, ckpts_dir, lock)
            except Exception as exc:
                return {
                    "ident": {"cb_link_id": target.get("cb_link_id"), "url": target.get("login_url")},
                    "errors": [f"unhandled: {exc}"],
                    "phases": {},
                    "final_decision": {"inactive_flagged": True, "exited_at_phase": "unhandled_exception"},
                }

    results = await asyncio.gather(*(bound(t) for t in targets))

    # ---------- summary ----------
    by_exit = {}
    total_errors = 0
    per_phase_error: dict[str, int] = {}
    for r in results:
        exit_phase = (r.get("final_decision") or {}).get("exited_at_phase", "unknown")
        by_exit[exit_phase] = by_exit.get(exit_phase, 0) + 1
        for e in r.get("errors", []):
            total_errors += 1
            tag = e.split(":", 1)[0].strip()
            per_phase_error[tag] = per_phase_error.get(tag, 0) + 1

    lines = [
        f"Pipeline checkpoints run at {ts}",
        f"Targets:        {len(targets)}",
        f"Concurrency:    {args.concurrency}",
        f"Output dir:     {ckpts_dir}",
        "",
        "Exit phase distribution:",
    ]
    for phase, count in sorted(by_exit.items()):
        lines.append(f"  {phase:<24s} {count}")
    lines += ["", "Per-phase error counts (0 means working):"]
    for phase in ["phase1_health", "phase1.5_audience", "phase2a_extractor", "phase2b_matcher", "phase3_country"]:
        lines.append(f"  {phase:<24s} {per_phase_error.get(phase, 0)}")
    lines += ["", f"Total errors across all phases: {total_errors}"]
    lines += ["", "Files written:"]
    for p in sorted(ckpts_dir.glob("*.jsonl")):
        size = p.stat().st_size
        line_count = len(p.read_text(encoding="utf-8").splitlines())
        lines.append(f"  {p.name:<24s}  {line_count} records  {size} bytes")
    summary = "\n".join(lines)
    (ckpts_dir / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    asyncio.run(main())
