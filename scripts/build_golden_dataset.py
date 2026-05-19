"""Build golden dataset from the best scrape-only JSONL + db_rows_dump.

Reads:
  output/scrape_only_<latest>.jsonl   — best scraping data (ground truth)
  output/db_rows_dump.json            — DB rows with login_service + cb_link_id

Writes:
  tests/goldens/golden_dataset.json   — full structured golden cases

Each golden case has:
  id, description, input (provider, service_name, url, page_data), expected_output

The "expected_output" fields that require human judgement
(bank_matched, service_matched, confidence_score, url_confidence_score) are left
as null — the dataset is intended as input to LLM evaluation runs where the LLM
fills in the predictions and humans/metrics validate them.

For unreachable and empty_server_response URLs, expected_output is pre-filled
deterministically (inactive_flagged=true, needs_human_review=true/false, etc.).

Usage:
    venv\\Scripts\\python scripts\\build_golden_dataset.py
    venv\\Scripts\\python scripts\\build_golden_dataset.py --scrape-file output/scrape_only_20260417_040949.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

OUTPUT_DIR = REPO_ROOT / "output"
GOLDENS_DIR = REPO_ROOT / "tests" / "goldens"
DB_DUMP_PATH = OUTPUT_DIR / "db_rows_dump.json"


def _slug(text: str) -> str:
    """Turn arbitrary text into a lowercase hyphen-slug for use as an ID."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")[:60]


_REPEATED_GARBAGE = re.compile(r"([^\s]{2,})\1{5,}")  # any token repeated 5+ times


def _clean_text(text: str | None) -> str | None:
    """Normalize whitespace, strip repeated font-icon garbage, trim to 800 chars."""
    if not text:
        return text
    # Collapse all whitespace sequences to a single space
    text = re.sub(r"\s+", " ", text).strip()
    # Remove tokens that repeat 5+ times in a row (font-icon/spinner artifacts)
    text = _REPEATED_GARBAGE.sub("[...]", text)
    # Final collapse of any newly adjacent spaces
    text = re.sub(r" {2,}", " ", text).strip()
    if len(text) > 800:
        return text[:800] + " [...]"
    return text


def _clean_list(items: list[str] | None) -> list[str]:
    """Strip whitespace from each item and drop empties."""
    if not items:
        return []
    cleaned = []
    for s in items:
        s = re.sub(r"\s+", " ", s or "").strip()
        if s:
            cleaned.append(s)
    return cleaned


def _trim_text(text: str | None, max_chars: int = 800) -> str | None:
    """Trim visible_text to a sensible length for the golden payload."""
    if not text:
        return text
    text = text.strip()
    if len(text) > max_chars:
        return text[:max_chars] + " [...]"
    return text


def _classify_expected(scraping: dict | None, error: str | None) -> dict:
    """Derive deterministic expected_output fields from scraping metadata.

    Fields that require LLM judgement are left as null.
    """
    if error == "timeout" or (scraping and scraping.get("network_error") in (
        "TIMEOUT", "DNS_FAILURE", "CONNECTION_RESET", "CONNECTION_REFUSED",
        "HTTP2_ERROR", "SSL_ERROR", "CONNECTION_ERROR",
    )):
        return {
            "health_check": False,
            "inactive_flagged": True,
            "marked_for_deletion": True,
            "needs_human_review": False,
            "bank_matched": None,
            "service_matched": None,
            "confidence_score": None,
            "url_confidence_score": None,
            "is_login_page": None,
            "reason_category": "UNREACHABLE",
        }

    q = (scraping or {}).get("extraction_quality", "empty")

    if q == "empty_server_response":
        return {
            "health_check": False,
            "inactive_flagged": True,
            "marked_for_deletion": False,
            "needs_human_review": True,
            "bank_matched": None,
            "service_matched": None,
            "confidence_score": None,
            "url_confidence_score": None,
            "is_login_page": None,
            "reason_category": "EMPTY_SERVER_RESPONSE",
        }

    health = (scraping or {}).get("health", "UNKNOWN")
    http_status = (scraping or {}).get("http_status")
    is_ok = health in ("OK", "REDIRECT")

    return {
        "health_check": is_ok,
        "inactive_flagged": None,
        "marked_for_deletion": None,
        "needs_human_review": None,
        "bank_matched": None,
        "service_matched": None,
        "confidence_score": None,
        "url_confidence_score": None,
        "is_login_page": (scraping or {}).get("login_form_present"),
        "reason_category": "OK" if is_ok else f"HTTP_{http_status}",
    }


def load_db_lookup(dump_path: Path) -> dict[str, dict]:
    """Return url → {cb_link_id, login_service} mapping from the DB dump."""
    with open(dump_path, encoding="utf-8") as f:
        dump = json.load(f)

    lookup: dict[str, dict] = {}
    for cb_link_id, rows in dump.items():
        for r in rows:
            url = (r.get("login_url") or "").strip()
            if url:
                lookup[url] = {
                    "cb_link_id": cb_link_id,
                    "login_service": r.get("login_service") or "",
                    "display_name": r.get("display_name") or "",
                }
    return lookup


def build_cases(scrape_path: Path, db_lookup: dict) -> list[dict]:
    cases: list[dict] = []
    seen_ids: set[str] = set()

    with open(scrape_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)

            url = entry.get("url") or ""
            scraping = entry.get("scraping") or {}
            error = entry.get("error")

            # Prefer DB metadata from the dump; fall back to what's in the JSONL
            db_meta = db_lookup.get(url) or {}
            cb_link_id = db_meta.get("cb_link_id") or entry.get("cb_link_id") or ""
            login_service = db_meta.get("login_service") or entry.get("login_service") or ""
            display_name = db_meta.get("display_name") or login_service

            # Build a stable, unique ID
            base_id = _slug(f"{cb_link_id}-{login_service}")
            case_id = base_id
            suffix = 2
            while case_id in seen_ids:
                case_id = f"{base_id}-{suffix}"
                suffix += 1
            seen_ids.add(case_id)

            # Page data for the LLM input — cleaned and deduplicated
            raw_headings = _clean_list(scraping.get("headings"))
            raw_buttons = _clean_list(scraping.get("buttons"))
            # Deduplicate while preserving order
            seen_h: set[str] = set()
            headings = [h for h in raw_headings if not (h in seen_h or seen_h.add(h))]  # type: ignore[func-returns-value]
            seen_b: set[str] = set()
            buttons = [b for b in raw_buttons if not (b in seen_b or seen_b.add(b))]  # type: ignore[func-returns-value]

            page_data = {
                "title": re.sub(r"\s+", " ", scraping.get("title") or "").strip(),
                "headings": headings[:10],
                "buttons": buttons[:15],
                "login_form_present": scraping.get("login_form_present") or False,
                "visible_text": _clean_text(scraping.get("visible_text")),
                "visible_text_length": scraping.get("visible_text_length") or 0,
                "extraction_quality": scraping.get("extraction_quality") or error or "unknown",
                "final_url": scraping.get("final_url") or url,
                "http_status": scraping.get("http_status"),
                "health": scraping.get("health") or ("timeout" if error == "timeout" else "UNKNOWN"),
                "network_error": scraping.get("network_error"),
            }

            expected = _classify_expected(scraping, error)

            description_parts = [login_service or display_name]
            if cb_link_id:
                description_parts.append(f"cb_link_id={cb_link_id}")
            if expected["reason_category"] not in ("OK",):
                description_parts.append(f"[{expected['reason_category']}]")

            case = {
                "id": case_id,
                "description": " | ".join(description_parts),
                "cb_link_id": cb_link_id,
                "input": {
                    "provider": cb_link_id,
                    "service_name": login_service,
                    "url": url,
                    "page_data": page_data,
                },
                "expected_output": expected,
            }
            cases.append(case)

    return cases


def print_summary(cases: list[dict]) -> None:
    from collections import Counter
    cats = Counter(c["expected_output"]["reason_category"] for c in cases)
    login_form = sum(
        1 for c in cases if c["input"]["page_data"].get("login_form_present")
    )
    q_counts = Counter(
        c["input"]["page_data"].get("extraction_quality") for c in cases
    )

    print(f"\n=== GOLDEN DATASET SUMMARY ===")
    print(f"Total cases: {len(cases)}")
    print(f"\nReason categories:")
    for k, v in cats.most_common():
        print(f"  {k:30s}  {v}")
    print(f"\nExtraction quality:")
    for k in ("high", "medium", "low", "empty_server_response", "unreachable", "empty", "unknown"):
        v = q_counts.get(k, 0)
        if v:
            print(f"  {k:22s}  {v}")
    print(f"\nLogin form detected: {login_form} / {len(cases)}")


def main():
    parser = argparse.ArgumentParser(description="Build golden dataset from scrape data")
    parser.add_argument(
        "--scrape-file",
        type=Path,
        default=None,
        help="Path to scrape_only JSONL (default: latest in output/)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=GOLDENS_DIR / "golden_dataset.json",
        help="Output path for golden dataset JSON",
    )
    args = parser.parse_args()

    # Auto-select latest scrape file if not specified
    scrape_path = args.scrape_file
    if scrape_path is None:
        candidates = sorted(OUTPUT_DIR.glob("scrape_only_*.jsonl"))
        if not candidates:
            print("ERROR: no scrape_only_*.jsonl found in output/")
            sys.exit(1)
        scrape_path = candidates[-1]

    print(f"[golden] scrape source : {scrape_path.name}")
    print(f"[golden] db dump       : {DB_DUMP_PATH.name}")
    print(f"[golden] output        : {args.out}")

    db_lookup = load_db_lookup(DB_DUMP_PATH)
    cases = build_cases(scrape_path, db_lookup)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2, ensure_ascii=False, default=str)

    print_summary(cases)
    print(f"\n[golden] written {len(cases)} cases to {args.out}")


if __name__ == "__main__":
    main()
