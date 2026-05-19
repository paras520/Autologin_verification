"""Offline Phase-1 dedupe report.

Reads output/db_rows_dump.json, runs detect_duplicates (Phase 1: Cases 1 + 3)
across every cb_link_id, and writes:

    output/dedupe_report_<ts>.jsonl   one line per cb_link_id with counts
    output/dedupe_report_<ts>.txt     human-readable summary by case

Phase 2 (Case 2) requires verification scores and is not exercised here —
the offline report only validates what the pre-verification pass can see.
A tiny smoke block at the bottom still exercises Phase 2 against a hand-
crafted fixture so both helpers stay honest.

Usage:
    venv\\Scripts\\python scripts\\run_dedupe_report.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.services.duplicate_service import (  # noqa: E402
    ACTION_DELETE_SAME_NAME_WORSE,
    ACTION_DELETE_SAME_URL,
    ACTION_KEEP,
    _normalize_name,
    _normalize_url,
    detect_duplicates,
    resolve_parent_child,
    resolve_same_name_duplicates,
)

DB_DUMP_PATH = REPO_ROOT / "output" / "db_rows_dump.json"
OUTPUT_DIR = REPO_ROOT / "output"


def _classify_group_case(group: list[dict]) -> str:
    """Label a multi-row URL group as case_1 (same name) or case_3 (diff names)."""
    names = {_normalize_name(r.get("login_service") or "") for r in group}
    return "case_3" if len(names) > 1 else "case_1"


def build_report(local_db: dict) -> tuple[list[dict], str]:
    jsonl_entries: list[dict] = []
    txt_lines: list[str] = []

    total_rows = 0
    total_keep = 0
    total_delete_url = 0
    cb_groups_case1 = 0
    cb_groups_case3 = 0
    case_examples: list[dict] = []

    for cb_link_id in sorted(local_db.keys()):
        rows = [dict(r) for r in local_db[cb_link_id]]  # defensive copy
        total_rows += len(rows)

        rows = detect_duplicates(rows)

        # Group by normalized URL for case-level reporting
        groups: dict[str, list[dict]] = {}
        for row in rows:
            norm = _normalize_url(row.get("login_url") or "")
            if norm:
                groups.setdefault(norm, []).append(row)

        case1_groups = 0
        case3_groups = 0
        for norm_url, group in groups.items():
            if len(group) < 2:
                continue
            label = _classify_group_case(group)
            if label == "case_1":
                case1_groups += 1
            else:
                case3_groups += 1
            case_examples.append({
                "cb_link_id": cb_link_id,
                "case": label,
                "normalized_url": norm_url,
                "group_size": len(group),
                "names": sorted({r.get("login_service") or "" for r in group}),
                "canonical_display_name": next(
                    (r.get("canonical_display_name") for r in group
                     if r.get("canonical_display_name")),
                    None,
                ),
                "survivor_id": next(
                    (str(r.get("id") or "") for r in group
                     if r.get("dedupe_action") == ACTION_KEEP),
                    None,
                ),
            })

        cb_groups_case1 += case1_groups
        cb_groups_case3 += case3_groups

        action_counts = Counter(r.get("dedupe_action") for r in rows)
        keep = action_counts.get(ACTION_KEEP, 0)
        del_url = action_counts.get(ACTION_DELETE_SAME_URL, 0)
        total_keep += keep
        total_delete_url += del_url

        jsonl_entries.append({
            "cb_link_id": cb_link_id,
            "total_rows": len(rows),
            "keep": keep,
            "delete_same_url": del_url,
            "case_1_groups": case1_groups,
            "case_3_groups": case3_groups,
        })

    txt_lines.append("=" * 70)
    txt_lines.append("DEDUPE REPORT  (Phase 1 only - Cases 1 + 3)")
    txt_lines.append("=" * 70)
    txt_lines.append(f"cb_link_ids scanned:      {len(local_db)}")
    txt_lines.append(f"rows scanned:             {total_rows}")
    txt_lines.append(f"rows kept:                {total_keep}")
    txt_lines.append(f"rows flagged delete_url:  {total_delete_url}")
    txt_lines.append(f"Case 1 groups total:      {cb_groups_case1}  (same name, same URL)")
    txt_lines.append(f"Case 3 groups total:      {cb_groups_case3}  (diff names, same URL)")
    txt_lines.append("")

    if case_examples:
        txt_lines.append("-" * 70)
        txt_lines.append("Groups that triggered Case 1 or Case 3:")
        txt_lines.append("-" * 70)
        for ex in case_examples:
            txt_lines.append(
                f"  [{ex['case']}] cb_link_id={ex['cb_link_id']}  size={ex['group_size']}"
            )
            txt_lines.append(f"      url:    {ex['normalized_url']}")
            txt_lines.append(f"      names:  {ex['names']}")
            if ex["case"] == "case_3":
                txt_lines.append(
                    f"      canonical_display_name -> {ex['canonical_display_name']!r}"
                )
            txt_lines.append(f"      survivor_id: {ex['survivor_id']}")
            txt_lines.append("")
    else:
        txt_lines.append("(no Case 1 or Case 3 duplicates found)")

    return jsonl_entries, "\n".join(txt_lines)


def run_phase2_smoke() -> str:
    """Hand-crafted fixture to prove Phase 2 picks winners by score."""
    fixture = [
        {
            "id": "row-a", "cb_link_id": "X", "login_service": "NetBanking",
            "url": "https://a.example.com/login", "sorting_order": 1,
            "health_check": True,  "page_match_score": 90,
            "marked_for_deletion": False, "dedupe_action": ACTION_KEEP,
            "is_duplicate": False,
        },
        {
            "id": "row-b", "cb_link_id": "X", "login_service": "NetBanking",
            "url": "https://b.example.com/login", "sorting_order": 2,
            "health_check": True,  "page_match_score": 30,
            "marked_for_deletion": False, "dedupe_action": ACTION_KEEP,
            "is_duplicate": False,
        },
        {
            "id": "row-c", "cb_link_id": "X", "login_service": "NetBanking",
            "url": "https://c.example.com/login", "sorting_order": 3,
            "health_check": False, "page_match_score": None,
            "marked_for_deletion": True, "dedupe_action": ACTION_KEEP,
            "is_duplicate": False,
        },
    ]
    resolve_same_name_duplicates(fixture)
    resolve_parent_child(fixture)

    lines = []
    lines.append("-" * 70)
    lines.append("Phase 2 smoke (Case 2 — same name, different URLs):")
    lines.append("-" * 70)
    for row in fixture:
        lines.append(
            f"  id={row['id']}  action={row['dedupe_action']}"
            f"  dup_of={row.get('duplicate_of_id')}"
            f"  score={row['page_match_score']}  health={row['health_check']}"
        )
    lines.append("")

    # Sanity assertions — crash loud if the helper regresses
    by_id = {r["id"]: r for r in fixture}
    assert by_id["row-a"]["dedupe_action"] == ACTION_KEEP, \
        f"row-a should win, got {by_id['row-a']['dedupe_action']}"
    assert by_id["row-b"]["dedupe_action"] == ACTION_DELETE_SAME_NAME_WORSE, \
        f"row-b should lose, got {by_id['row-b']['dedupe_action']}"
    assert by_id["row-c"]["dedupe_action"] == ACTION_DELETE_SAME_NAME_WORSE, \
        f"row-c should lose, got {by_id['row-c']['dedupe_action']}"
    assert by_id["row-b"]["duplicate_of_id"] == "row-a"
    lines.append("  [OK] Phase 2 smoke assertions passed")
    lines.append("")
    return "\n".join(lines)


def main():
    if not DB_DUMP_PATH.exists():
        print(f"[error] dump file missing: {DB_DUMP_PATH}")
        sys.exit(1)

    with open(DB_DUMP_PATH, encoding="utf-8") as f:
        local_db = json.load(f)

    jsonl_entries, txt = build_report(local_db)
    smoke_block = run_phase2_smoke()

    OUTPUT_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_path = OUTPUT_DIR / f"dedupe_report_{ts}.jsonl"
    txt_path = OUTPUT_DIR / f"dedupe_report_{ts}.txt"

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for entry in jsonl_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(txt)
        f.write("\n\n")
        f.write(smoke_block)

    print(txt)
    print(smoke_block)
    print(f"[output] wrote {jsonl_path}")
    print(f"[output] wrote {txt_path}")


if __name__ == "__main__":
    main()
