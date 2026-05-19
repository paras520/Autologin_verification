"""Inspect failing/low-quality entries from the latest scrape_only run."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
out_dir = REPO_ROOT / "output"

# Find the latest scrape_only file
files = sorted(out_dir.glob("scrape_only_*.jsonl"))
if not files:
    print("no scrape_only files found")
    sys.exit(1)

target = files[-1]
print(f"inspecting {target.name}\n")

lookup_urls = sys.argv[1:] if len(sys.argv) > 1 else None

with open(target, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        e = json.loads(line)
        url = e.get("url") or ""
        s = e.get("scraping") or {}
        q = s.get("extraction_quality") or "?"
        tl = s.get("visible_text_length") or 0
        http = s.get("http_status")

        if lookup_urls:
            if not any(u in url for u in lookup_urls):
                continue
        else:
            # Only show low/empty/unknown/errored
            if q == "high" or q == "medium":
                continue

        print(f"=== {url} ===")
        print(f"  quality={q}  http_status={http}  text_len={tl}  "
              f"attempts={s.get('extraction_attempts')}  fallback={s.get('used_html_fallback')}")
        print(f"  health={s.get('health')}  reason={s.get('reason')}  network_error={s.get('network_error')}")
        print(f"  final_url={s.get('final_url')}")
        print(f"  title={(s.get('title') or '')[:80]!r}")
        print(f"  load_error={(s.get('load_error') or '')[:200]}")
        text = (s.get("visible_text") or "").replace("\n", " ")[:300]
        print(f"  text[:300]={text!r}")
        print(f"  headings={(s.get('headings') or [])[:5]}")
        print(f"  buttons={(s.get('buttons') or [])[:5]}")
        print(f"  err={e.get('error')}")
        print()
