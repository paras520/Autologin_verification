"""
Sync prompts from Langfuse into the local prompts/ directory.

NOTE: The runtime no longer reads these files — prompts are fetched live from
Langfuse via langfuse_helper.get_prompts_from_langfuse. This script is kept as
an OPTIONAL offline inspection tool (handy for diffs, evals, and git history).

Usage:
    python scripts/sync_prompts.py

Reads LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and
LANGFUSE_PROMPT_LABEL from .env (or environment).
"""

import json
import os
import sys
from pathlib import Path
from urllib.parse import quote

import requests
from dotenv import load_dotenv

load_dotenv()

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

PROMPTS_TO_SYNC = [
    ("autologinQA/identifier_extractor", "identifier_extractor.json"),
    ("autologinQA/service_matcher", "service_matcher.json"),
    ("autologinQA/customer_facing_classifier", "customer_facing_classifier.json"),
]


def fetch_prompt(host: str, pub: str, sec: str, label: str, path: str) -> dict:
    encoded = quote(path, safe="")
    url = f"{host.rstrip('/')}/api/public/v2/prompts/{encoded}"
    r = requests.get(url, params={"label": label}, auth=(pub, sec), timeout=30)
    r.raise_for_status()
    return r.json()


def main() -> None:
    host = os.getenv("LANGFUSE_HOST")
    pub = os.getenv("LANGFUSE_PUBLIC_KEY")
    sec = os.getenv("LANGFUSE_SECRET_KEY")
    label = os.getenv("LANGFUSE_PROMPT_LABEL", "stage2")

    missing = [k for k, v in {"LANGFUSE_HOST": host, "LANGFUSE_PUBLIC_KEY": pub, "LANGFUSE_SECRET_KEY": sec}.items() if not v]
    if missing:
        print(f"[sync_prompts] ERROR: missing env vars: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)

    PROMPTS_DIR.mkdir(exist_ok=True)

    for prompt_path, filename in PROMPTS_TO_SYNC:
        print(f"[sync_prompts] fetching '{prompt_path}' (label={label}) ...", end=" ", flush=True)
        try:
            raw = fetch_prompt(host, pub, sec, label, prompt_path)
        except requests.HTTPError as e:
            print(f"FAILED ({e.response.status_code}): {e.response.text[:200]}")
            continue
        except Exception as e:
            print(f"FAILED: {e}")
            continue

        local = {
            "name": raw.get("name"),
            "version": raw.get("version"),
            "label": label,
            "config": raw.get("config") or {},
            "prompt": raw.get("prompt"),
        }

        out_path = PROMPTS_DIR / filename
        out_path.write_text(json.dumps(local, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"OK → prompts/{filename} (v{local['version']})")

    print("[sync_prompts] done.")


if __name__ == "__main__":
    main()
