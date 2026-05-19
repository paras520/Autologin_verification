"""Post-GEPA compression pass.

Run this AFTER GEPA finishes. For each stage's live prompt:

  1. Take the current prompt.
  2. Ask Qwen to shrink it (preserving every functional instruction +
     output schema) by ~40-60%.
  3. Re-evaluate the shrunk prompt against goldens with the same metrics
     used by GEPA. (cheap: just runs the metric, not full GEPA mutation
     loop.)
  4. If shrunk score >= original score - QUALITY_TOLERANCE, KEEP shrunk.
     Otherwise revert to original.
  5. Save report -> output/compression_report_<ts>.json

Usage:
  python -m scripts.compress_prompts                          # all stages
  python -m scripts.compress_prompts --stage extractor        # one stage
  python -m scripts.compress_prompts --target-ratio 0.5       # target 50% size
  python -m scripts.compress_prompts --tolerance 0.05         # allow 5% quality drop
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

QWEN_BASE_URL = (
    os.getenv("QWEN_BASE_URL")
    or os.getenv("OPENAI_BASE_URL")
    or os.getenv("OPENAI_API_BASE")
)
QWEN_API_KEY = (
    os.getenv("QWEN_API_KEY")
    or os.getenv("OPENAI_API_KEY")
)
QWEN_MODEL = os.getenv("QWEN_MODEL", "Qwen/Qwen3.5-9B")

if QWEN_BASE_URL:
    os.environ["OPENAI_BASE_URL"] = QWEN_BASE_URL
    os.environ["OPENAI_API_BASE"] = QWEN_BASE_URL
if QWEN_API_KEY:
    os.environ["OPENAI_API_KEY"] = QWEN_API_KEY

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

PROMPTS_DIR = REPO_ROOT / "prompts"
OUTPUT_DIR = REPO_ROOT / "output"

STAGES = {
    "extractor": "identifier_extractor.json",
    "matcher": "service_matcher.json",
    "classifier": "customer_facing_classifier.json",
}

COMPRESSION_INSTRUCTION = """
You are a prompt-compression assistant. Your task: rewrite the SYSTEM prompt
and USER template below to be ~{target_pct}% of their original length while
preserving:

  * Every functional instruction (what to do, what to extract, what to return).
  * The exact JSON output schema (field names, types, allowed values).
  * Any examples or constraints that affect output quality.

Aggressively remove:
  * Filler phrases ("Please", "Make sure to", "It's important that...").
  * Redundant restatements.
  * Long preambles.
  * Unnecessary punctuation/hedging.
  * Whitespace and blank lines.

DO NOT:
  * Drop fields from the schema.
  * Change the meaning of any rule.
  * Translate to a different language.
  * Add new instructions.

Return ONLY a JSON object of this exact shape (no commentary, no fences):
{{
  "system": "<compressed system prompt>",
  "user": "<compressed user template; keep all {{variable}} placeholders intact>"
}}

ORIGINAL PROMPT TO COMPRESS:
==== SYSTEM ====
{system}

==== USER ====
{user}
"""


async def _qwen_compress(system: str, user: str, target_pct: int) -> dict:
    """Call Qwen to compress the prompt. Returns dict with 'system' and 'user'."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=QWEN_BASE_URL, api_key=QWEN_API_KEY, timeout=300.0)
    prompt = COMPRESSION_INSTRUCTION.format(
        target_pct=target_pct, system=system, user=user
    )
    resp = await client.chat.completions.create(
        model=QWEN_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4096,
        temperature=0.0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    msg = resp.choices[0].message
    text = (msg.content or "") + (getattr(msg, "reasoning", "") or "")
    text = text.strip()
    # Strip code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    # Best-effort JSON extraction
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        first = text.find("{")
        last = text.rfind("}")
        if first != -1 and last > first:
            return json.loads(text[first:last + 1])
        raise


def _load_prompt(filename: str) -> dict:
    with open(PROMPTS_DIR / filename, encoding="utf-8") as f:
        data = json.load(f)
    for msg in data.get("prompt", []):
        if "_content_lines" in msg:
            msg["content"] = "\n".join(msg.pop("_content_lines"))
    return data


def _save_prompt(filename: str, data: dict) -> None:
    out = copy.deepcopy(data)
    for msg in out.get("prompt", []):
        if "content" in msg:
            msg["_content_lines"] = msg.pop("content").split("\n")
    with open(PROMPTS_DIR / filename, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


def _split_messages(data: dict) -> tuple[str, str]:
    sys_text, user_text = "", ""
    for msg in data.get("prompt", []):
        if msg.get("role") == "system":
            sys_text += msg.get("content", "")
        elif msg.get("role") == "user":
            user_text += msg.get("content", "")
    return sys_text, user_text


def _replace_messages(data: dict, system: str, user: str) -> dict:
    new = copy.deepcopy(data)
    new_msgs = []
    has_sys = has_user = False
    for msg in new.get("prompt", []):
        if msg.get("role") == "system" and not has_sys:
            new_msgs.append({**msg, "content": system})
            has_sys = True
        elif msg.get("role") == "user" and not has_user:
            new_msgs.append({**msg, "content": user})
            has_user = True
    if not has_sys:
        new_msgs.insert(0, {"role": "system", "content": system})
    if not has_user:
        new_msgs.append({"role": "user", "content": user})
    new["prompt"] = new_msgs
    new["version"] = (new.get("version") or 0) + 1
    return new


async def _evaluate_stage(stage: str) -> float:
    """Run the existing eval harness for one stage; return overall pass-rate.

    We import lazily to avoid loading deepeval until needed.
    """
    from tests.eval.run_eval import build_test_cases, load_goldens, _EVAL_ASYNC
    from deepeval import evaluate
    from tests.eval.extractor_metrics import EXTRACTOR_METRICS
    from tests.eval.matcher_metrics import MATCHER_METRICS
    from tests.eval.classifier_metrics import CLASSIFIER_METRICS

    metrics = {
        "extractor": EXTRACTOR_METRICS,
        "matcher": MATCHER_METRICS,
        "classifier": CLASSIFIER_METRICS,
    }[stage]

    cases = load_goldens()
    test_cases = await build_test_cases(cases, stage)
    if not test_cases:
        return 0.0
    sub_cases = (
        [tc for tc in test_cases if tc.expected_output is not None]
        if stage in ("matcher", "classifier")
        else [tc for tc in test_cases if tc.expected_output is None]
    )
    if not sub_cases:
        return 0.0
    result = evaluate(test_cases=sub_cases, metrics=list(metrics), async_config=_EVAL_ASYNC)
    scores = []
    for tr in result.test_results:
        for m in tr.metrics_data:
            if m.score is not None:
                scores.append(m.score)
    return sum(scores) / len(scores) if scores else 0.0


async def compress_one(stage: str, target_ratio: float, tolerance: float) -> dict:
    filename = STAGES[stage]
    print("\n" + "=" * 70)
    print(f"COMPRESSING: {stage}  ({filename})")
    print("=" * 70)

    original = _load_prompt(filename)
    sys_text, user_text = _split_messages(original)
    orig_chars = len(sys_text) + len(user_text)
    print(f"  original size: {orig_chars} chars")

    # Backup original
    backup_path = PROMPTS_DIR / (filename + ".pre_compress.bak")
    shutil.copy2(PROMPTS_DIR / filename, backup_path)
    print(f"  backup saved -> {backup_path.name}")

    # Score original
    print(f"  scoring original ...")
    orig_score = await _evaluate_stage(stage)
    print(f"  original score: {orig_score:.4f}")

    # Compress
    target_pct = int(target_ratio * 100)
    print(f"  asking Qwen to compress to ~{target_pct}% ...")
    t0 = time.time()
    try:
        compressed = await _qwen_compress(sys_text, user_text, target_pct)
    except Exception as e:
        print(f"  ERROR during compression: {type(e).__name__}: {e}")
        return {
            "stage": stage, "kept": False, "reason": f"compression call failed: {e}",
            "orig_chars": orig_chars, "orig_score": orig_score,
        }
    new_sys = compressed.get("system", sys_text)
    new_user = compressed.get("user", user_text)
    new_chars = len(new_sys) + len(new_user)
    ratio = new_chars / max(1, orig_chars)
    print(f"  compressed size: {new_chars} chars  ({ratio:.0%} of original) in {time.time() - t0:.1f}s")

    # Apply and score
    candidate = _replace_messages(original, new_sys, new_user)
    _save_prompt(filename, candidate)
    print(f"  scoring compressed ...")
    new_score = await _evaluate_stage(stage)
    print(f"  compressed score: {new_score:.4f}")

    keep = new_score >= (orig_score - tolerance)
    if keep:
        print(f"  KEEP compressed (delta={new_score - orig_score:+.4f}, tolerance={tolerance})")
    else:
        # Revert
        shutil.copy2(backup_path, PROMPTS_DIR / filename)
        print(f"  REVERT to original (delta={new_score - orig_score:+.4f} below tolerance={tolerance})")

    return {
        "stage": stage,
        "kept": keep,
        "orig_chars": orig_chars,
        "new_chars": new_chars,
        "compression_ratio": round(ratio, 3),
        "orig_score": round(orig_score, 4),
        "new_score": round(new_score, 4),
        "delta": round(new_score - orig_score, 4),
        "tolerance": tolerance,
        "backup": backup_path.name,
    }


async def main():
    parser = argparse.ArgumentParser(description="Compress prompts after GEPA")
    parser.add_argument("--stage", choices=list(STAGES.keys()) + ["all"], default="all")
    parser.add_argument("--target-ratio", type=float, default=0.5,
                        help="Target compression ratio (default 0.5 = aim for 50%% size)")
    parser.add_argument("--tolerance", type=float, default=0.03,
                        help="Allowed quality drop (default 0.03 = 3%%)")
    args = parser.parse_args()

    targets = list(STAGES.keys()) if args.stage == "all" else [args.stage]
    results = []
    for s in targets:
        r = await compress_one(s, args.target_ratio, args.tolerance)
        results.append(r)

    OUTPUT_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = OUTPUT_DIR / f"compression_report_{ts}.json"
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print("COMPRESSION SUMMARY")
    print("=" * 70)
    for r in results:
        status = "KEPT  " if r.get("kept") else "REVERT"
        print(
            f"  [{status}] {r['stage']:11}  "
            f"size={r.get('orig_chars', '?')}->{r.get('new_chars', '?')} "
            f"({r.get('compression_ratio', '?')*100:.0f}%)  "
            f"score={r.get('orig_score', '?'):.3f}->{r.get('new_score', '?'):.3f}  "
            f"delta={r.get('delta', '?'):+.4f}"
        )
    print(f"\n  full report -> {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
