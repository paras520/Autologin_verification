"""Pre-flight checks before kicking off the long GEPA run.

Validates:
  1. Qwen proxy reachable (raw HTTP call)
  2. Qwen returns valid response within 5 min
  3. DeepEval can route a judge call to Qwen (a single GEval invocation)
  4. All goldens parse cleanly and contain expected_output for matcher/classifier
  5. All 6 prompt files (3 live + 3 expanded) load cleanly

Exit codes:
  0  -> all checks passed, safe to launch
  1  -> at least one check failed, DO NOT launch

Usage:
  python -m scripts.preflight_gepa
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / ".env")

QWEN_BASE_URL = (
    os.getenv("QWEN_BASE_URL")
    or os.getenv("LITELLM_PROXY_URL")
    or os.getenv("OPENAI_BASE_URL")
    or os.getenv("OPENAI_API_BASE")
)
QWEN_API_KEY = (
    os.getenv("QWEN_API_KEY")
    or os.getenv("LITELLM_PROXY_API_KEY")
    or os.getenv("OPENAI_API_KEY")
)
QWEN_MODEL = os.getenv("QWEN_MODEL", "Qwen/Qwen3.5-9B")

if QWEN_BASE_URL:
    os.environ["OPENAI_BASE_URL"] = QWEN_BASE_URL
    os.environ["OPENAI_API_BASE"] = QWEN_BASE_URL
if QWEN_API_KEY:
    os.environ["OPENAI_API_KEY"] = QWEN_API_KEY

# Same DeepEval timeout settings as the real run, so judge ping respects them
os.environ.setdefault("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE", "600")
os.environ.setdefault("DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE", "1200")
os.environ.setdefault("DEEPEVAL_RETRY_MAX_ATTEMPTS", "3")

PROMPTS_DIR = REPO_ROOT / "prompts"
GOLDENS_PATH = REPO_ROOT / "tests" / "goldens" / "golden_cases.json"

REQUIRED_PROMPTS = [
    "identifier_extractor.json",
    "identifier_extractor_expanded.json",
    "service_matcher.json",
    "service_matcher_expanded.json",
    "customer_facing_classifier.json",
    "customer_facing_classifier_expanded.json",
]

results: list[tuple[str, bool, str]] = []


def _record(name: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}" + (f"  -- {detail}" if detail else ""))
    results.append((name, ok, detail))


# ---------------------------------------------------------------------------
# Check 1: Qwen direct
# ---------------------------------------------------------------------------

async def check_qwen_direct() -> None:
    print("\n[1/5] Qwen proxy direct call ...")
    if not QWEN_BASE_URL or not QWEN_API_KEY:
        _record("qwen-direct", False, "missing QWEN_BASE_URL or QWEN_API_KEY in env")
        return
    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(base_url=QWEN_BASE_URL, api_key=QWEN_API_KEY, timeout=300.0)
        t0 = time.time()
        resp = await client.chat.completions.create(
            model=QWEN_MODEL,
            messages=[{"role": "user", "content": "Reply with the single word: OK"}],
            max_tokens=10,
            temperature=0.0,
        )
        dt = time.time() - t0
        msg = resp.choices[0].message
        text = (msg.content or "") + (getattr(msg, "reasoning", "") or "")
        _record("qwen-direct", bool(text.strip()), f"{dt:.1f}s, len={len(text)}")
    except Exception as e:
        _record("qwen-direct", False, f"{type(e).__name__}: {str(e)[:200]}")


# ---------------------------------------------------------------------------
# Check 2: Qwen via langfuse_helper (the path the pipeline uses)
# ---------------------------------------------------------------------------

async def check_qwen_via_helper() -> None:
    print("\n[2/5] Qwen via langfuse_helper.call_litellm ...")
    try:
        from langfuse_helper import call_litellm

        config = {"model": QWEN_MODEL, "max_tokens": 50, "temperature": 0.0}
        messages = [{"role": "user", "content": "Reply with valid JSON: {\"status\": \"ok\"}"}]
        t0 = time.time()
        resp = await call_litellm(
            config=config,
            messages=messages,
            session_id="preflight",
            tag_suffix="preflight",
        )
        dt = time.time() - t0
        msg = resp.choices[0].message
        text = (msg.content or "") + (getattr(msg, "reasoning", "") or "")
        _record("qwen-helper", bool(text.strip()), f"{dt:.1f}s, len={len(text)}")
    except Exception as e:
        _record("qwen-helper", False, f"{type(e).__name__}: {str(e)[:200]}")


# ---------------------------------------------------------------------------
# Check 3: DeepEval judge -> Qwen (one GEval call on one test case)
# ---------------------------------------------------------------------------

def check_deepeval_judge() -> None:
    print("\n[3/5] DeepEval judge -> Qwen routing (via LocalModel) ...")
    try:
        from deepeval.metrics import GEval
        from deepeval.test_case import LLMTestCase, LLMTestCaseParams
        from tests.eval.qwen_judge import get_qwen_judge

        metric = GEval(
            name="Preflight Sanity",
            criteria=(
                "Determine if the actual_output is a syntactically valid JSON object. "
                "If yes, score >= 0.8. If no, score <= 0.3."
            ),
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
            threshold=0.5,
            model=get_qwen_judge(),
            async_mode=False,
        )
        tc = LLMTestCase(
            input='{"page": "test"}',
            actual_output='{"status": "ok", "value": 42}',
        )
        t0 = time.time()
        metric.measure(tc)
        dt = time.time() - t0
        _record(
            "deepeval-judge",
            metric.score is not None,
            f"{dt:.1f}s, score={metric.score}",
        )
    except Exception as e:
        _record("deepeval-judge", False, f"{type(e).__name__}: {str(e)[:200]}")


# ---------------------------------------------------------------------------
# Check 4: Goldens validate
# ---------------------------------------------------------------------------

def check_goldens() -> None:
    print("\n[4/5] Golden cases validate ...")
    try:
        with open(GOLDENS_PATH, encoding="utf-8") as f:
            cases = json.load(f)
        if not cases:
            _record("goldens", False, "empty file")
            return
        missing_input = [c.get("id", "?") for c in cases if not c.get("input")]
        missing_expected = [
            c.get("id", "?")
            for c in cases
            if not c.get("expected_output")
        ]
        if missing_input:
            _record("goldens", False, f"missing input: {missing_input[:5]}")
            return
        if missing_expected:
            _record(
                "goldens",
                False,
                f"missing expected_output: {missing_expected[:5]} (matcher/classifier need these)",
            )
            return
        _record("goldens", True, f"{len(cases)} cases, all valid")
    except Exception as e:
        _record("goldens", False, f"{type(e).__name__}: {str(e)[:200]}")


# ---------------------------------------------------------------------------
# Check 5: Prompt files load
# ---------------------------------------------------------------------------

def check_prompts() -> None:
    print("\n[5/5] Prompt files load ...")
    missing = []
    bad = []
    for fname in REQUIRED_PROMPTS:
        p = PROMPTS_DIR / fname
        if not p.exists():
            missing.append(fname)
            continue
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            if not data.get("prompt"):
                bad.append(f"{fname}: no 'prompt' key")
        except Exception as e:
            bad.append(f"{fname}: {type(e).__name__}: {str(e)[:80]}")
    if missing or bad:
        _record(
            "prompts",
            False,
            f"missing={missing} bad={bad}",
        )
    else:
        _record("prompts", True, f"all {len(REQUIRED_PROMPTS)} files loaded")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def _run_async_checks():
    await check_qwen_direct()
    await check_qwen_via_helper()


def main() -> int:
    print("=" * 70)
    print("GEPA PRE-FLIGHT CHECKS")
    print("=" * 70)
    print(f"  QWEN_BASE_URL = {QWEN_BASE_URL}")
    print(f"  QWEN_MODEL    = {QWEN_MODEL}")
    print(f"  GOLDENS_PATH  = {GOLDENS_PATH}")

    asyncio.run(_run_async_checks())
    check_deepeval_judge()
    check_goldens()
    check_prompts()

    print("\n" + "=" * 70)
    print("PRE-FLIGHT SUMMARY")
    print("=" * 70)
    failed = [name for name, ok, _ in results if not ok]
    for name, ok, detail in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}: {detail}")
    if failed:
        print(f"\n[FAIL] {len(failed)} check(s) failed: {failed}")
        print("[FAIL] DO NOT launch GEPA until these are fixed.")
        return 1
    print("\n[OK] All pre-flight checks passed. Safe to launch GEPA.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
