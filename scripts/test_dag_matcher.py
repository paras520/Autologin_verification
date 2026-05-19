"""Sanity-test the matcher DAG metric.

Verifies:
  1. Metric runs without crashing
  2. CASE A: actual matches expected exactly  -> high score (~10/10)
  3. CASE B: actual disagrees on bank_matched -> 0 (gated branch should cut chain)
  4. CASE C: bank=true but scores far off    -> low-mid score (4/10 calibration)

If the DAG is correctly gated, score(B) << score(A), and (C) sits between them.
If the DAG is misconfigured (children listed as flat siblings), gated branches
fire even when they shouldn't, inflating B and C.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
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
if QWEN_BASE_URL:
    os.environ["OPENAI_BASE_URL"] = QWEN_BASE_URL
    os.environ["OPENAI_API_BASE"] = QWEN_BASE_URL
if QWEN_API_KEY:
    os.environ["OPENAI_API_KEY"] = QWEN_API_KEY

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from deepeval.test_case import LLMTestCase
from tests.eval.matcher_metrics import matcher_correctness_metric as matcher_dag_metric

EXPECTED = {
    "bank_matched": True,
    "service_matched": True,
    "confidence_score": 95,
    "url_confidence_score": 95,
}

CASES = {
    "A_exact_match": {
        "actual": {
            "bank_matched": True,
            "service_matched": True,
            "confidence_score": 95,
            "url_confidence_score": 95,
        },
        "predict": "high score (~10/10), all gates open",
    },
    "B_bank_mismatch": {
        "actual": {
            "bank_matched": False,
            "service_matched": True,  # GATED OFF if DAG correct
            "confidence_score": 95,    # GATED OFF if DAG correct
            "url_confidence_score": 30,  # url calibration is independent
        },
        "predict": "low score (~0-3/10) IF gated correctly. If DAG broken, will be inflated.",
    },
    "C_bank_match_scores_off": {
        "actual": {
            "bank_matched": True,
            "service_matched": True,
            "confidence_score": 50,    # 45 points off -> "more than 15"
            "url_confidence_score": 50,
        },
        "predict": "mid score (4-6/10) due to score calibration penalty",
    },
}


async def run_case(name: str, actual: dict) -> float:
    tc = LLMTestCase(
        input='{"provider": "HDFC", "service": "NetBanking"}',
        actual_output=json.dumps(actual, indent=2),
        expected_output=json.dumps(EXPECTED, indent=2),
    )
    print(f"\n>>> Case {name}")
    print(f"    actual:   {json.dumps(actual)}")
    print(f"    expected: {json.dumps(EXPECTED)}")
    try:
        await matcher_dag_metric.a_measure(tc)
        score = matcher_dag_metric.score
        reason = (matcher_dag_metric.reason or "")[:200]
        print(f"    SCORE = {score:.3f}")
        print(f"    REASON = {reason}")
        return score
    except Exception as e:
        print(f"    ERROR: {type(e).__name__}: {e}")
        return -1.0


async def main() -> int:
    print("=" * 70)
    print("MATCHER DAG SANITY TEST")
    print("=" * 70)
    scores: dict[str, float] = {}
    for name, info in CASES.items():
        scores[name] = await run_case(name, info["actual"])
        print(f"    EXPECTED BEHAVIOR: {info['predict']}")

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    print(f"  A_exact_match:           {scores['A_exact_match']:.3f}")
    print(f"  B_bank_mismatch:         {scores['B_bank_mismatch']:.3f}")
    print(f"  C_bank_match_scores_off: {scores['C_bank_match_scores_off']:.3f}")

    issues = []
    if scores["A_exact_match"] < 0.7:
        issues.append("A scores too low — perfect match should be > 0.7")
    if scores["B_bank_mismatch"] > 0.3:
        issues.append(
            f"B scores too high ({scores['B_bank_mismatch']:.3f}) — bank mismatch should "
            "score near 0 if DAG gating works. Higher score = double-counting bug."
        )
    if scores["C_bank_match_scores_off"] >= scores["A_exact_match"]:
        issues.append("C >= A — score calibration not penalizing as expected")

    if issues:
        print("\n[DAG ISSUES DETECTED]")
        for i in issues:
            print(f"  - {i}")
        return 1
    print("\n[OK] DAG behaves as designed.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
