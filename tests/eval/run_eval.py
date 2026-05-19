"""Evaluation runner for the three-stage LLM verification pipeline.

Loads golden test cases, runs each through the actual extractor, matcher,
and customer-facing classifier, then evaluates with G-Eval + DAG metrics.

Usage:
    cd <repo root>
    python -m tests.eval.run_eval                # run all stages
    python -m tests.eval.run_eval --ids hdfc-netbanking-correct indian-bank-staff-portal
    python -m tests.eval.run_eval --stage extractor   # only extractor metrics
    python -m tests.eval.run_eval --stage matcher     # only matcher metrics
    python -m tests.eval.run_eval --stage classifier  # only classifier metrics
    python -m tests.eval.run_eval --verbose           # print DAG intermediate steps

Requires:
    - .env with Langfuse + LiteLLM credentials (for the pipeline LLM calls)
    - OPENAI_API_KEY in env for DeepEval's LLM-as-a-judge (G-Eval / DAG nodes)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

# DeepEval's internal OpenAI client defaults to api.openai.com.
# Force judge calls to the same direct Qwen endpoint.
qwen_base_url = (
    os.getenv("QWEN_BASE_URL")
    or os.getenv("LITELLM_PROXY_URL")
    or os.getenv("OPENAI_BASE_URL")
    or os.getenv("OPENAI_API_BASE")
)
qwen_api_key = (
    os.getenv("QWEN_API_KEY")
    or os.getenv("LITELLM_PROXY_API_KEY")
    or os.getenv("OPENAI_API_KEY")
)
if qwen_base_url:
    os.environ["OPENAI_BASE_URL"] = qwen_base_url
    os.environ["OPENAI_API_BASE"] = qwen_base_url
if qwen_api_key:
    os.environ["OPENAI_API_KEY"] = qwen_api_key

# Raise DeepEval's per-attempt / per-task timeouts so slow Qwen reasoning
# (often 60-250s per call) doesn't trip the default ~60-180s cap inside
# deepeval/models/retry_policy.py (asyncio.wait_for(coro, per_attempt_timeout)).
# Must be set before any `deepeval` import so settings pick them up.
os.environ.setdefault("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE", "600")  # 10 min / attempt
os.environ.setdefault("DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE", "3600")    # 60 min outer
os.environ.setdefault("DEEPEVAL_RETRY_MAX_ATTEMPTS", "8")                       # ride out 5xx storms
os.environ.setdefault("DEEPEVAL_RETRY_INITIAL_SECONDS", "5")
os.environ.setdefault("DEEPEVAL_RETRY_CAP_SECONDS", "120")
os.environ.setdefault("DEEPEVAL_RETRY_EXP_BASE", "2")

# Windows event loop fix for Playwright (if pipeline calls trigger extraction)
if sys.platform == "win32":
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.evaluate.configs import AsyncConfig

# Concurrency knobs (Qwen GPU returns 5xx under heavy load — keep modest)
_EVAL_CONCURRENCY = int(os.getenv("EVAL_CONCURRENCY", "2"))           # parallel pipeline runs
_JUDGE_CONCURRENCY = int(os.getenv("EVAL_JUDGE_CONCURRENCY", "2"))    # parallel judge calls inside evaluate()
_EVAL_ASYNC = AsyncConfig(run_async=True, max_concurrent=_JUDGE_CONCURRENCY)

from src.heuristics import (
    extract_and_score,
    assess_match_with_identifiers,
    classify_customer_facing,
)
from tests.eval.extractor_metrics import EXTRACTOR_METRICS
from tests.eval.matcher_metrics import MATCHER_METRICS
from tests.eval.classifier_metrics import CLASSIFIER_METRICS
GOLDENS_PATH = REPO_ROOT / "tests" / "goldens" / "golden_cases.json"

# Baseline prompt swap (formerly: swap local *_expanded.json files). Prompts are
# now fetched from Langfuse, so the "baseline" variant must live as a separate
# Langfuse label. We swap LANGFUSE_PROMPT_LABEL temporarily, controlled by
# LANGFUSE_BASELINE_LABEL (defaults to "baseline").
_ORIGINAL_PROMPT_LABEL: str | None = None


def _swap_to_expanded_prompts():
    """Switch the active Langfuse prompt label to the baseline variant.

    Has no effect if LANGFUSE_BASELINE_LABEL is not set or equals the current
    label — in that case the eval simply re-uses the production prompts.
    """
    global _ORIGINAL_PROMPT_LABEL
    baseline = os.getenv("LANGFUSE_BASELINE_LABEL", "")
    current = os.getenv("LANGFUSE_PROMPT_LABEL", "production")
    if not baseline or baseline == current:
        print(
            "[baseline] no-op — set LANGFUSE_BASELINE_LABEL to a different "
            "Langfuse label to enable the baseline swap"
        )
        return
    _ORIGINAL_PROMPT_LABEL = current
    os.environ["LANGFUSE_PROMPT_LABEL"] = baseline
    print(f"[baseline] LANGFUSE_PROMPT_LABEL: {current} -> {baseline}")


def _restore_prompts():
    """Restore the previously active Langfuse prompt label."""
    global _ORIGINAL_PROMPT_LABEL
    if _ORIGINAL_PROMPT_LABEL is None:
        return
    os.environ["LANGFUSE_PROMPT_LABEL"] = _ORIGINAL_PROMPT_LABEL
    print(f"[baseline] LANGFUSE_PROMPT_LABEL restored -> {_ORIGINAL_PROMPT_LABEL}")
    _ORIGINAL_PROMPT_LABEL = None


def load_goldens(ids: list[str] | None = None) -> list[dict]:
    with open(GOLDENS_PATH, encoding="utf-8") as f:
        cases = json.load(f)
    if ids:
        cases = [c for c in cases if c["id"] in ids]
    return cases


def _format_input(case: dict) -> str:
    """Serialize the input dict into a string for the LLMTestCase."""
    inp = case["input"]
    return json.dumps(inp, indent=2, ensure_ascii=False)


async def _run_extractor(case: dict) -> str:
    """Run the extractor on a single golden case and return JSON string output."""
    inp = case["input"]
    result = await extract_and_score(
        provider=inp["provider"],
        service_name=inp["service_name"],
        url=inp["url"],
        page_result=inp["page_data"],
        session_id=f"eval-{case['id']}",
    )
    return json.dumps(result, indent=2, ensure_ascii=False)


async def _run_matcher(case: dict, extractor_result: dict) -> str:
    """Run the matcher on a single golden case and return JSON string output."""
    inp = case["input"]
    result = await assess_match_with_identifiers(
        provider=inp["provider"],
        service_name=inp["service_name"],
        url=inp["url"],
        extractor_result=extractor_result,
        session_id=f"eval-{case['id']}",
    )
    return json.dumps(result, indent=2, ensure_ascii=False)


async def _run_classifier(case: dict) -> str:
    """Run the customer-facing classifier on a single golden case."""
    inp = case["input"]
    result = await classify_customer_facing(
        provider=inp["provider"],
        service_name=inp["service_name"],
        url=inp["url"],
        page_result=inp["page_data"],
        session_id=f"eval-{case['id']}",
    )
    return json.dumps(result, indent=2, ensure_ascii=False)


async def _process_one_case(case: dict, stage: str, sem: asyncio.Semaphore) -> list[LLMTestCase]:
    """Run extractor + (classifier and/or matcher) in parallel for a single golden case."""
    async with sem:
        case_id = case["id"]
        input_str = _format_input(case)
        expected_str = json.dumps(case["expected_output"], indent=2, ensure_ascii=False)

        print(f"[eval] running pipeline for case: {case_id} ...")
        t0 = asyncio.get_event_loop().time()

        extractor_task = asyncio.create_task(_run_extractor(case))
        classifier_task = (
            asyncio.create_task(_run_classifier(case))
            if stage in ("classifier", "all")
            else None
        )

        extractor_output_str = await extractor_task
        extractor_result = json.loads(extractor_output_str)

        matcher_output_str = None
        if stage in ("matcher", "all"):
            matcher_output_str = await _run_matcher(case, extractor_result)

        classifier_output_str = await classifier_task if classifier_task else None

        elapsed = asyncio.get_event_loop().time() - t0
        print(f"[eval] done: {case_id} ({elapsed:.1f}s)")

        out: list[LLMTestCase] = []
        if stage in ("extractor", "all"):
            out.append(LLMTestCase(input=input_str, actual_output=extractor_output_str))
        if stage in ("matcher", "all") and matcher_output_str is not None:
            out.append(LLMTestCase(
                input=input_str,
                actual_output=matcher_output_str,
                expected_output=expected_str,
            ))
        if stage in ("classifier", "all") and classifier_output_str is not None:
            out.append(LLMTestCase(
                input=input_str,
                actual_output=classifier_output_str,
                expected_output=expected_str,
            ))
        return out


async def build_test_cases(
    cases: list[dict],
    stage: str,
) -> list[LLMTestCase]:
    """Run all goldens in parallel (capped by EVAL_CONCURRENCY)."""
    sem = asyncio.Semaphore(_EVAL_CONCURRENCY)
    print(f"[eval] running {len(cases)} cases in parallel (concurrency={_EVAL_CONCURRENCY})")
    results = await asyncio.gather(
        *[_process_one_case(case, stage, sem) for case in cases],
        return_exceptions=False,
    )
    test_cases: list[LLMTestCase] = []
    for r in results:
        test_cases.extend(r)
    return test_cases


def _get_prompt_stats() -> dict:
    """Capture character + line counts for the three live prompt files."""
    stats = {}
    for name in ("identifier_extractor", "service_matcher", "customer_facing_classifier"):
        path = REPO_ROOT / "prompts" / f"{name}.json"
        expanded = REPO_ROOT / "prompts" / f"{name}_expanded.json"
        entry = {"live_path": str(path)}
        if path.exists():
            text = path.read_text(encoding="utf-8")
            entry["live_chars"] = len(text)
            entry["live_lines"] = text.count("\n") + 1
        if expanded.exists():
            text = expanded.read_text(encoding="utf-8")
            entry["expanded_chars"] = len(text)
            entry["expanded_lines"] = text.count("\n") + 1
        stats[name] = entry
    return stats


def main():
    parser = argparse.ArgumentParser(description="Run DeepEval evaluation on the verification pipeline")
    parser.add_argument("--ids", nargs="*", help="Only run specific golden case IDs")
    parser.add_argument(
        "--stage",
        choices=["extractor", "matcher", "classifier", "all"],
        default="all",
        help="Which stage to evaluate (default: all)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose mode on DAG metrics")
    parser.add_argument("--baseline", action="store_true", help="Save scores + prompt lengths to output/eval_baseline_<ts>.json")
    parser.add_argument("--use-expanded", action="store_true", help="Temporarily load *_expanded.json prompts instead of live prompts")
    args = parser.parse_args()

    if args.use_expanded:
        _swap_to_expanded_prompts()

    cases = load_goldens(args.ids)
    if not cases:
        print("[eval] No golden cases found. Check tests/goldens/golden_cases.json")
        sys.exit(1)

    print(f"[eval] loaded {len(cases)} golden case(s)")
    print(f"[eval] stage: {args.stage}")

    # Build test cases by running the actual pipeline
    test_cases = asyncio.run(build_test_cases(cases, args.stage))

    if not test_cases:
        print("[eval] No test cases to evaluate.")
        sys.exit(1)

    # DeepEval's evaluate() feeds ALL metrics to ALL test cases. DAG metrics
    # crash when a test case lacks expected_output, so we run per-stage batches.
    stages_to_run = ["extractor", "matcher", "classifier"] if args.stage == "all" else [args.stage]

    all_results: list[tuple[str, object]] = []
    metric_scores: dict[str, list[float]] = {}

    for sub_stage in stages_to_run:
        if sub_stage == "extractor":
            sub_cases = [tc for tc in test_cases if tc.expected_output is None]
            sub_metrics = list(EXTRACTOR_METRICS)
        elif sub_stage == "matcher":
            sub_cases = [tc for tc in test_cases if tc.expected_output is not None]
            sub_metrics = list(MATCHER_METRICS)
            if args.verbose:
                for m in sub_metrics:
                    m.verbose_mode = True
        elif sub_stage == "classifier":
            sub_cases = [tc for tc in test_cases if tc.expected_output is not None]
            sub_metrics = list(CLASSIFIER_METRICS)
        else:
            continue

        if not sub_cases:
            print(f"[eval] no test cases for stage {sub_stage}, skipping")
            continue

        print(f"[eval] evaluating {len(sub_cases)} test case(s) for stage '{sub_stage}' with {len(sub_metrics)} metric(s) ... (judge concurrency={_JUDGE_CONCURRENCY})")
        result = evaluate(test_cases=sub_cases, metrics=sub_metrics, async_config=_EVAL_ASYNC)
        all_results.append((sub_stage, result))

        for test_result in result.test_results:
            for metric_result in test_result.metrics_data:
                metric_scores.setdefault(metric_result.name, []).append(metric_result.score)

    if args.use_expanded:
        _restore_prompts()

    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    for sub_stage, result in all_results:
        print(f"\n--- Stage: {sub_stage} ---")
        for test_result in result.test_results:
            print(f"\nTest Case: {test_result.input[:80]}...")
            for metric_result in test_result.metrics_data:
                status = "PASS" if metric_result.success else "FAIL"
                print(f"  [{status}] {metric_result.name}: {metric_result.score:.2f} (threshold: {metric_result.threshold})")
                if metric_result.reason:
                    print(f"         reason: {metric_result.reason[:200]}")

    total_metrics = 0
    passed_metrics = 0
    for _, result in all_results:
        for test_result in result.test_results:
            for metric_result in test_result.metrics_data:
                total_metrics += 1
                if metric_result.success:
                    passed_metrics += 1
    overall_pass_rate = passed_metrics / total_metrics if total_metrics else 0
    print(f"\nOverall pass rate: {overall_pass_rate:.1%}")

    # Prompt length snapshot
    prompt_stats = _get_prompt_stats()
    print("\n" + "-" * 70)
    print("PROMPT LENGTH SNAPSHOT")
    print("-" * 70)
    for name, s in prompt_stats.items():
        live = f"{s.get('live_chars', '?')} chars / {s.get('live_lines', '?')} lines"
        expanded = f"{s.get('expanded_chars', '?')} chars / {s.get('expanded_lines', '?')} lines" if 'expanded_chars' in s else "N/A"
        print(f"  {name}: live={live}  expanded={expanded}")

    if args.baseline:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        baseline = {
            "timestamp": ts,
            "stage": args.stage,
            "cases_run": [c["id"] for c in cases],
            "overall_pass_rate": overall_pass_rate,
            "metric_scores": {k: {"mean": round(sum(v)/len(v), 4), "scores": v} for k, v in metric_scores.items()},
            "prompt_stats": prompt_stats,
        }
        out_path = REPO_ROOT / "output" / f"eval_baseline_{ts}.json"
        out_path.parent.mkdir(exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(baseline, f, indent=2, ensure_ascii=False)
        print(f"\n[baseline] saved to {out_path}")


if __name__ == "__main__":
    main()
