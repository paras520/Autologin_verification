"""GEPA-based prompt optimizer for the three-stage LLM verification pipeline.

Optimizes prompts sequentially:
  1. Extractor prompt  (identifier_extractor.json)     — scored with G-Eval metrics
  2. Matcher prompt    (service_matcher.json)          — scored with DAG metric
  3. Classifier prompt (customer_facing_classifier.json) — scored with G-Eval metrics

Each optimizer starts from the EXPANDED prompt (e.g. identifier_extractor_expanded.json)
and writes the shrunk result back to the live prompt file (e.g. identifier_extractor.json).

Prompt length + mutation tracking is logged for every stage.

Usage:
    cd <repo root>
    python -m tests.eval.optimize_prompts                        # optimize all 3
    python -m tests.eval.optimize_prompts --stage extractor      # only extractor
    python -m tests.eval.optimize_prompts --stage matcher        # only matcher
    python -m tests.eval.optimize_prompts --stage classifier     # only classifier
    python -m tests.eval.optimize_prompts --iterations 10        # more GEPA iterations
    python -m tests.eval.optimize_prompts --track-metrics        # detailed mutation log

Requires:
    - .env with Langfuse + LiteLLM credentials (pipeline LLM calls)
    - OPENAI_API_KEY for DeepEval judge + GEPA mutation LLM
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

# Route DeepEval judge/optimizer calls to the same direct Qwen endpoint.
# Must happen BEFORE deepeval imports.
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

# Raise DeepEval's per-attempt / per-task timeouts so slow Qwen reasoning
# (often 60-250s per call) doesn't trip the default 60-180s caps used inside
# deepeval/models/retry_policy.py (asyncio.wait_for(coro, per_attempt_timeout)).
# These MUST be set before any `deepeval` import so settings pick them up.
os.environ.setdefault("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE", "600")  # 10 min / attempt
os.environ.setdefault("DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE", "3600")    # 60 min outer (8 attempts)
os.environ.setdefault("DEEPEVAL_RETRY_MAX_ATTEMPTS", "8")                       # ride out 5xx storms
os.environ.setdefault("DEEPEVAL_RETRY_INITIAL_SECONDS", "5")
os.environ.setdefault("DEEPEVAL_RETRY_CAP_SECONDS", "120")                      # cap backoff at 2min
os.environ.setdefault("DEEPEVAL_RETRY_EXP_BASE", "2")                           # 5, 10, 20, 40, 80, 120, 120, 120s

if sys.platform == "win32":
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# DeepEval's PromptOptimizer calls asyncio.run() inside an already-running loop.
# Patch so nested event loops are allowed.
import nest_asyncio
nest_asyncio.apply()

from deepeval.dataset import Golden
from deepeval.optimizer import PromptOptimizer
from deepeval.optimizer.algorithms import GEPA
from deepeval.prompt import Prompt, PromptMessage
from deepeval.evaluate.configs import AsyncConfig
from tests.eval.qwen_judge import get_qwen_judge


# ---------------------------------------------------------------------------
# Early-stop helper for GEPA
# ---------------------------------------------------------------------------
#   - Stops the stage when MAX_REJECT_STREAK consecutive REJECTs occur OR the
#     parent has hit a perfect score (>= EARLY_STOP_PERFECT_SCORE) and at
#     least one iteration has been REJECTED at that score (proving GEPA can't
#     beat it).
#   - Implemented by monkey-patching `_should_accept_child` to raise a
#     sentinel exception which GEPA's loop catches and breaks on. The post-
#     loop code still computes the best prompt found so far.

class _GEPAStopSignal(Exception):
    """Sentinel raised to break GEPA's iteration loop early (clean exit)."""


_EARLY_STOP_REJECT_STREAK = int(os.getenv("GEPA_EARLY_STOP_REJECTS", "3"))
_EARLY_STOP_PERFECT_SCORE = float(os.getenv("GEPA_EARLY_STOP_PERFECT", "0.99"))
# New: stop if the running best score hasn't improved for N iterations,
# even if individual ACCEPTs are still happening (minibatch-noise wins).
# Set to 0 to disable. Default 5 is conservative for noisy minibatch metrics.
_EARLY_STOP_NO_IMPROVE_ITERS = int(os.getenv("GEPA_EARLY_STOP_NO_IMPROVE", "5"))
# How much score improvement counts as "real" (epsilon to filter pure noise).
_EARLY_STOP_IMPROVE_EPS = float(os.getenv("GEPA_EARLY_STOP_EPS", "0.005"))


def _wrap_gepa_with_early_stop(gepa: GEPA) -> GEPA:
    """Patch a GEPA instance to track REJECT streaks and stop early.

    Three independent stop conditions:
      1. Perfect score plateau: best_parent >= EARLY_STOP_PERFECT and at least
         one REJECT has proved no mutation can beat it.
      2. Reject streak: REJECT_STREAK consecutive REJECTs (search plateaued).
      3. No-improvement window: NO_IMPROVE_ITERS iterations have passed without
         the best observed score moving up by EARLY_STOP_EPS. Catches the case
         where ACCEPTs keep firing on minibatch noise but the actual best
         prompt isn't getting better (e.g. matcher peaked at iter 2 then bounced
         around for 18 more iters).
    """
    original_should_accept = gepa._should_accept_child
    state = {
        "reject_streak": 0,
        "best_score": 0.0,
        "iters_since_best": 0,
        "iter_count": 0,
    }

    def patched(parent_score: float, child_score: float) -> bool:
        accepted = original_should_accept(parent_score, child_score)
        state["iter_count"] += 1
        observed = max(parent_score, child_score) if accepted else parent_score
        if observed > state["best_score"] + _EARLY_STOP_IMPROVE_EPS:
            state["best_score"] = observed
            state["iters_since_best"] = 0
        else:
            state["iters_since_best"] += 1

        if accepted:
            state["reject_streak"] = 0
        else:
            state["reject_streak"] += 1
            print(
                f"[EARLY-STOP] reject_streak={state['reject_streak']}/{_EARLY_STOP_REJECT_STREAK}  "
                f"best_score={state['best_score']:.4f}  "
                f"iters_since_best={state['iters_since_best']}"
            )

        stop_reason = None
        if state["best_score"] >= _EARLY_STOP_PERFECT_SCORE and state["reject_streak"] >= 1:
            stop_reason = "perfect score reached"
        elif state["reject_streak"] >= _EARLY_STOP_REJECT_STREAK:
            stop_reason = f"{state['reject_streak']} consecutive REJECTs"
        elif (
            _EARLY_STOP_NO_IMPROVE_ITERS > 0
            and state["iters_since_best"] >= _EARLY_STOP_NO_IMPROVE_ITERS
            and state["iter_count"] >= _EARLY_STOP_NO_IMPROVE_ITERS  # don't fire on iter 0
        ):
            stop_reason = (
                f"no improvement (>{_EARLY_STOP_IMPROVE_EPS}) in last "
                f"{state['iters_since_best']} iterations; best={state['best_score']:.4f}"
            )

        if stop_reason:
            print(f"[EARLY-STOP] STOPPING: {stop_reason}")
            raise _GEPAStopSignal(stop_reason)
        return accepted

    gepa._should_accept_child = patched
    return gepa

# Cap GEPA's internal callback concurrency. Higher = faster but more pressure
# on the single Qwen GPU (more 524s). Tune via env.
_OPT_CONCURRENCY = int(os.getenv("OPT_CALLBACK_CONCURRENCY", "2"))  # safe default for shared GPU
_OPTIMIZER_ASYNC = AsyncConfig(run_async=True, max_concurrent=_OPT_CONCURRENCY)

from src.heuristics import extract_and_score, assess_match_with_identifiers, classify_customer_facing
from tests.eval.extractor_metrics import EXTRACTOR_METRICS
from tests.eval.matcher_metrics import MATCHER_METRICS
from tests.eval.classifier_metrics import CLASSIFIER_METRICS

GOLDENS_PATH = REPO_ROOT / "tests" / "goldens" / "golden_cases.json"
PROMPTS_DIR = REPO_ROOT / "prompts"
OUTPUT_DIR = REPO_ROOT / "output"
CHECKPOINTS_DIR = OUTPUT_DIR / "gepa_checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)


def _snapshot_prompt(stage: str, prompt) -> None:
    """Best-effort: write the prompt currently being tested to a rolling checkpoint.

    GEPA exposes the candidate prompt to our callback. We can't tell which one
    is "best so far" but we always overwrite a `<stage>_latest.json` file so
    if GEPA crashes mid-run, we have something close to the latest candidate
    to fall back to (vs losing the whole stage).
    """
    try:
        out = []
        for msg in getattr(prompt, "messages_template", []) or []:
            out.append({"role": msg.role, "content": msg.content})
        if not out:
            return
        path = CHECKPOINTS_DIR / f"{stage}_latest.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"stage": stage, "ts": time.time(), "prompt": out}, f, indent=2, ensure_ascii=False)
    except Exception:
        pass  # snapshot is best-effort; never break the pipeline

# ---------------------------------------------------------------------------
# Tracking / metric helpers
# ---------------------------------------------------------------------------

_tracker: dict = {"stages": []}


def _prompt_text(data: dict) -> str:
    """Flatten prompt messages into a single string for length measurement."""
    parts = []
    for msg in data.get("prompt", []):
        content = msg.get("content") or ""
        if isinstance(content, str):
            parts.append(content)
    return "\n".join(parts)


def _count_tokens(text: str) -> int:
    """Very rough token estimate (1 token ≈ 4 chars for English prose)."""
    return max(1, len(text) // 4)


def _log_initial(stage: str, filename: str) -> dict:
    """Record the starting prompt length before optimization."""
    path = PROMPTS_DIR / filename
    data = _load_prompt_file(filename)
    text = _prompt_text(data)
    entry = {
        "stage": stage,
        "filename": filename,
        "initial_chars": len(text),
        "initial_lines": text.count("\n") + 1,
        "initial_tokens": _count_tokens(text),
        "mutations": [],
        "start_time": time.time(),
    }
    _tracker["stages"].append(entry)
    print(f"[track] {stage} initial: {entry['initial_chars']} chars / {entry['initial_lines']} lines / ~{entry['initial_tokens']} tokens")
    return entry


def _log_mutation(entry: dict, mutation_index: int, old_text: str, new_text: str, accepted: bool) -> None:
    """Record a single mutation with delta stats."""
    delta = len(new_text) - len(old_text)
    entry["mutations"].append({
        "index": mutation_index,
        "delta_chars": delta,
        "accepted": accepted,
        "new_length": len(new_text),
    })
    status = "ACCEPTED" if accepted else "REJECTED"
    print(f"[track] mutation {mutation_index}: {status}  delta={delta:+d} chars  new_len={len(new_text)}")


def _log_final(entry: dict, final_text: str) -> None:
    """Record final stats and print a compression summary."""
    entry["end_time"] = time.time()
    entry["elapsed_sec"] = round(entry["end_time"] - entry["start_time"], 2)
    entry["final_chars"] = len(final_text)
    entry["final_lines"] = final_text.count("\n") + 1
    entry["final_tokens"] = _count_tokens(final_text)
    entry["compression_chars"] = round(
        (1 - entry["final_chars"] / max(1, entry["initial_chars"])) * 100, 2
    )
    entry["compression_tokens"] = round(
        (1 - entry["final_tokens"] / max(1, entry["initial_tokens"])) * 100, 2
    )
    accepted = sum(1 for m in entry["mutations"] if m["accepted"])
    rejected = len(entry["mutations"]) - accepted
    print(f"[track] {entry['stage']} final: {entry['final_chars']} chars / {entry['final_lines']} lines / ~{entry['final_tokens']} tokens")
    print(f"[track] {entry['stage']} compression: {entry['compression_chars']}% chars  {entry['compression_tokens']}% tokens  accepted={accepted}  rejected={rejected}  time={entry['elapsed_sec']}s")


def _write_tracker_report() -> None:
    """Write the full mutation/compression report to output/."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUTPUT_DIR / f"prompt_optimize_report_{ts}.json"
    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_tracker, f, indent=2, ensure_ascii=False)
    print(f"[track] full report written to {path}")


# ---------------------------------------------------------------------------
# Prompt I/O helpers
# ---------------------------------------------------------------------------

def load_goldens() -> list[dict]:
    with open(GOLDENS_PATH, encoding="utf-8") as f:
        return json.load(f)


def _load_prompt_file(filename: str) -> dict:
    with open(PROMPTS_DIR / filename, encoding="utf-8") as f:
        data = json.load(f)
    for msg in data.get("prompt", []):
        if "_content_lines" in msg:
            msg["content"] = "\n".join(msg.pop("_content_lines"))
    return data


def _save_prompt_file(filename: str, data: dict) -> None:
    for msg in data.get("prompt", []):
        if "content" in msg:
            msg["_content_lines"] = msg.pop("content").split("\n")
    with open(PROMPTS_DIR / filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[optimize] saved optimized prompt to prompts/{filename}")


def _build_golden_input(case: dict) -> str:
    return json.dumps(case["input"], indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Extractor optimization
# ---------------------------------------------------------------------------

def build_extractor_prompt(expanded: bool = True) -> Prompt:
    src = "identifier_extractor_expanded.json" if expanded else "identifier_extractor.json"
    data = _load_prompt_file(src)
    messages = []
    for msg in data["prompt"]:
        messages.append(PromptMessage(role=msg["role"], content=msg["content"]))
    return Prompt(alias="identifier_extractor", messages_template=messages)


def build_extractor_goldens(cases: list[dict]) -> list[Golden]:
    goldens = []
    for case in cases:
        goldens.append(Golden(input=_build_golden_input(case)))
    return goldens


async def _extractor_callback_async(prompt: Prompt, golden: Golden) -> str:
    inp = json.loads(golden.input)
    page_data = inp.get("page_data", inp)

    system_content = ""
    user_template = ""
    for msg in prompt.messages_template:
        if msg.role == "system":
            system_content = msg.content
        elif msg.role == "user":
            user_template = msg.content

    result = await extract_and_score(
        provider=inp.get("provider", ""),
        service_name=inp.get("service_name", ""),
        url=inp.get("url", ""),
        page_result=page_data,
        session_id="optimize-extractor",
    )
    return json.dumps(result, ensure_ascii=False)


def extractor_callback(prompt: Prompt, golden: Golden) -> str:
    _snapshot_prompt("extractor", prompt)
    return asyncio.run(_extractor_callback_async(prompt, golden))


def optimize_extractor(cases: list[dict], iterations: int, track: bool) -> None:
    print("\n" + "=" * 70)
    print("OPTIMIZING EXTRACTOR PROMPT")
    print("=" * 70)

    prompt = build_extractor_prompt(expanded=True)
    goldens = build_extractor_goldens(cases)

    entry = _log_initial("extractor", "identifier_extractor_expanded.json") if track else None

    optimizer = PromptOptimizer(
        metrics=EXTRACTOR_METRICS,
        model_callback=extractor_callback,
        optimizer_model=get_qwen_judge(),
        algorithm=_wrap_gepa_with_early_stop(GEPA(iterations=iterations, pareto_size=min(3, len(goldens)))),
        async_config=_OPTIMIZER_ASYNC,
    )

    optimized_prompt = optimizer.optimize(prompt=prompt, goldens=goldens)

    data = _load_prompt_file("identifier_extractor.json")
    new_messages = []
    for msg in optimized_prompt.messages_template:
        new_messages.append({"role": msg.role, "content": msg.content})
    data["prompt"] = new_messages
    data["version"] = data.get("version", 0) + 1
    _save_prompt_file("identifier_extractor.json", data)

    if track and entry:
        final_text = _prompt_text(data)
        _log_final(entry, final_text)

    print("[optimize] extractor prompt optimization complete")


# ---------------------------------------------------------------------------
# Matcher optimization
# ---------------------------------------------------------------------------

def build_matcher_prompt(expanded: bool = True) -> Prompt:
    src = "service_matcher_expanded.json" if expanded else "service_matcher.json"
    data = _load_prompt_file(src)
    messages = []
    for msg in data["prompt"]:
        messages.append(PromptMessage(role=msg["role"], content=msg["content"]))
    return Prompt(alias="service_matcher", messages_template=messages)


def build_matcher_goldens(cases: list[dict]) -> list[Golden]:
    goldens = []
    for case in cases:
        goldens.append(Golden(
            input=_build_golden_input(case),
            expected_output=json.dumps(case["expected_output"], ensure_ascii=False),
        ))
    return goldens


async def _matcher_callback_async(prompt: Prompt, golden: Golden) -> str:
    inp = json.loads(golden.input)
    page_data = inp.get("page_data", inp)

    extractor_result = await extract_and_score(
        provider=inp.get("provider", ""),
        service_name=inp.get("service_name", ""),
        url=inp.get("url", ""),
        page_result=page_data,
        session_id="optimize-matcher",
    )

    matcher_result = await assess_match_with_identifiers(
        provider=inp.get("provider", ""),
        service_name=inp.get("service_name", ""),
        url=inp.get("url", ""),
        extractor_result=extractor_result,
        session_id="optimize-matcher",
    )
    return json.dumps(matcher_result, ensure_ascii=False)


def matcher_callback(prompt: Prompt, golden: Golden) -> str:
    _snapshot_prompt("matcher", prompt)
    return asyncio.run(_matcher_callback_async(prompt, golden))


def optimize_matcher(cases: list[dict], iterations: int, track: bool) -> None:
    print("\n" + "=" * 70)
    print("OPTIMIZING MATCHER PROMPT")
    print("=" * 70)

    prompt = build_matcher_prompt(expanded=True)
    goldens = build_matcher_goldens(cases)

    entry = _log_initial("matcher", "service_matcher_expanded.json") if track else None

    optimizer = PromptOptimizer(
        metrics=MATCHER_METRICS,
        model_callback=matcher_callback,
        optimizer_model=get_qwen_judge(),
        algorithm=_wrap_gepa_with_early_stop(GEPA(iterations=iterations, pareto_size=min(3, len(goldens)))),
        async_config=_OPTIMIZER_ASYNC,
    )

    optimized_prompt = optimizer.optimize(prompt=prompt, goldens=goldens)

    data = _load_prompt_file("service_matcher.json")
    new_messages = []
    for msg in optimized_prompt.messages_template:
        new_messages.append({"role": msg.role, "content": msg.content})
    data["prompt"] = new_messages
    data["version"] = data.get("version", 0) + 1
    _save_prompt_file("service_matcher.json", data)

    if track and entry:
        final_text = _prompt_text(data)
        _log_final(entry, final_text)

    print("[optimize] matcher prompt optimization complete")


# ---------------------------------------------------------------------------
# Classifier optimization  (NEW)
# ---------------------------------------------------------------------------

def build_classifier_prompt(expanded: bool = True) -> Prompt:
    src = "customer_facing_classifier_expanded.json" if expanded else "customer_facing_classifier.json"
    data = _load_prompt_file(src)
    messages = []
    for msg in data["prompt"]:
        messages.append(PromptMessage(role=msg["role"], content=msg["content"]))
    return Prompt(alias="customer_facing_classifier", messages_template=messages)


def build_classifier_goldens(cases: list[dict]) -> list[Golden]:
    goldens = []
    for case in cases:
        goldens.append(Golden(
            input=_build_golden_input(case),
            expected_output=json.dumps({
                "is_customer_facing": case["expected_output"].get("is_customer_facing"),
                "category": case["expected_output"].get("category"),
            }, ensure_ascii=False),
        ))
    return goldens


async def _classifier_callback_async(prompt: Prompt, golden: Golden) -> str:
    inp = json.loads(golden.input)
    page_data = inp.get("page_data", inp)

    result = await classify_customer_facing(
        provider=inp.get("provider", ""),
        service_name=inp.get("service_name", ""),
        url=inp.get("url", ""),
        page_result=page_data,
        session_id="optimize-classifier",
    )
    return json.dumps(result, ensure_ascii=False)


def classifier_callback(prompt: Prompt, golden: Golden) -> str:
    _snapshot_prompt("classifier", prompt)
    return asyncio.run(_classifier_callback_async(prompt, golden))


def optimize_classifier(cases: list[dict], iterations: int, track: bool) -> None:
    print("\n" + "=" * 70)
    print("OPTIMIZING CLASSIFIER PROMPT")
    print("=" * 70)

    prompt = build_classifier_prompt(expanded=True)
    goldens = build_classifier_goldens(cases)

    entry = _log_initial("classifier", "customer_facing_classifier_expanded.json") if track else None

    optimizer = PromptOptimizer(
        metrics=CLASSIFIER_METRICS,
        model_callback=classifier_callback,
        optimizer_model=get_qwen_judge(),
        algorithm=_wrap_gepa_with_early_stop(GEPA(iterations=iterations, pareto_size=min(3, len(goldens)))),
        async_config=_OPTIMIZER_ASYNC,
    )

    optimized_prompt = optimizer.optimize(prompt=prompt, goldens=goldens)

    data = _load_prompt_file("customer_facing_classifier.json")
    new_messages = []
    for msg in optimized_prompt.messages_template:
        new_messages.append({"role": msg.role, "content": msg.content})
    data["prompt"] = new_messages
    data["version"] = data.get("version", 0) + 1
    _save_prompt_file("customer_facing_classifier.json", data)

    if track and entry:
        final_text = _prompt_text(data)
        _log_final(entry, final_text)

    print("[optimize] classifier prompt optimization complete")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Optimize prompts with GEPA")
    parser.add_argument(
        "--stage",
        choices=["extractor", "matcher", "classifier", "all"],
        default="all",
        help="Which prompt to optimize (default: all — extractor, matcher, classifier)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="GEPA iterations per prompt (default: 20)",
    )
    parser.add_argument(
        "--track-metrics",
        action="store_true",
        help="Log prompt lengths, mutation deltas, and write a compression report",
    )
    args = parser.parse_args()

    cases = load_goldens()
    if not cases:
        print("[optimize] No golden cases found.")
        sys.exit(1)

    print(f"[optimize] loaded {len(cases)} golden case(s)")
    print(f"[optimize] stage: {args.stage}, iterations: {args.iterations}")

    if args.stage in ("extractor", "all"):
        optimize_extractor(cases, args.iterations, track=args.track_metrics)

    if args.stage in ("matcher", "all"):
        optimize_matcher(cases, args.iterations, track=args.track_metrics)

    if args.stage in ("classifier", "all"):
        optimize_classifier(cases, args.iterations, track=args.track_metrics)

    if args.track_metrics:
        _write_tracker_report()

    print("\n[optimize] all done.")


if __name__ == "__main__":
    main()
