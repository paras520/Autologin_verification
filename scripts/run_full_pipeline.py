"""Full pipeline: baseline eval → optimize (30 iters × 3 stages) → post-opt eval → batch → compare.

Stages:
  1. baseline_eval  — Run current prompts on golden cases, save baseline.json
  2. optimize       — Run 30 GEPA iterations for each of 3 stages (extractor, matcher, classifier)
                     Each stage writes its optimized prompt to <stage>_opt.json (doesn't overwrite live prompts)
  3. post_opt_eval  — Run same golden cases through optimized prompts, save post_opt.json
  4. batch_baseline — Run batch on cb_link_ids.xlsx using live (pre-opt) prompts, save batch_baseline.jsonl
  5. batch_opt      — Run batch on cb_link_ids.xlsx using opt prompts, save batch_opt.jsonl
  6. compare        — Merge all results, produce output/compare_report_*.json

Usage:
    python scripts/run_full_pipeline.py                    — run all stages
    python scripts/run_full_pipeline.py --stage baseline   — baseline eval only
    python scripts/run_full_pipeline.py --stage optimize   — optimize only
    python scripts/run_full_pipeline.py --stage post_eval  — post-opt eval only
    python scripts/run_full_pipeline.py --stage batch      — both batch runs
    python scripts/run_full_pipeline.py --stage compare    — diff only
    python scripts/run_full_pipeline.py --iterations 30    — GEPA iterations (default: 30)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import shutil
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

# Route OpenAI client through LiteLLM proxy
OPENAI_BASE = os.getenv("OPENAI_API_BASE")
LITELLM_PROXY = os.getenv("LITELLM_PROXY_URL")
LITELLM_API_KEY = os.getenv("LITELLM_PROXY_API_KEY")

if LITELLM_PROXY and LITELLM_API_KEY:
    os.environ.setdefault("OPENAI_BASE_URL", LITELLM_PROXY)
    os.environ.setdefault("OPENAI_API_KEY", LITELLM_API_KEY)
elif OPENAI_BASE:
    os.environ.setdefault("OPENAI_BASE_URL", OPENAI_BASE)

# Windows event loop fix
if sys.platform == "win32":
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# DeepEval patch for nested event loops
import nest_asyncio
nest_asyncio.apply()

import litellm

# ---- Paths and constants ----
PROMPTS_DIR = REPO_ROOT / "prompts"
OUTPUT_DIR = REPO_ROOT / "output"
INPUT_FILE = "C:\\Users\\user\\Documents\\bank_cb_link_ids.xlsx"

EXTRACTOR_STAGES = "extractor"
MATCHER_STAGES = "matcher"
CLASSIFIER_STAGES = "classifier"

# Mapping: stage → (live_prompt_file, opt_prompt_file, expanded_prompt_file)
STAGE_MAP = {
    EXTRACTOR_STAGES: {
        "live": "identifier_extractor.json",
        "opt": "identifier_extractor_opt.json",
        "expanded": "identifier_extractor_expanded.json",
    },
    MATCHER_STAGES: {
        "live": "service_matcher.json",
        "opt": "service_matcher_opt.json",
        "expanded": "service_matcher_expanded.json",
    },
    CLASSIFIER_STAGES: {
        "live": "customer_facing_classifier.json",
        "opt": "customer_facing_classifier_opt.json",
        "expanded": "customer_facing_classifier_expanded.json",
    },
}


# =====================================================================
# Stage 1: Baseline eval
# =====================================================================

def stage_baseline_eval():
    print("\n" + "=" * 70)
    print("STAGE 1: BASELINE EVAL")
    print("=" * 70)

    # We'll directly run the pipeline functions from run_eval.py logic
    # But we need to import those functions — let's use subprocess instead
    # to avoid circular imports
    cmd = [
        sys.executable, "-m", "tests.eval.run_eval",
        "--baseline",
    ]
    print(f"[pipeline] running: {' '.join(cmd)}")
    os.system(" ".join(cmd))
    print("[baseline] complete")


# =====================================================================
# Stage 2: Optimization (30 iterations × 3 stages)
# =====================================================================

def _build_expanded_prompt(path: Path) -> dict:
    """Load an expanded prompt file and convert _content_lines to content strings."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for msg in data.get("prompt", []):
        if "_content_lines" in msg:
            msg["content"] = "\n".join(msg.pop("_content_lines"))
    return data


def _save_opt_prompt(filename: str, data: dict) -> None:
    """Save optimized prompt to the _opt.json file (don't overwrite live)."""
    path = PROMPTS_DIR / filename
    # Restore _content_lines if present
    for msg in data.get("prompt", []):
        if "content" in msg:
            msg["_content_lines"] = msg.pop("content").split("\n")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[optimize] saved optimized prompt to {path}")


def optimize_single_stage(
    stage_name: str,
    iterations: int,
    metric_class,
    metrics_list: list,
    model_callback: callable,
):
    """Run optimization for a single stage."""
    from deepeval.dataset import Golden
    from deepeval.optimizer import PromptOptimizer
    from deepeval.optimizer.algorithms import GEPA
    from deepeval.prompt import Prompt, PromptMessage
    import json as _json

    stage_info = STAGE_MAP[stage_name]
    expanded_path = PROMPTS_DIR / stage_info["expanded"]
    if not expanded_path.exists():
        print(f"[{stage_name}] No expanded prompt found at {expanded_path}, skipping")
        return

    print(f"\n{'='*70}")
    print(f"OPTIMIZING {stage_name.upper()} — {iterations} iterations")
    print(f"{'='*70}")

    # Load goldens
    goldens_path = REPO_ROOT / "tests" / "goldens" / "golden_cases.json"
    with open(goldens_path, encoding="utf-8") as f:
        cases = json.load(f)

    def load_prompt_file(path: Path) -> dict:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for msg in data.get("prompt", []):
            if "_content_lines" in msg:
                msg["content"] = "\n".join(msg.pop("_content_lines"))
        return data

    # Load expanded prompt
    data = load_prompt_file(expanded_path)
    messages = []
    for msg in data["prompt"]:
        messages.append(PromptMessage(role=msg["role"], content=msg["content"]))
    prompt = Prompt(alias=f"{stage_name}_optimized", messages_template=messages)

    # Build goldens (only pass input, not expected_output, for reference-free)
    # All three stages need expected_output for proper comparison
    goldens = []
    for case in cases:
        inp = _json.dumps(case["input"], indent=2, ensure_ascii=False)
        exp_out = _json.dumps(case["expected_output"], ensure_ascii=False)
        goldens.append(Golden(input=inp, expected_output=exp_out))

    print(f"[{stage_name}] loaded {len(goldens)} golden case(s)")

    # Run optimizer
    optimizer = PromptOptimizer(
        metrics=metrics_list,
        model_callback=model_callback,
        algorithm=GEPA(iterations=iterations, pareto_size=min(3, len(goldens))),
    )

    optimized_prompt = optimizer.optimize(prompt=prompt, goldens=goldens)

    # Save optimized result
    new_messages = []
    for msg in optimized_prompt.messages_template:
        new_messages.append({"role": msg.role, "content": msg.content})
    data["prompt"] = new_messages
    data["version"] = data.get("version", 0) + 1
    data["optimized_at"] = datetime.now().isoformat()
    _save_opt_prompt(stage_info["opt"], data)

    print(f"[{stage_name}] optimization complete")


def stage_optimize(iterations: int):
    print("\n" + "=" * 70)
    print("STAGE 2: OPTIMIZATION")
    print("=" * 70)

    # Import callbacks after setting up paths
    from src.heuristics import extract_and_score, assess_match_with_identifiers, classify_customer_facing
    from tests.eval.extractor_metrics import EXTRACTOR_METRICS
    from tests.eval.matcher_metrics import MATCHER_METRICS
    from tests.eval.classifier_metrics import CLASSIFIER_METRICS
    from deepeval.dataset import Golden
    import json as _json

    # --- Extractor ---
    def extractor_callback(prompt, golden):
        inp = _json.loads(golden.input)
        page_data = inp.get("page_data", inp)
        return asyncio.run(extract_and_score(
            provider=inp.get("provider", ""),
            service_name=inp.get("service_name", ""),
            url=inp.get("url", ""),
            page_result=page_data,
            session_id="opt-extractor",
        )).__json__()

    optimize_single_stage(
        EXTRACTOR_STAGES, iterations,
        None, EXTRACTOR_METRICS, extractor_callback,
    )

    # --- Matcher ---
    def matcher_callback(prompt, golden):
        inp = _json.loads(golden.input)
        page_data = inp.get("page_data", inp)
        return asyncio.run(extract_and_score(
            provider=inp.get("provider", ""),
            service_name=inp.get("service_name", ""),
            url=inp.get("url", ""),
            page_result=page_data,
            session_id="opt-matcher",
        )).__json__()

    optimize_single_stage(
        MATCHER_STAGES, iterations,
        None, MATCHER_METRICS, matcher_callback,
    )

    # --- Classifier ---
    def classifier_callback(prompt, golden):
        inp = _json.loads(golden.input)
        page_data = inp.get("page_data", inp)
        return asyncio.run(classify_customer_facing(
            provider=inp.get("provider", ""),
            service_name=inp.get("service_name", ""),
            url=inp.get("url", ""),
            page_result=page_data,
            session_id="opt-classifier",
        )).__json__()

    optimize_single_stage(
        CLASSIFIER_STAGES, iterations,
        None, CLASSIFIER_METRICS, classifier_callback,
    )

    print("[optimize] all stages complete")


# =====================================================================
# Stage 3: Post-opt eval (swaps in opt prompts, evals, restores)
# =====================================================================

def _swap_to_opt_prompts():
    """Temporarily point the pipeline to use _opt.json prompts."""
    for stage_name, stage_info in STAGE_MAP.items():
        opt_path = PROMPTS_DIR / stage_info["opt"]
        live_path = PROMPTS_DIR / stage_info["live"]
        if opt_path.exists():
            # Copy opt → live temporarily
            shutil.copy2(opt_path, live_path)
            print(f"[swap] {stage_info['live']} → {stage_info['opt']}")


def _restore_prompts():
    """Restore from git or original copies."""
    # Run git checkout to restore originals
    import subprocess
    for stage_info in STAGE_MAP.values():
        live_path = PROMPTS_DIR / stage_info["live"]
        if live_path.exists():
            subprocess.run(
                ["git", "checkout", str(live_path.resolve())],
                capture_output=True,
            )
            print(f"[restore] restored {stage_info['live']}")


def stage_post_opt_eval():
    print("\n" + "=" * 70)
    print("STAGE 3: POST-OPT EVALUATION")
    print("=" * 70)

    if not PROMPTS_DIR.joinpath("identifier_extractor_opt.json").exists():
        print("[post_opt] No optimized prompts found. Run optimization first.")
        return

    _swap_to_opt_prompts()
    try:
        # Run eval with baseline flag
        cmd = [
            sys.executable, "-m", "tests.eval.run_eval",
            "--baseline",
        ]
        print(f"[pipeline] running: {' '.join(cmd)}")
        os.system(" ".join(cmd))
    finally:
        _restore_prompts()

    print("[post_opt] complete")


# =====================================================================
# Stage 4: Batch runs (baseline vs opt)
# =====================================================================

def read_cb_link_ids(path: str) -> list[str]:
    import openpyxl
    wb = openpyxl.load_workbook(path)
    ws = wb.active
    ids = []
    for row in ws.iter_rows(min_row=1, values_only=True):
        val = row[0]
        if val:
            ids.append(str(val).strip())
    return ids


def _copy_opt_as_live():
    """Temporarily make _opt prompts the live ones and save originals."""
    original_map = {}
    for stage_name, stage_info in STAGE_MAP.items():
        opt_path = PROMPTS_DIR / stage_info["opt"]
        live_path = PROMPTS_DIR / stage_info["live"]
        if opt_path.exists():
            # Backup live version
            original_map[stage_info["live"]] = shutil.copy2(live_path, str(live_path.parent / f"{stage_info['live']}.backup"))
            # Copy opt → live
            shutil.copy2(opt_path, live_path)
            print(f"[batch] using {stage_info['opt']} for {stage_info['live']}")
    return original_map


def _restore_live_prompts(original_map):
    """Restore original live prompts from backups."""
    for live_file, backup_path in original_map.items():
        backup_file = Path(backup_path)
        if backup_file.exists():
            shutil.copy2(backup_file, PROMPTS_DIR / live_file)
            backup_file.unlink()  # cleanup
            print(f"[batch] restored {live_file}")


def stage_batch():
    print("\n" + "=" * 70)
    print("STAGE 4: BATCH RUNS")
    print("=" * 70)

    if not os.path.isfile(INPUT_FILE):
        print(f"[batch] Input file not found: {INPUT_FILE}")
        print("[batch] Make sure C:\\Users\\user\\Documents\\bank_cb_link_ids.xlsx exists")
        return

    # ---- Baseline batch ----
    print("\n--- Baseline Batch (live prompts) ---")
    cmd = [sys.executable, "scripts/run_batch.py"]
    os.system(" ".join(cmd))

    # ---- Opt batch ----
    print("\n--- Opt Batch (optimized prompts) ---")
    # Use _opt file name as a marker
    for stage_name, stage_info in STAGE_MAP.items():
        opt_path = PROMPTS_DIR / stage_info["opt"]
        live_path = PROMPTS_DIR / stage_info["live"]
        if opt_path.exists():
            # Backup live
            backup_path = str(live_path.parent / f"{stage_info['live']}.opt_backup")
            shutil.copy2(live_path, backup_path)
            # Copy opt → live
            shutil.copy2(opt_path, live_path)
            print(f"[opt_batch] using {stage_info['opt']} for {stage_info['live']}")

    os.system(" ".join([sys.executable, "scripts/run_batch.py"]))

    # Restore
    for stage_name, stage_info in STAGE_MAP.items():
        live_path = PROMPTS_DIR / stage_info["live"]
        backup_path = PROMPTS_DIR / f"{stage_info['live']}.opt_backup"
        if backup_path.exists():
            shutil.copy2(backup_path, live_path)
            backup_path.unlink()

    print("[batch] complete")


# =====================================================================
# Stage 5: Compare
# =====================================================================

def stage_compare():
    print("\n" + "=" * 70)
    print("STAGE 5: COMPARISON")
    print("=" * 70)

    from datetime import datetime as _dt
    from pathlib import Path as _P

    OUTPUT = REPO_ROOT / "output" if "OUTPUT" not in dir() else Path("output")

    # Find latest eval baseline files
    eval_files = sorted(OUTPUT.glob("eval_baseline_*.json"))
    if not eval_files:
        print("[compare] No eval baseline files found. Run baseline + post-opt eval first.")
        return

    baseline_file = eval_files[0] if len(eval_files) == 1 else eval_files[-1]  # take most recent
    # If only one file, assume it's post-opt. Try to find both by reading files.
    # For now, if multiple exist, compare first vs last

    if len(eval_files) < 2:
        print(f"[compare] Only one eval file found: {baseline_file}")
        ts = _dt.now().strftime("%Y%m%d_%H%M%S")
        out_path = OUTPUT / f"compare_report_{ts}.json"
        report = {
            "timestamp": ts,
            "status": "single_file_only",
            "files": [str(f) for f in eval_files],
            "data": {f.stem: json.loads(f.read_text(encoding="utf-8")) for f in eval_files},
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"[compare] saved to {out_path}")
        return

    baseline_data = json.loads(eval_files[0].read_text(encoding="utf-8"))
    post_data = json.loads(eval_files[-1].read_text(encoding="utf-8"))

    # Extract scores
    baseline_scores = {}
    for m, v in baseline_data.get("metric_scores", {}).items():
        baseline_scores[m] = v.get("mean", 0)

    post_scores = {}
    for m, v in post_data.get("metric_scores", {}).items():
        post_scores[m] = v.get("mean", 0)

    comparisons = {}
    for metric in set(baseline_scores.keys()) | set(post_scores.keys()):
        b = baseline_scores.get(metric, 0)
        p = post_scores.get(metric, 0)
        delta = p - b
        comparisons[metric] = {
            "baseline": b,
            "post_opt": p,
            "delta": delta,
            "direction": "improved" if delta > 0 else ("worse" if delta < 0 else "no_change"),
        }

    report = {
        "timestamp": _dt.now().isoformat(),
        "baseline_file": str(baseline_file),
        "post_opt_file": str(eval_files[-1]),
        "baseline_pass_rate": baseline_data.get("overall_pass_rate", 0),
        "post_opt_pass_rate": post_data.get("overall_pass_rate", 0),
        "overall_delta": (post_data.get("overall_pass_rate", 0) - baseline_data.get("overall_pass_rate", 0)),
        "metric_comparisons": comparisons,
        "metric_details": {
            "baseline": baseline_data.get("metric_scores", {}),
            "post_opt": post_data.get("metric_scores", {}),
        },
    }

    ts = _dt.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT / f"compare_report_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"Baseline pass rate: {baseline_data.get('overall_pass_rate', 0):.1%}")
    print(f"Post-opt pass rate: {post_data.get('overall_pass_rate', 0):.1%}")
    print(f"Overall delta: {report['overall_delta']:+.1%}")
    print()

    for metric, comp in sorted(comparisons.items(), key=lambda x: x[1].get("delta", 0) * -1):
        direction = "▲" if comp["direction"] == "improved" else ("▼" if comp["direction"] == "worse" else "→")
        print(f"  {direction} {metric}: {comp['baseline']:.3f} → {comp['post_opt']:.3f} ({comp['delta']:+.3f})")

    print(f"\n[compare] full report saved to {out_path}")


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Full eval → optimize → batch pipeline")
    parser.add_argument(
        "--stage",
        choices=["baseline", "optimize", "post_eval", "batch", "compare", "all"],
        default="all",
    )
    parser.add_argument("--iterations", type=int, default=30, help="GEPA iterations per stage (default: 30)")
    args = parser.parse_args()

    if args.stage in ("baseline", "all"):
        stage_baseline_eval()

    if args.stage in ("optimize", "all"):
        stage_optimize(args.iterations)

    if args.stage in ("post_eval", "all"):
        stage_post_opt_eval()

    if args.stage in ("batch", "all"):
        stage_batch()

    if args.stage in ("compare", "all"):
        stage_compare()

    print("\n[full_pipeline] done")


if __name__ == "__main__":
    main()
