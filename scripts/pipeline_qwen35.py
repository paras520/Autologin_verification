"""Unified pipeline: baseline -> optimize (20 iters x 3 stages) -> post_eval -> batch -> compare
All LLM calls go through Qwen 3.5 with full logging to output/pipeline_logs/
Usage: python scripts/pipeline_qwen35.py --stage all --iterations 20
       python scripts/pipeline_qwen35.py --stage baseline
       python scripts/pipeline_qwen35.py --stage optimize
       python scripts/pipeline_qwen35.py --stage post_eval
       python scripts/pipeline_qwen35.py --stage batch
       python scripts/pipeline_qwen35.py --stage compare
"""
from __future__ import annotations
import argparse, json, logging, os, shutil, subprocess, sys, time
from datetime import datetime
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent / "output" / "pipeline_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
fh = logging.FileHandler(LOG_DIR / "pipeline.log", mode="a", encoding="utf-8")
fsh = logging.StreamHandler(sys.stdout)
logger = logging.getLogger("pipeline")
logger.setLevel(logging.DEBUG)
logger.handlers.clear()
logger.addHandler(fh)
logger.addHandler(fsh)

MAX_TOKENS = 48000
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
QWEN_MODEL = "Qwen/Qwen3.5-9B"
QWEN_BASE = "https://il3e3qpwnnpinq-8000.proxy.runpod.net/v1"
QWEN_API_KEY = "DIRO@123"
PROMPTS_DIR = REPO_ROOT / "prompts"
OUTPUT_DIR = REPO_ROOT / "output"
EXCEL_PATH = r"C:\Users\user\Documents\bank_cb_link_ids.xlsx"
LOG_DIR = REPO_ROOT / "output" / "pipeline_logs"

STAGE_MAP = {
    "extractor": dict(live="identifier_extractor.json", opt="identifier_extractor_opt.json", expanded="identifier_extractor_expanded.json"),
    "matcher": dict(live="service_matcher.json", opt="service_matcher_opt.json", expanded="service_matcher_expanded.json"),
    "classifier": dict(live="customer_facing_classifier.json", opt="customer_facing_classifier_opt.json", expanded="customer_facing_classifier_expanded.json"),
}

def log_api_call(stage, stage_name, prompt, resp_dict, elapsed_ms, tokens, error=""):
    """Log every LLM API call with full context."""
    entry = dict(ts=datetime.now().isoformat(), stage=stage, stage_name=stage_name, model=QWEN_MODEL,
                 prompt=prompt[:4000], response=resp_dict, elapsed_ms=elapsed_ms, tokens=tokens, error=error)
    log_path = LOG_DIR / f"api_calls_{stage}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logger.info(f"[API] stage={stage_name} model={QWEN_MODEL} time={elapsed_ms:.0f}ms tokens={tokens} err={'!!' if error else 'ok'}")

def call_qwen(prompt, max_tokens=MAX_TOKENS, temperature=0.1, stage="pipeline", stage_name="pipeline"):
    """Call local Qwen 3.5 via OpenAI SDK. Qwen outputs to reasoning field."""
    import warnings
    from openai import OpenAI
    start = time.time()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            client = OpenAI(base_url=QWEN_BASE, api_key=QWEN_API_KEY)
            resp = client.chat.completions.create(model=QWEN_MODEL, messages=[{"role": "user", "content": prompt}],
                                                   max_tokens=max_tokens, temperature=temperature, timeout=300)
        choice = resp.choices[0].message
        elapsed = (time.time() - start) * 1000
        tokens = resp.usage.total_tokens if resp.usage else 0
        reasoning = choice.reasoning or ""
        content = choice.content or ""
        combined = reasoning or content
        resp_dict = dict(reasoning=reasoning[:2000], content=content[:2000], finish_reason=resp.choices[0].finish_reason,
                         tokens=tokens, elapsed_ms=elapsed)
        log_api_call(stage, stage_name, prompt[:2000], resp_dict, elapsed, tokens)
        combined = combined.strip()
        if combined.startswith("```"):
            lines = combined.split("\n")
            if lines[0].startswith("```"): lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"): lines = lines[:-1]
            combined = "\n".join(lines)
        try:
            return dict(data=json.loads(combined), parse_error=False)
        except (json.JSONDecodeError, TypeError):
            return dict(data=combined[:2000], parse_error=True)
    except Exception as e:
        elapsed = (time.time() - start) * 1000
        error = str(e)
        log_api_call(stage, stage_name, prompt[:2000], dict(error=error), elapsed, 0, error=error)
        logger.info(f"[QWEN ERROR] {error}")
        return dict(data="", parse_error=True, error=error)

def load_prompt(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for msg in data.get("prompt", []):
        if "_content_lines" in msg:
            msg["content"] = "\n".join(msg.pop("_content_lines"))
    return data

def save_prompt(path, data):
    for msg in data.get("prompt", []):
        if "content" in msg:
            msg["_content_lines"] = msg.pop("content").split("\n")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_goldens():
    with open(REPO_ROOT / "tests" / "goldens" / "golden_cases.json", encoding="utf-8") as f:
        return json.load(f)

def swap_prompts(for_optimize):
    backups = {}
    for name, info in STAGE_MAP.items():
        old_path = PROMPTS_DIR / info["live"]
        new_path = PROMPTS_DIR / info["opt"] if for_optimize else PROMPTS_DIR / info["expanded"]
        if (new_path if for_optimize else old_path).exists():
            bk = PROMPTS_DIR / (info["live"] + ".bak")
            shutil.copy2(old_path, bk)
            backups[info["live"]] = bk
            if for_optimize:
                shutil.copy2(new_path, old_path)
            else:
                shutil.copy2(old_path, new_path)
    return backups

def restore_prompts(backups):
    for live_file, backup in backups.items():
        if backup.exists():
            shutil.copy2(backup, PROMPTS_DIR / live_file)
            backup.unlink()

# ================================================================
# STAGE 1: Baseline eval
# ================================================================

def run_subprocess(cmd: list, label: str):
    """Run a command in REPO_ROOT with UTF-8 output, stream to console, log result."""
    logger.info(f"  Running: {' '.join(cmd)}")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"  # prevent cp1252 Unicode crash on Windows
    # DeepEval per-attempt / per-task timeout overrides (Qwen reasoning is slow,
    # default 60-180s per-attempt cap trips with TimeoutError -> RetryError -> "no candidate prompt").
    env.setdefault("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE", "300")  # 5 min / attempt
    env.setdefault("DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE", "1800")    # 30 min outer
    env.setdefault("DEEPEVAL_RETRY_MAX_ATTEMPTS", "5")                       # ride out 5xx from Qwen GPU
    env.setdefault("DEEPEVAL_RETRY_INITIAL_SECONDS", "5")
    env.setdefault("DEEPEVAL_RETRY_CAP_SECONDS", "60")
    env.setdefault("DEEPEVAL_RETRY_EXP_BASE", "2")
    result = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        text=True,
        capture_output=False,  # stream stdout/stderr directly to console
    )
    if result.returncode != 0:
        logger.info(f"  [{label}] exited with code {result.returncode}")
    else:
        logger.info(f"  [{label}] completed OK")
    return result.returncode


def baseline_eval():
    logger.info("=" * 60)
    logger.info("STAGE 1: BASELINE EVALUATION")
    logger.info("=" * 60)
    run_subprocess([sys.executable, '-m', 'tests.eval.run_eval', '--baseline'], "baseline_eval")
    import glob
    files = sorted(glob.glob(str(OUTPUT_DIR / "eval_baseline_*.json")))
    logger.info(f"Found {len(files)} eval_baseline files")
    for f in files:
        logger.info(f"  {Path(f).name}")

# ================================================================
# STAGE 2: Optimization (20 GEPA iterations x 3 stages)
# Delegates to existing tests/eval/optimize_prompts.py
# ================================================================

def optimize_stage(stage_name, iterations=20):
    """Delegate to tests/eval/optimize_prompts.py for a single stage."""
    logger.info("=" * 60)
    logger.info(f"STAGE 2.{stage_name}: OPTIMIZE ({iterations} iters)")
    logger.info("=" * 60)
    run_subprocess(
        [sys.executable, '-m', 'tests.eval.optimize_prompts',
         '--stage', stage_name, '--iterations', str(iterations), '--track-metrics'],
        f"optimize_{stage_name}",
    )

# ================================================================
# STAGE 3: Post-eval
# ================================================================

def post_eval():
    logger.info("=" * 60)
    logger.info("STAGE 3: POST-OPTIMIZATION EVALUATION")
    logger.info("=" * 60)
    backups = swap_prompts(for_optimize=True)
    try:
        run_subprocess([sys.executable, '-m', 'tests.eval.run_eval', '--baseline'], "post_eval")
    finally:
        restore_prompts(backups)
    logger.info("Post-eval complete")

# ================================================================
# STAGE 4: Batch
# ================================================================

def batch_run():
    logger.info("=" * 60)
    logger.info("STAGE 4: BATCH PROCESSING")
    logger.info("=" * 60)
    if not os.path.isfile(EXCEL_PATH):
        logger.error(f"Excel file not found: {EXCEL_PATH}")
        return

    opt_exists = any((PROMPTS_DIR / info["opt"]).exists() for info in STAGE_MAP.values())
    if not opt_exists:
        logger.error("No optimized prompts found. Run optimize first.")
        return

    # Baseline batch (live prompts)
    logger.info("Phase 1: Baseline batch (current prompts)")
    run_subprocess([sys.executable, 'scripts/run_batch.py'], "batch_baseline")

    # Opt batch (optimized prompts)
    backups = swap_prompts(for_optimize=True)
    try:
        logger.info("Phase 2: Optimized batch")
        run_subprocess([sys.executable, 'scripts/run_batch.py'], "batch_optimized")
    finally:
        restore_prompts(backups)
    logger.info("Batch run complete")

# ================================================================
# STAGE 5: Compare
# ================================================================

def compare():
    logger.info("=" * 60)
    logger.info("STAGE 5: COMPARISON")
    logger.info("=" * 60)
    import glob
    baseline_files = sorted(glob.glob(str(OUTPUT_DIR / "eval_baseline_*.json")))
    opt_files = sorted(glob.glob(str(OUTPUT_DIR / "eval_optimized_*.json")))
    baseline_results = sorted(glob.glob(str(OUTPUT_DIR / "results_*.jsonl")))
    opt_results = sorted(glob.glob(str(OUTPUT_DIR / "results_opt_*.jsonl")))

    logger.info(f"Baseline evals:  {len(baseline_files)}")
    logger.info(f"Opt evals:       {len(opt_files)}")
    logger.info(f"Baseline results:{len(baseline_results)}")
    logger.info(f"Opt results:     {len(opt_results)}")

    # Compare batch results
    if baseline_results and opt_results:
        br = baseline_results[-1]
        or_ = opt_results[-1]
        logger.info(f"\nComparing latest batch results:")
        logger.info(f"  Baseline: {br.split('/')[-1]}")
        logger.info(f"  Optimized: {or_.split('/')[-1]}")

        def count_lines(f):
            with open(f) as fh:
                return sum(1 for _ in fh)
        logger.info(f"  Baseline lines: {count_lines(br)}")
        logger.info(f"  Optimized lines: {count_lines(or_)}")

    # Compare eval scores
    if baseline_files:
        logger.info("\nBaseline eval scores:")
        with open(baseline_files[-1]) as f:
            bs = json.load(f)
        for k, v in bs.items():
            logger.info(f"  {k}: {v}")

    if opt_files:
        logger.info("\nOptimized eval scores:")
        with open(opt_files[-1]) as f:
            os = json.load(f)
        for k, v in opt_files.items():
            logger.info(f"  {k}: {v}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all",
                        choices=["all", "baseline", "optimize", "post_eval", "batch", "compare"])
    parser.add_argument("--iterations", type=int, default=20, help="GEPA iterations per stage")
    args = parser.parse_args()

    if args.stage == "all":
        baseline_eval()
        for name in STAGE_MAP:
            optimize_stage(name, args.iterations)
        post_eval()
        batch_run()
        compare()
    elif args.stage == "baseline":
        baseline_eval()
    elif args.stage == "optimize":
        for name in STAGE_MAP:
            optimize_stage(name, args.iterations)
    elif args.stage == "post_eval":
        post_eval()
    elif args.stage == "batch":
        batch_run()
    elif args.stage == "compare":
        compare()
