"""Resilient GEPA orchestrator.

Runs the 3 GEPA stages (extractor, matcher, classifier) with these guarantees:

  1. PRE-FLIGHT: refuses to launch if scripts/preflight_gepa.py fails
  2. PER-STAGE SUBPROCESS: each stage is its own python -m process
                            (one stage's crash never poisons the others)
  3. AUTO-RESTART: each stage retried up to MAX_STAGE_RETRIES times on failure
                   with a cooldown between retries
  4. RESUME: a stage that already completed successfully is skipped
             (creates output/gepa_checkpoints/<stage>.done marker)
  5. PROGRESS LOG: continuous heartbeat -> output/gepa_checkpoints/run.log

Usage:
  python -m scripts.run_gepa_resilient                       # full run, all 3 stages, 20 iters each
  python -m scripts.run_gepa_resilient --iterations 20
  python -m scripts.run_gepa_resilient --reset extractor     # force re-run extractor
  python -m scripts.run_gepa_resilient --reset all           # force re-run everything
  python -m scripts.run_gepa_resilient --skip-preflight      # skip pre-flight checks
  python -m scripts.run_gepa_resilient --max-retries 5       # retry each failing stage up to 5 times
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = REPO_ROOT / "output" / "gepa_checkpoints"
LOG_DIR = REPO_ROOT / "output" / "pipeline_logs"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

STAGES = ["extractor", "matcher", "classifier"]

# DeepEval & retry settings used by every child subprocess
DEEPEVAL_ENV = {
    "DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE": "600",
    "DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE": "3600",
    "DEEPEVAL_RETRY_MAX_ATTEMPTS": "8",
    "DEEPEVAL_RETRY_INITIAL_SECONDS": "5",
    "DEEPEVAL_RETRY_CAP_SECONDS": "120",
    "DEEPEVAL_RETRY_EXP_BASE": "2",
    "OPT_CALLBACK_CONCURRENCY": "2",
    "EVAL_CONCURRENCY": "2",
    "EVAL_JUDGE_CONCURRENCY": "2",
    "QWEN_MAX_ATTEMPTS": "8",
    "QWEN_BACKOFF_BASE": "5",
    "QWEN_BACKOFF_CAP": "120",
    # Early-stop controls (read by tests/eval/optimize_prompts.py)
    "GEPA_EARLY_STOP_REJECTS": os.getenv("GEPA_EARLY_STOP_REJECTS", "3"),
    "GEPA_EARLY_STOP_PERFECT": os.getenv("GEPA_EARLY_STOP_PERFECT", "0.99"),
    "GEPA_EARLY_STOP_NO_IMPROVE": os.getenv("GEPA_EARLY_STOP_NO_IMPROVE", "5"),
    "GEPA_EARLY_STOP_EPS": os.getenv("GEPA_EARLY_STOP_EPS", "0.005"),
    "PYTHONIOENCODING": "utf-8",
    "PYTHONUNBUFFERED": "1",
}


def _setup_logger() -> logging.Logger:
    log = logging.getLogger("gepa_orchestrator")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fh = logging.FileHandler(CHECKPOINTS_DIR / "run.log", mode="a", encoding="utf-8")
    sh = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)
    return log


logger = _setup_logger()


def _stage_marker(stage: str) -> Path:
    return CHECKPOINTS_DIR / f"{stage}.done"


def _is_stage_done(stage: str) -> bool:
    return _stage_marker(stage).exists()


def _mark_stage_done(stage: str, iterations: int, attempt: int, elapsed_s: float) -> None:
    payload = {
        "stage": stage,
        "iterations": iterations,
        "attempt": attempt,
        "elapsed_seconds": round(elapsed_s, 1),
        "completed_at": datetime.now().isoformat(),
    }
    _stage_marker(stage).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info(f"[{stage}] marker written -> {_stage_marker(stage)}")


def _reset_stage(stage: str) -> None:
    p = _stage_marker(stage)
    if p.exists():
        p.unlink()
        logger.info(f"[{stage}] reset (marker deleted)")


def _run_preflight() -> bool:
    logger.info("=" * 70)
    logger.info("STEP 0: PRE-FLIGHT CHECKS")
    logger.info("=" * 70)
    env = os.environ.copy()
    env.update(DEEPEVAL_ENV)
    result = subprocess.run(
        [sys.executable, "-m", "scripts.preflight_gepa"],
        cwd=str(REPO_ROOT),
        env=env,
        text=True,
        capture_output=False,
    )
    ok = result.returncode == 0
    logger.info(f"Pre-flight {'PASSED' if ok else 'FAILED'} (exit={result.returncode})")
    return ok


def _run_stage_once(stage: str, iterations: int, attempt: int) -> tuple[bool, float]:
    """Run one stage of GEPA in a subprocess. Returns (success, elapsed_seconds)."""
    log_path = LOG_DIR / f"gepa_{stage}_attempt{attempt}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger.info(f"[{stage}] attempt {attempt}: launching, log -> {log_path}")
    env = os.environ.copy()
    env.update(DEEPEVAL_ENV)
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "tests.eval.optimize_prompts",
        "--stage",
        stage,
        "--iterations",
        str(iterations),
        "--track-metrics",
    ]
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as logfh:
        result = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=logfh,
            stderr=subprocess.STDOUT,
            text=True,
        )
    dt = time.time() - t0
    ok = result.returncode == 0
    logger.info(
        f"[{stage}] attempt {attempt} {'OK' if ok else 'FAILED'} "
        f"(exit={result.returncode}, elapsed={dt/60:.1f}min)"
    )
    if not ok:
        # Tail of log for quick diagnosis
        try:
            with open(log_path, encoding="utf-8") as f:
                tail = f.readlines()[-30:]
            logger.warning(f"[{stage}] last 30 lines of log:")
            for line in tail:
                logger.warning(f"    {line.rstrip()}")
        except Exception:
            pass
    return ok, dt


def _run_stage(stage: str, iterations: int, max_retries: int, cooldown: int) -> bool:
    if _is_stage_done(stage):
        marker = json.loads(_stage_marker(stage).read_text(encoding="utf-8"))
        logger.info(f"[{stage}] SKIP (already done at {marker.get('completed_at')})")
        return True

    logger.info("=" * 70)
    logger.info(f"STAGE: {stage.upper()}  (iterations={iterations}, max_retries={max_retries})")
    logger.info("=" * 70)

    for attempt in range(1, max_retries + 1):
        ok, elapsed = _run_stage_once(stage, iterations, attempt)
        if ok:
            _mark_stage_done(stage, iterations, attempt, elapsed)
            return True
        if attempt < max_retries:
            logger.warning(
                f"[{stage}] attempt {attempt}/{max_retries} failed; cooling down {cooldown}s before retry"
            )
            time.sleep(cooldown)
    logger.error(f"[{stage}] FAILED after {max_retries} attempts. Moving on (other stages still run).")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Resilient GEPA orchestrator")
    parser.add_argument("--iterations", type=int, default=20, help="GEPA iterations per stage (default 20)")
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries per stage on failure (default 3)",
    )
    parser.add_argument(
        "--cooldown",
        type=int,
        default=60,
        help="Cooldown seconds between stage retries (default 60)",
    )
    parser.add_argument(
        "--reset",
        choices=STAGES + ["all"],
        help="Wipe completion marker so the stage re-runs",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip pre-flight checks (NOT recommended)",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=STAGES,
        default=STAGES,
        help="Subset of stages to run",
    )
    args = parser.parse_args()

    logger.info("#" * 70)
    logger.info(f"# GEPA RESILIENT ORCHESTRATOR  -- {datetime.now().isoformat()}")
    logger.info("#" * 70)
    logger.info(
        f"iterations={args.iterations}, max_retries={args.max_retries}, "
        f"cooldown={args.cooldown}s, stages={args.stages}, "
        f"skip_preflight={args.skip_preflight}, reset={args.reset}"
    )

    if args.reset:
        targets = STAGES if args.reset == "all" else [args.reset]
        for s in targets:
            _reset_stage(s)

    if not args.skip_preflight:
        if not _run_preflight():
            logger.error("ABORT: pre-flight failed. Use --skip-preflight to override.")
            return 2

    overall = {"start": datetime.now().isoformat(), "stages": {}}
    t_start = time.time()
    successes: list[str] = []
    failures: list[str] = []
    for stage in args.stages:
        ok = _run_stage(stage, args.iterations, args.max_retries, args.cooldown)
        overall["stages"][stage] = "ok" if ok else "failed"
        (successes if ok else failures).append(stage)

    overall["elapsed_minutes"] = round((time.time() - t_start) / 60, 1)
    overall["end"] = datetime.now().isoformat()
    summary_path = CHECKPOINTS_DIR / "run_summary.json"
    summary_path.write_text(json.dumps(overall, indent=2), encoding="utf-8")

    logger.info("#" * 70)
    logger.info("# FINAL SUMMARY")
    logger.info("#" * 70)
    logger.info(f"  total elapsed: {overall['elapsed_minutes']} min")
    logger.info(f"  successes: {successes}")
    logger.info(f"  failures:  {failures}")
    logger.info(f"  summary -> {summary_path}")

    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
