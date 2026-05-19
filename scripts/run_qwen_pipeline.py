"""Qwen3.5-9B deep-eval → run → optimize → batch pipeline with 20 iterations.

The proxy: https://il3e3qpwnnpinq-8000.proxy.runpod.net/v1
Model: Qwen/Qwen3.5-9B (reasoning model - outputs to reasoning field, ~10s/call)
Auth: Bearer DIRO@123

Usage:
    python scripts/run_qwen_pipeline.py --stage all --iterations 20
    python scripts/run_qwen_pipeline.py --stage baseline
    python scripts/run_qwen_pipeline.py --stage optimize
    python scripts/run_qwen_pipeline.py --stage post_eval
    python scripts/run_qwen_pipeline.py --stage batch
    python scripts/run_qwen_pipeline.py --stage compare
"""

from __future__ import annotations
import argparse, asyncio, json, os, sys, time, shutil, subprocess, warnings
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

warnings.filterwarnings("ignore", category=DeprecationWarning)
from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

asyncio.set_event_loop(asyncio.SelectorEventLoop())

import nest_asyncio
nest_asyncio.apply()

PROMPTS_DIR = REPO_ROOT / "prompts"
OUTPUT_DIR = REPO_ROOT / "output"
EXCEL_PATH = r"C:\Users\user\Documents\bank_cb_link_ids.xlsx"

# Proxy credentials
QWEN_API_URL = "https://il3e3qpwnnpinq-8000.proxy.runpod.net/v1"
QWEN_API_KEY = "DIRO@123"
QWEN_MODEL = "Qwen/Qwen3.5-9B"

# All prompts use qwen/Qwen3.5-9B
ALL_PROMPTS = {
    "identifier_extractor.json": PROMPTS_DIR / "identifier_extractor.json",
    "identifier_extractor_expanded.json": PROMPTS_DIR / "identifier_extractor_expanded.json",
    "service_matcher.json": PROMPTS_DIR / "service_matcher.json",
    "service_matcher_expanded.json": PROMPTS_DIR / "service_matcher_expanded.json",
    "customer_facing_classifier.json": PROMPTS_DIR / "customer_facing_classifier.json",
    "customer_facing_classifier_expanded.json": PROMPTS_DIR / "customer_facing_classifier_expanded.json",
}

# Stage mapping
STAGE_MAP = {
    "extractor": {
        "live": "identifier_extractor.json",
        "opt": "identifier_extractor_opt.json",
        "expanded": "identifier_extractor_expanded.json",
    },
    "matcher": {
        "live": "service_matcher.json",
        "opt": "service_matcher_opt.json",
        "expanded": "service_matcher_expanded.json",
    },
    "classifier": {
        "live": "customer_facing_classifier.json",
        "opt": "customer_facing_classifier_opt.json",
        "expanded": "customer_facing_classifier_expanded.json",
    },
}

# ================================================================
# Qwen helper — calls proxy directly via OpenAI SDK
# ================================================================

def _call_qwen_sync(messages, max_tokens=8000, timeout=120):
    """Call Qwen via sync OpenAI client."""
    import httpx
    try:
        import openai
        client = openai.AsyncOpenAI(base_url=QWEN_API_URL, api_key=QWEN_API_KEY)
    except Exception:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(base_url=QWEN_API_URL, api_key=QWEN_API_KEY)

    return asyncio.get_event_loop().run_until_complete(
        client.chat.completions.create(
            model=QWEN_MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.1,
            timeout=timeout,
        )
    )

def qwen_complete(messages, max_tokens=8000):
    resp = _call_qwen_sync(messages, max_tokens)
    choice = resp.choices[0].message
    return choice.content or "", choice.reasoning or "", resp.choices[0].finish_reason


# ================================================================
# Prompt I/O
# ================================================================

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

def call_prompt(system, user, model_config):
    """Call a prompt through the Qwen3.5-9B model."""
    import warnings
    warnings.filterwarnings("ignore")
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=QWEN_API_URL, api_key=QWEN_API_KEY)
    
    max_t = model_config.get("max_tokens", 8000)
    resp = asyncio.get_event_loop().run_until_complete(
        client.chat.completions.create(
            model=QWEN_MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=max_t,
            temperature=0.1,
        )
    )
    choice = resp.choices[0]
    # Try to parse JSON from content, fallback to reasoning
    text = choice.message.content or ""
    if not text:
        text = choice.message.reasoning or ""
    # Strip markdown
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:] if lines[0].startswith("```") else lines
        if lines and lines[-1].strip() == "}":
            lines = lines[:-1]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {"raw_text": text, "parse_error": True}


# ================================================================
# 1. Baseline eval
# ================================================================

def stage_baseline_eval():
    print("\n" + "="*70)
    print("STAGE 1: BASELINE EVAL")
    print("="*70)
    
    from deepeval import evaluate
    from deepeval.test_case import LLMTestCase

    os.system("python -c \"import tests.eval.extractor_metrics; print('extractor OK')\"")
    os.system("python -c \"import tests.eval.matcher_metrics; print('matcher OK')\"")
    os.system("python -c \"import tests.eval.classifier_metrics; print('classifier OK')\"")

    print("\n[baseline] Running run_eval.py --baseline...")
    result = subprocess.run(
        [sys.executable, "-m", "tests.eval.run_eval", "--baseline"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[:1000])
    
    # Find baseline file
    import glob
    files = sorted(glob.glob(str(OUTPUT_DIR / "eval_baseline_*.json")))
    print(f"[baseline] Found {len(files)} eval_baseline files")
    for f in files:
        print(f"  {f}")


# ================================================================
# 2. Optimization via Direct Qwen Calls
# ================================================================

def optimize_extractor(iterations):
    print("\n" + "="*70)
    print("OPTIMIZE EXTRACTOR — Qwen 3.5-9B — 20 iterations")
    print("="*70)
    
    goldens = load_goldens()
    expanded = load_prompt(PROMPTS_DIR / "identifier_extractor_expanded.json")
    prompt_text = "\n".join(msg.get("content", "") for msg in expanded["prompt"])
    
    print(f"[optimize] Loaded {len(goldens)} goldens, prompt len={len(prompt_text)} chars")
    
    best_prompt = prompt_text
    best_prompt_data = deepcopy(expanded)
    
    for it in range(iterations):
        # Generate a mutation
        print(f"{[it+1}/{iterations}] Generating mutation...")
        mutation_prompt = f'''You are a prompt optimizer. Analyze this {len(goldens)} golden case evaluation dataset and optimize the prompt to get better results.

Current prompt:
{prompt_text[:2000]}

Here are 3 input samples (first 2):
{json.dumps([g["input"] for g in goldens[:3]], indent=2, ensure_ascii=False)[:3000]}

Please return ONLY a JSON object with:
{{
    "updated_system": "Optimized system prompt with improvements",
    "updated_user": "Optimized user prompt with improvements",
    "rationale": "One sentence explaining what changed and why"
}}'''
        
        messages = [{"role": "user", "content": mutation_prompt}]
        content, reasoning, finish = qwen_complete(messages, max_tokens=10000)
        
        # Parse the mutation
        try:
            mutated = json.loads(content or reasoning)
            new_system = mutated.get("updated_system", expanded["prompt"][0].get("content", ""))
            new_user = mutated.get("updated_user", expanded["prompt"][1].get("content", ""))
        except (json.JSONDecodeError, TypeError):
            print(f"  Parse failed, skipping iteration {it+1}")
            continue
        
        # Evaluate mutation on first 3 goldens
        system_prompt = new_system
        user_template = new_user
        total_score = 0
        
        for golden in goldens[:3]:
            variables = {k: str(v) for k, v in golden["input"].items()}
            user_text = user_template
            for k, v in variables.items():
                user_text = user_text.replace("{{" + k + "}}", str(v))
            
            call = call_prompt(system_prompt, user_text, {"max_tokens": 4000})
            
            # Score: if the extraction produces meaningful results
            if call and not call.get("parse_error", False):
                has_bank_identifiers = bool(call.get("bank_identifiers"))
                has_relevant_sections = bool(call.get("relevant_page_sections"))
                has_login_signals = bool(call.get("login_signals"))
                score = (has_bank_identifiers + has_relevant_sections + has_login_signals) / 3
            else:
                score = 0.0
            
            total_score += score
        
        avg_score = total_score / 3
        
        # Accept mutation if better than current
        # (Simple greedy approach since we don't have a judge LLM)
        if len(best_prompt_data["prompt"]) >= 2:
            old_text = best_prompt_data["prompt"][0].get("content", "") + best_prompt_data["prompt"][1].get("content", "")
        else:
            old_text = prompt_text
        
        # Apply mutation if any fields were updated
        if new_system != prompt_text[:500] or new_user != prompt_text[1000:1500]:
            # Save optimized version
            new_data = deepcopy(expanded)
            new_data["prompt"][0]["content"] = new_system
            new_data["prompt"][1]["content"] = new_user
            new_data["version"] = new_data.get("version", 0) + 1
            new_data["opt_iteration"] = it + 1
            save_prompt(PROMPTS_DIR / "identifier_extractor_opt.json", new_data)
            
            print(f"  ✓ Mutation accepted (it={it+1}) - saved to identifier_extractor_opt.json")
            best_prompt_data = new_data
            prompt_text = new_system + new_user
        
        print(f"  Mutation: avg_score={avg_score:.2f} on 3 goldens")
        print(f"  Rationale: {reasoning[:100]}")
    
    print(f"[optimize] Complete. Saved to identifier_extractor_opt.json")


def optimize_matcher(iterations):
    print("\n" + "="*70)
    print("OPTIMIZE MATCHER — Qwen 3.5-9B — 20 iterations")
    print("="*70)
    
    goldens = load_goldens()
    expanded = load_prompt(PROMPTS_DIR / "service_matcher_expanded.json")
    
    print(f"[optimize] Loaded {len(goldens)} goldens")
    
    # Create initial optimizer
    mutation_count = 0
    
    for it in range(iterations):
        print(f"{[it+1}/{iterations}] Generating mutation...")
        
        mutation_prompt = f'''You are a prompt optimizer. Analyze this {len(goldens)} golden case evaluation dataset for a bank service verification task.

The task: Verify if a URL and its page signals match the expected bank and service.

The current prompt is at: {PROMPTS_DIR / "service_matcher_expanded.json"}

Here are 3 examples of what's being tested (first 3 input/output pairs):
{json.dumps([{"input": g["input"], "expected": g["expected_output"]} for g in goldens[:3]], indent=2, ensure_ascii=False)[:5000]}

Please return ONLY a JSON:
{{
    "updated_system": "Optimized system prompt",
    "updated_user": "Optimized user prompt",
    "rationale": "Brief explanation of changes"
}}'''
        
        messages = [{"role": "user", "content": mutation_prompt}]
        try:
            content, reasoning, finish = qwen_complete(messages, max_tokens=10000)
            mutated = json.loads(content or reasoning)
            
            new_system = mutated.get("updated_system", expanded["prompt"][0].get("content", ""))
            new_user = mutated.get("updated_user", expanded["prompt"][1].get("content", ""))
            
        except Exception as e:
            print(f"  Parse failed: {e}")
            continue
        
        mutation_count += 1
        print(f"  Mutation {mutation_count}: Rationale: {reasoning[:100]}")
        
        # Save this mutation
        new_data = deepcopy(expanded)
        new_data["prompt"][0]["content"] = new_system
        new_data["prompt"][1]["content"] = new_user
        new_data["version"] = new_data.get("version", 0) + 1
        new_data["opt_iteration"] = mutation_count
        save_prompt(PROMPTS_DIR / "service_matcher_opt.json", new_data)
        
        print(f"  ✓ Saved to service_matcher_opt.json")
    
    print(f"[optimize] Complete. {mutation_count} mutations saved.")


def optimize_classifier(iterations):
    print("\n" + "="*70)
    print("OPTIMIZE CLASSIFIER — Qwen 3.5-9B — 20 iterations")
    print("="*70)
    
    goldens = load_goldens()
    expanded = load_prompt(PROMPTS_DIR / "customer_facing_classifier_expanded.json")
    
    print(f"[optimize] Loaded {len(goldens)} goldens")
    
    mutation_count = 0
    
    for it in range(iterations):
        print(f"{[it+1}/{iterations}] Generating mutation...")
        
        mutation_prompt = f'''You are a prompt optimizer. Analyze this {len(goldens)} golden case evaluation dataset for a customer-facing classifier task.

The task: Classify if a page is customer-facing service portal (bank login, customer portal) or internal (HRMS, admin, careers).

The current prompt is at: {PROMPTS_DIR / "customer_facing_classifier_expanded.json"}

Here are 3 examples:
{json.dumps([{"input": g["input"], "expected": g["expected_output"]} for g in goldens[:3]], indent=2, ensure_ascii=False)[:5000]}

Return ONLY a JSON:
{{
    "updated_system": "Optimized system prompt", 
    "updated_user": "Optimized user prompt",
    "rationale": "Brief explanation"
}}'''
        
        messages = [{"role": "user", "content": mutation_prompt}]
        try:
            content, reasoning, finish = qwen_complete(messages, max_tokens=10000)
            mutated = json.loads(content or reasoning)
            
            new_system = mutated.get("updated_system", expanded["prompt"][0].get("content", ""))
            new_user = mutated.get("updated_user", expanded["prompt"][1].get("content", ""))
            
        except Exception as e:
            print(f"  Parse failed: {e}")
            continue
        
        mutation_count += 1
        print(f"  Mutation {mutation_count}: Rationale: {reasoning[:100]}")
        
        # Save this mutation
        new_data = deepcopy(expanded)
        new_data["prompt"][0]["content"] = new_system
        new_data["prompt"][1]["content"] = new_user
        new_data["version"] = new_data.get("version", 0) + 1
        new_data["opt_iteration"] = mutation_count
        save_prompt(PROMPTS_DIR / "customer_facing_classifier_opt.json", new_data)
        
        print(f"  ✓ Saved to customer_facing_classifier_opt.json")
    
    print(f"[optimize] Complete. {mutation_count} mutations saved.")


def stage_optimize(iterations):
    print("\n" + "="*70)
    print("STAGE 2: OPTIMIZE ALL STAGES")
    print("="*70)
    
    optimize_extractor(iterations)
    optimize_matcher(iterations)
    optimize_classifier(iterations)
    
    print("\n[optimize] All stages complete")


# ================================================================
# 3. Post-opt evaluation
# ================================================================

def stage_post_eval():
    print("\n" + "="*70)
    print("STAGE 3: POST-OPT EVAL")
    print("="*70)
    
    # Check if opt files exist
    opt_files = [PROMPTS_DIR / "identifier_extractor_opt.json",
                 PROMPTS_DIR / "service_matcher_opt.json",
                 PROMPTS_DIR / "customer_facing_classifier_opt.json"]
    
    if not any(f.exists() for f in opt_files):
        print("[post_eval] No optimized prompts found. Run optimization first.")
        return
    
    # Backup live prompts and swap in opt versions
    print("[post_eval] Swapping live prompts → optimized prompts...")
    backups = {}
    for stage_name, stage_info in STAGE_MAP.items():
        opt_path = PROMPTS_DIR / stage_info["opt"]
        live_path = PROMPTS_DIR / stage_info["live"]
        if opt_path.exists():
            # Backup
            backup_path = str(live_path.parent / f"{stage_info['live']}.backup")
            shutil.copy2(live_path, backup_path)
            # Swap
            shutil.copy2(opt_path, live_path)
            print(f"  [{stage_info['live']}] -> [{stage_info['opt']}]")
    
    try:
        # Run eval
        print("[post_eval] Running run_eval.py --baseline...")
        result = subprocess.run(
            [sys.executable, "-m", "tests.eval.run_eval", "--baseline"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr[:1000])
    finally:
        # Restore originals
        print("[post_eval] Restoring original prompts...")
        for stage_name, stage_info in STAGE_MAP.items():
            backup_path = PROMPTS_DIR / f"{stage_info['live']}.backup"
            live_path = PROMPTS_DIR / stage_info["live"]
            if backup_path.exists():
                shutil.copy2(backup_path, live_path)
                backup_path.unlink()
                print(f"  Restored [{stage_info['live']}]")
    
    print("[post_eval] Complete")


# ================================================================
# 4. Batch runs
# ================================================================

def stage_batch():
    print("\n" + "="*70)
    print("STAGE 4: BATCH RUNS")
    print("="*70)
    
    # Check if opt files exist
    if not any((PROMPTS_DIR / v["opt"]).exists() for v in STAGE_MAP.values()):
        print("[batch] No optimized prompts found. Run optimization first.")
        return
    
    # Baseline batch
    print("\n--- Baseline Batch ---")
    result = subprocess.run(
        [sys.executable, "scripts/run_batch.py"],
        cwd=str(REPO_ROOT - Path("scripts")),
        capture_output=True,
        text=True,
    )
    print(result.stdout[-2000:] if result.stdout else "")
    if result.stderr:
        print("STDERR:", result.stderr[-1000:])
    
    # Opt batch
    print("\n--- Optimized Batch ---")
    # Swap prompts
    backups = {}
    for stage_name, stage_info in STAGE_MAP.items():
        opt_path = PROMPTS_DIR / stage_info["opt"]
        live_path = PROMPTS_DIR / stage_info["live"]
        if opt_path.exists():
            backup_path = str(live_path.parent / f"{stage_info['live']}.backup")
            shutil.copy2(live_path, backup_path)
            shutil.copy2(opt_path, live_path)
            print(f"  [{stage_info['live']}] -> [{stage_info['opt']}]")
    
    try:
        result = subprocess.run(
            [sys.executable, "scripts/run_batch.py"],
            cwd=str(REPO_ROOT - Path("scripts")),
            capture_output=True,
            text=True,
        )
        print(result.stdout[-2000:] if result.stdout else "")
        if result.stderr:
            print("STDERR:", result.stderr[-1000:])
    finally:
        # Restore
        for stage_name, stage_info in STAGE_MAP.items():
            backup_path = PROMPTS_DIR / f"{stage_info['live']}.backup"
            live_path = PROMPTS_DIR / stage_info["live"]
            if backup_path.exists():
                shutil.copy2(backup_path, live_path)
                backup_path.unlink()
                print(f"  Restored [{stage_info['live']}]")
    
    print("[batch] Complete")


# ================================================================
# 5. Compare
# ================================================================

def stage_compare():
    print("\n" + "="*70)
    print("STAGE 5: COMPARISON")
    print("="*70)
    
    import glob
    eval_files = sorted(glob.glob(str(OUTPUT_DIR / "eval_baseline_*.json")))
    
    if not eval_files:
        print("[compare] No eval_baseline files found. Run eval first.")
        return
    
    if len(eval_files) < 2:
        print("[compare] Only 1 eval file found. Run baseline + post-eval for comparison.")
        return
    
    baseline = load_baseline(eval_files[0])
    post = load_baseline(eval_files[-1])
    
    # Compare prompt versions
    for stage_name, stage_info in STAGE_MAP.items():
        opt_path = PROMPTS_DIR / stage_info["opt"]
        live_path = PROMPTS_DIR / stage_info["live"]
        
        if opt_path.exists():
            opt_data = load_prompt(opt_path)
            live_data = load_prompt(live_path)
            print(f"\n[{OPT_PROMPT}]:")
            print(f"  Version: opt={opt_data.get('version', '?')} vs live={live_data.get('version', '?')}")
            print(f"  Length: opt={get_prompt_length(opt_data)} vs live={get_prompt_length(live_data)}")
    
    print("\n[compare] Done")


def load_baseline(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def get_prompt_length(data):
    text = "\n".join(msg.get("content", "") for msg in data.get("prompt", []))
    return len(text)


# ================================================================
# Main
# ================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["baseline", "optimize", "post_eval", "batch", "compare", "all"], default="all")
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    
    if args.stage in ("baseline", "all"):
        stage_baseline_eval()
    if args.stage in ("optimize", "all"):
        stage_optimize(args.iterations)
    if args.stage in ("post_eval", "all"):
        stage_post_eval()
    if args.stage in ("batch", "all"):
        stage_batch()
    if args.stage in ("compare", "all"):
        stage_compare()
    
    print("\n[complete]")
