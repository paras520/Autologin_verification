import requests
import json
import time

BASE = "https://il3e3qpwnnpinq-8000.proxy.runpod.net/v1"
AUTH = {"Authorization": "Bearer DIRO@123", "Content-Type": "application/json"}

def qwen(prompt, max_tokens=500, use_timeout=120):
    start = time.time()
    try:
        r = requests.post(f"{BASE}/chat/completions",
            json={"model": "Qwen/Qwen3.5-9B", "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": max_tokens, "temperature": 0.1},
            headers=AUTH, timeout=use_timeout)
        ms = (time.time() - start) * 1000
        data = r.json()
        c = data["choices"][0]["message"]
        content = c.get("content") or ""
        reasoning = c.get("reasoning") or ""
        finish = data["choices"][0]["finish_reason"]
        print(f"Elapsed: {ms:.0f}ms | finish: {finish}")
        print(f"Content: {repr(content)}")
        print(f"Reasoning: {repr(reasoning[:300] if reasoning else 'none')}")
        return content, reasoning, finish
    except Exception as e:
        print(f"Error after {time.time()-start:.1f}s: {e}")

# Test 1: Simple
print("=== Test 1: Simple 'Say hi' ===")
qwen("Say hi in 5 words", max_tokens=100)

# Test 2: Math (test reasoning)
print("\n=== Test 2: Math (2+2) ===")
qwen("What is 2+2? Answer with just the number.", max_tokens=50)

# Test 3: JSON output (bank match scenario)
print("\n=== Test 3: JSON bank match ===")
qwen('You are a bank verification system. Input: provider=HDFC, service=NetBanking. Return exactly this JSON: {"bank_matched": true, "confidence": 95}', max_tokens=200)

# Test 4: Classification task
print("\n=== Test 4: Classification ===")
qwen('Classify this as customer_login, hrms, or marketing_only: Title=Amazon India, Buttons=Sign In, Create Account, visible_text=shop electronics', max_tokens=200)

# Test 5: Judge-like evaluation
print("\n=== Test 5: Judge evaluation ===")
qwen('Judge: actual_output has bank_matched=false, expected_output has bank_matched=false. Match? Return agree or disagree.', max_tokens=200)
