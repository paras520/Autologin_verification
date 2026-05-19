import requests
import json

base = "https://il3e3qpwnnpinq-8000.proxy.runpod.net/v1/chat/completions"
auth = {"Authorization": "Bearer DIRO@123"}

def qwen(prompt, max_tokens=500):
    data = {
        "model": "Qwen/Qwen3.5-9B",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }
    r = requests.post(base, json=data, headers=auth)
    resp = r.json()
    choice = resp["choices"][0]
    content = choice["message"].get("content", "") or ""
    reasoning = choice["message"].get("reasoning", "") or ""
    return content.strip(), reasoning.strip(), choice["finish_reason"]

# Test 1: Simple
content, reason, finish = qwen("What is 2+2?", max_tokens=300)
print(f"Test 1 - 'Hello World': {repr(content)} | finish: {finish}")

# Test 2: JSON output
prompt2 = '''You are a bank page verification system.
Input: provider=HDFC Bank, service=NetBanking
Output only valid JSON with no markdown:
{"bank_matched": true, "conf": 95}'''
content, reason, finish = qwen(prompt2, max_tokens=300)
print(f"Test 2 - JSON: {repr(content[:100])} | finish: {finish}")

# Test 3: Reasoning quality (judge-style task)
prompt3 = '''You are evaluating whether this extraction output matches the expected output.

Actual output: {"bank_matched": false, "confidence_score": 5}
Expected output: {"bank_matched": false, "confidence_score": 3}

Does the bank_matched match? Return a verdict: agree or disagree.'''
content, reason, finish = qwen(prompt3, max_tokens=500)
print(f"Test 3 - Judge: {repr(content[:150])} | finish: {finish}")

# Test 4: Classification task
prompt4 = '''Classify this page:
Title: Amazon India - Online Shopping
Headings: Electronics, Fashion, Home
Buttons: Sign In, Create Account
Is this a bank login page? Answer true/false only.'''
content, reason, finish = qwen(prompt4, max_tokens=200)
print(f"Test 4 - Classification: {repr(content[:150])} | finish: {finish}")

print(f"\nReasoning sample (Test 1):\n{reason[:200]}")
