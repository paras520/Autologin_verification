"""
Test script to debug Langfuse prompt fetching (uses raw HTTP, no SDK)
"""
import os
import base64
import json
import requests
from dotenv import load_dotenv

load_dotenv()

host = os.getenv("LANGFUSE_HOST", "").rstrip("/")
pub  = os.getenv("LANGFUSE_PUBLIC_KEY", "")
sec  = os.getenv("LANGFUSE_SECRET_KEY", "")
token = base64.b64encode(f"{pub}:{sec}".encode()).decode()
headers = {"Authorization": f"Basic {token}"}

test_cases = [
    ("autologinQA/identifier_extractor/prompt", "stage2"),
    ("autologinQA/service_matcher/prompt",      "stage2"),
]

print("=" * 80)
print("TESTING LANGFUSE PROMPT FETCH (raw HTTP)")
print(f"Host: {host}")
print("=" * 80)

for prompt_path, label in test_cases:
    import urllib.parse
    encoded = urllib.parse.quote(prompt_path, safe="")
    url = f"{host}/api/public/v2/prompts/{encoded}?label={label}"
    print(f"\n--- {prompt_path}  label={label} ---")
    print(f"    GET {url}")
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.ok:
            data = resp.json()
            print(f"  status : {resp.status_code}")
            print(f"  version: {data.get('version')}")
            print(f"  config : {json.dumps(data.get('config'), indent=4)}")
            prompt = data.get("prompt", [])
            for msg in prompt:
                role = msg.get("role", "?")
                content = (msg.get("content") or "")[:120].replace("\n", " ")
                print(f"  [{role}]: {content}...")
        else:
            print(f"  FAILED  status={resp.status_code}  body={resp.text[:300]}")
    except Exception as e:
        print(f"  ERROR: {e}")

print("\n" + "=" * 80)
