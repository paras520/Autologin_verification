"""Quick test: does Qwen 3.5 give us JSON in content vs reasoning?"""
import asyncio, warnings
from openai import AsyncOpenAI

async def test():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        client = AsyncOpenAI(
            base_url='https://il3e3qpwnnpinq-8000.proxy.runpod.net/v1',
            api_key='DIRO@123',
        )
    resp = await client.chat.completions.create(
        model='Qwen/Qwen3.5-9B',
        messages=[
            {'role': 'system', 'content': 'Return ONLY valid JSON.'},
            {'role': 'user', 'content': 'Classify URL bankofamerica.com login page. Return JSON object with category and confidence fields.'},
        ],
        max_tokens=4000,
        timeout=120,
    )
    msg = resp.choices[0].message
    print('=== content ===')
    print(repr(msg.content))
    print()
    print('=== reasoning (first 1500 chars) ===')
    print((msg.reasoning or '')[:1500])
    print()
    print('=== finish_reason ===', resp.choices[0].finish_reason)
    print('=== usage ===', resp.usage)

asyncio.run(test())
