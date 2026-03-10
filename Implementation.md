# 🤖 Cursor Agent Instructions - LLM Integration Implementation

> **FOR AI AGENTS**: This file contains complete instructions to implement ultra-simple Langfuse + LiteLLM integration in any Python project. Follow these instructions carefully and systematically.

---

## 🎯 OBJECTIVE

Implement a production-ready, ultra-simple LLM integration system that:
- Works with OpenAI (GPT-4o, O3, GPT-5) and Gemini models
- Fetches prompts and configs dynamically from Langfuse
- Handles text, images, PDFs, and function calling
- Provides automatic tracing via Langfuse
- Requires zero code changes to switch providers or update prompts

---

## 📋 IMPLEMENTATION CHECKLIST

### Phase 1: Setup & Dependencies
- [ ] Install required packages
- [ ] Set up environment variables
- [ ] Create utils directory structure
- [ ] Copy core helper module

### Phase 2: Core Integration
- [ ] Implement `langfuse_helper.py`
- [ ] Test basic functionality
- [ ] Verify imports work

### Phase 3: Usage Implementation
- [ ] Update existing LLM calls to use new system
- [ ] Test with different providers
- [ ] Verify Langfuse tracing

### Phase 4: Production Readiness
- [ ] Add error handling
- [ ] Test all edge cases
- [ ] Document configuration
- [ ] Deploy to production

---

## 🔧 STEP 1: INSTALL DEPENDENCIES

### Required Packages

```bash
pip install litellm>=1.0.0
pip install langfuse>=2.0.0
pip install openai>=1.0.0
```

### Verify Installation

```python
import litellm
import langfuse
import openai
print("✅ All dependencies installed successfully")
```

---

## 🔐 STEP 2: ENVIRONMENT VARIABLES

### Required Environment Variables

Create or update `.env` file:

```bash
# Langfuse Configuration
LANGFUSE_PUBLIC_KEY=pk-lf-your-public-key
LANGFUSE_SECRET_KEY=sk-lf-your-secret-key
LANGFUSE_HOST=https://cloud.langfuse.com

# OpenAI Configuration
OPENAI_API_KEY=sk-your-openai-key

# LiteLLM Proxy (if using proxy)
LITELLM_PROXY_URL=http://your-litellm-proxy:4000

# Optional: Google Cloud (if using direct Vertex AI)
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
```

### Load Environment Variables

```python
import os
from dotenv import load_dotenv

load_dotenv()

# Verify critical variables
assert os.getenv('LANGFUSE_PUBLIC_KEY'), "LANGFUSE_PUBLIC_KEY not set"
assert os.getenv('LANGFUSE_SECRET_KEY'), "LANGFUSE_SECRET_KEY not set"
assert os.getenv('OPENAI_API_KEY'), "OPENAI_API_KEY not set"
```

---

## 📁 STEP 3: CREATE DIRECTORY STRUCTURE

```bash
your-project/
├── src/
│   ├── utils/
│   │   ├── __init__.py
│   │   └── langfuse_helper.py  # ← Create this file
│   └── ...
└── ...
```

---

## 🔨 STEP 4: IMPLEMENT CORE HELPER MODULE

### Create `src/utils/langfuse_helper.py`

**CRITICAL**: Copy this EXACT implementation. This is production-tested code.

```python
"""
Ultra-Simple Langfuse + LiteLLM Integration Helper

This module provides a simplified interface for:
1. Fetching prompts and configs from Langfuse
2. Building OpenAI-compatible messages (with images/PDFs)
3. Calling LiteLLM with automatic provider handling
4. Parsing responses (text or function calls)

Usage:
    from utils.langfuse_helper import get_prompts_from_langfuse, build_messages, call_litellm, parse_response
    
    # Get prompts and config from Langfuse
    system_prompt, user_prompt, config, prompt_obj = get_prompts_from_langfuse(
        prompt_path="your-company/your-prompt",
        session_id="unique-session-id"
    )
    
    # Build messages
    messages = build_messages(
        system_prompt=system_prompt,
        user_prompt=user_prompt
    )
    
    # Call LiteLLM
    response = await call_litellm(
        config=config,
        messages=messages,
        session_id=session_id,
        api_endpoint="/your-endpoint",
        prompt=prompt_obj
    )
    
    # Parse response
    result = parse_response(response, has_functions=False)
"""

import os
import json
import base64
import litellm
from langfuse import Langfuse
from typing import List, Dict, Any, Optional, Tuple
from utils.sendLogs import send_logs


def get_prompts_from_langfuse(
    prompt_path: str,
    session_id: str,
    variables: Optional[Dict[str, str]] = None
) -> Tuple[str, str, Dict[str, Any], Any]:
    """
    Fetch prompts and configuration from Langfuse.
    
    Args:
        prompt_path: Langfuse prompt path (e.g., "company/prompt-name")
        session_id: Unique session identifier for logging
        variables: Optional dict for template variable substitution (e.g., {"key": "value"} for {{key}})
    
    Returns:
        Tuple of (system_prompt, user_prompt, config, prompt_object)
        - system_prompt (str): System prompt text
        - user_prompt (str): User prompt text
        - config (dict): Model configuration from Langfuse
        - prompt_object: Langfuse prompt object for tracing
    
    Raises:
        Exception: If prompt cannot be loaded from Langfuse
    """
    try:
        # Initialize Langfuse client
        langfuse_client = Langfuse(
            public_key=os.getenv('LANGFUSE_PUBLIC_KEY'),
            secret_key=os.getenv('LANGFUSE_SECRET_KEY'),
            host=os.getenv('LANGFUSE_HOST', 'https://cloud.langfuse.com')
        )
        
        # Fetch prompt from Langfuse
        prompt = langfuse_client.get_prompt(
            name=prompt_path,
            label=os.getenv('LANGFUSE_PROMPT_LABEL', 'production')
        )
        
        # Compile prompt (applies variables if provided)
        if variables:
            compiled = prompt.compile(**variables)
        else:
            compiled = prompt.compile()
        
        # Extract system and user prompts
        system_prompt = ""
        user_prompt = ""
        
        if isinstance(compiled, list):
            # ChatPrompt format (list of messages)
            for msg in compiled:
                role = msg.get("role", "").lower()
                content = msg.get("content", "")
                if role == "system":
                    system_prompt = content
                elif role == "user":
                    user_prompt = content
        elif isinstance(compiled, str):
            # TextPrompt format (single string)
            system_prompt = compiled
            user_prompt = ""
        
        # Extract config from Langfuse
        config = prompt.config or {}
        
        # Log success
        send_logs(
            session_id,
            "LANGFUSE_PROMPT_LOADED",
            f"• Prompt: {prompt_path}\n• Label: {os.getenv('LANGFUSE_PROMPT_LABEL', 'production')}\n• Status: Success",
            facility="langfuse_helper"
        )
        
        # Extract and log config details
        model = config.get('model', 'not specified')
        provider = config.get('provider', 'not specified')
        send_logs(
            session_id,
            "PROMPT_CONFIG_EXTRACTED",
            f"• Prompt: {prompt_path}\n• Model: {model}\n• Config: {json.dumps(config)}\n• Status: Success",
            facility="langfuse_helper"
        )
        
        return system_prompt, user_prompt, config, prompt
        
    except Exception as e:
        send_logs(
            session_id,
            "LANGFUSE_PROMPT_ERROR",
            f"• Prompt: {prompt_path}\n• Error: {str(e)}\n• Status: Failed",
            facility="langfuse_helper"
        )
        raise Exception(f"Failed to load prompt '{prompt_path}' from Langfuse: {str(e)}") from e


def build_messages(
    system_prompt: Optional[str] = None,
    user_prompt: Optional[str] = None,
    images: Optional[List[bytes]] = None,
    pdf_file: Optional[bytes] = None
) -> List[Dict[str, Any]]:
    """
    Build OpenAI-compatible message array for LiteLLM.
    
    Args:
        system_prompt: System prompt text (optional)
        user_prompt: User prompt text (optional)
        images: List of image bytes (optional, for multimodal)
        pdf_file: PDF file bytes (optional, for PDF analysis)
    
    Returns:
        List of message dicts compatible with LiteLLM
    
    Example:
        # Text only
        messages = build_messages(
            system_prompt="You are helpful",
            user_prompt="Analyze this"
        )
        
        # With images
        messages = build_messages(
            system_prompt="You are helpful",
            user_prompt="Analyze these images",
            images=[image_bytes1, image_bytes2]
        )
        
        # With PDF
        messages = build_messages(
            system_prompt="You are helpful",
            user_prompt="Analyze this PDF",
            pdf_file=pdf_bytes
        )
    """
    messages = []
    
    # Add system message
    if system_prompt:
        messages.append({
            "role": "system",
            "content": system_prompt
        })
    
    # Build user message content
    if user_prompt or images or pdf_file:
        user_content = []
        
        # Add text
        if user_prompt:
            user_content.append({
                "type": "text",
                "text": user_prompt
            })
        
        # Add images
        if images:
            for img_bytes in images:
                # Convert to base64
                img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}"
                    }
                })
        
        # Add PDF (LiteLLM native support)
        if pdf_file:
            pdf_base64 = base64.b64encode(pdf_file).decode('utf-8')
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:application/pdf;base64,{pdf_base64}"
                }
            })
        
        # Add user message
        if len(user_content) == 1 and user_content[0]["type"] == "text":
            # Simple text message
            messages.append({
                "role": "user",
                "content": user_prompt
            })
        else:
            # Multimodal message
            messages.append({
                "role": "user",
                "content": user_content
            })
    
    return messages


async def call_litellm(
    config: Dict[str, Any],
    messages: List[Dict[str, Any]],
    session_id: str,
    api_endpoint: str,
    functions: Optional[List[Dict[str, Any]]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    prompt: Optional[Any] = None
) -> Any:
    """
    Call LiteLLM with automatic provider handling and Langfuse tracing.
    
    Args:
        config: Model configuration from Langfuse (must include 'model' and 'provider')
        messages: OpenAI-compatible message array
        session_id: Unique session identifier
        api_endpoint: API endpoint name (for tracing)
        functions: Optional function schemas for function calling
        tools: Optional tool schemas for tool calling
        prompt: Optional Langfuse prompt object (for tracing)
    
    Returns:
        LiteLLM response object
    
    Raises:
        Exception: If LiteLLM call fails
    
    Example:
        response = await call_litellm(
            config={"model": "gpt-4o", "temperature": 0.2, "max_tokens": 2000},
            messages=[{"role": "user", "content": "Hello"}],
            session_id="session-123",
            api_endpoint="/chat"
        )
    """
    try:
        # Extract model and provider
        model = config.get('model')
        provider = config.get('provider', 'openai')
        
        if not model:
            raise ValueError("Model not specified in config")
        
        # Build LiteLLM parameters
        params = {
            "model": model,
            "messages": messages
        }
        
        # Add metadata for Langfuse tracing (via extra_body)
        tags = ["m75_source_verify", "Smart_Upload"]
        if api_endpoint:
            tags.append(api_endpoint)
        
        # CRITICAL: Use extra_body to pass metadata to LiteLLM for Langfuse callback
        params["extra_body"] = {
            "metadata": {
                "session_id": session_id or "",
                "tags": tags,
                "prompt": prompt  # Pass the entire prompt object directly
            }
        }
        
        # Add optional parameters from config
        # Handle different model types (O-series, GPT-5, normal)
        if 'temperature' in config:
            params['temperature'] = config['temperature']
        
        if 'max_tokens' in config:
            params['max_tokens'] = config['max_tokens']
        
        if 'max_completion_tokens' in config:
            params['max_completion_tokens'] = config['max_completion_tokens']
        
        if 'seed' in config:
            params['seed'] = config['seed']
        
        if 'reasoning_effort' in config:
            params['reasoning_effort'] = config['reasoning_effort']
        
        if 'top_p' in config:
            params['top_p'] = config['top_p']
        
        # Add function calling
        if functions:
            params["functions"] = functions
            # CRITICAL: Only add function_call for OpenAI (Gemini doesn't support it)
            if provider == "openai":
                params["function_call"] = {"name": functions[0]["name"]}
        
        # Add tool calling
        if tools:
            params["tools"] = tools
        
        # Call LiteLLM
        response = await litellm.acompletion(**params)
        
        return response
        
    except Exception as e:
        send_logs(
            session_id,
            "LITELLM_CALL_ERROR",
            f"• Model: {config.get('model')}\n• Error: {str(e)}\n• Status: Failed",
            facility="langfuse_helper"
        )
        raise


def parse_response(
    response: Any,
    has_functions: bool = False
) -> Any:
    """
    Parse LiteLLM response to extract content or function calls.
    
    Args:
        response: LiteLLM response object
        has_functions: True if expecting function call response, False for text
    
    Returns:
        - If has_functions=False: String content
        - If has_functions=True: Dict with function call arguments
    
    Example:
        # Text response
        text = parse_response(response, has_functions=False)
        
        # Function call response
        args = parse_response(response, has_functions=True)
        # Returns: {"field1": "value1", "field2": "value2"}
    """
    try:
        message = response.choices[0].message
        
        if has_functions:
            # Extract function call arguments
            if hasattr(message, 'function_call') and message.function_call:
                args_str = message.function_call.arguments
                return json.loads(args_str)
            elif hasattr(message, 'tool_calls') and message.tool_calls:
                args_str = message.tool_calls[0].function.arguments
                return json.loads(args_str)
            else:
                # Fallback to content
                return message.content or ""
        else:
            # Extract text content
            return message.content or ""
            
    except Exception as e:
        raise Exception(f"Failed to parse response: {str(e)}") from e


# Convenience function: All-in-one
async def get_and_call_litellm(
    prompt_path: str,
    session_id: str,
    api_endpoint: str,
    variables: Optional[Dict[str, str]] = None,
    images: Optional[List[bytes]] = None,
    pdf_file: Optional[bytes] = None,
    functions: Optional[List[Dict[str, Any]]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    has_functions: bool = False
) -> Any:
    """
    Convenience function that combines all steps:
    1. Get prompts from Langfuse
    2. Build messages
    3. Call LiteLLM
    4. Parse response
    
    Args:
        prompt_path: Langfuse prompt path
        session_id: Unique session identifier
        api_endpoint: API endpoint name
        variables: Optional template variables
        images: Optional image bytes
        pdf_file: Optional PDF bytes
        functions: Optional function schemas
        tools: Optional tool schemas
        has_functions: True if expecting function response
    
    Returns:
        Parsed response (text or function arguments dict)
    
    Example:
        result = await get_and_call_litellm(
            prompt_path="company/prompt",
            session_id="session-123",
            api_endpoint="/analyze"
        )
    """
    # Get prompts and config
    system_prompt, user_prompt, config, prompt_obj = get_prompts_from_langfuse(
        prompt_path=prompt_path,
        session_id=session_id,
        variables=variables
    )
    
    # Build messages
    messages = build_messages(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        images=images,
        pdf_file=pdf_file
    )
    
    # Call LiteLLM
    response = await call_litellm(
        config=config,
        messages=messages,
        session_id=session_id,
        api_endpoint=api_endpoint,
        functions=functions,
        tools=tools,
        prompt=prompt_obj
    )
    
    # Parse and return
    return parse_response(response, has_functions=has_functions)
```

---

## ✅ STEP 5: VERIFY INSTALLATION

Create a test file `test_langfuse_helper.py`:

```python
import asyncio
from utils.langfuse_helper import get_prompts_from_langfuse, build_messages, call_litellm, parse_response

async def test_basic():
    """Test basic functionality"""
    try:
        # Test 1: Get prompts from Langfuse
        print("Test 1: Fetching prompts from Langfuse...")
        system_prompt, user_prompt, config, prompt_obj = get_prompts_from_langfuse(
            prompt_path="test/simple-prompt",  # Replace with your test prompt
            session_id="test-session-001"
        )
        print(f"✅ System prompt: {system_prompt[:50]}...")
        print(f"✅ Config: {config}")
        
        # Test 2: Build messages
        print("\nTest 2: Building messages...")
        messages = build_messages(
            system_prompt=system_prompt,
            user_prompt="Hello, world!"
        )
        print(f"✅ Messages built: {len(messages)} messages")
        
        # Test 3: Call LiteLLM
        print("\nTest 3: Calling LiteLLM...")
        response = await call_litellm(
            config=config,
            messages=messages,
            session_id="test-session-001",
            api_endpoint="/test",
            prompt=prompt_obj
        )
        print(f"✅ Response received")
        
        # Test 4: Parse response
        print("\nTest 4: Parsing response...")
        result = parse_response(response, has_functions=False)
        print(f"✅ Result: {result[:100]}...")
        
        print("\n🎉 All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(test_basic())
```

Run the test:
```bash
python test_langfuse_helper.py
```

---

## 🔄 STEP 6: MIGRATE EXISTING CODE

### Pattern 1: Replace Hardcoded LLM Calls

**BEFORE** (Hardcoded):
```python
import openai

async def analyze_text(text: str):
    response = await openai.ChatCompletion.acreate(
        model="gpt-4o",
        temperature=0.2,
        max_tokens=2000,
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": text}
        ]
    )
    return response.choices[0].message.content
```

**AFTER** (Dynamic with Langfuse):
```python
from utils.langfuse_helper import get_prompts_from_langfuse, build_messages, call_litellm, parse_response

async def analyze_text(text: str, session_id: str):
    # Get prompts and config from Langfuse
    system_prompt, user_prompt, config, prompt_obj = get_prompts_from_langfuse(
        prompt_path="company/text-analysis",
        session_id=session_id
    )
    
    # Build messages
    messages = build_messages(
        system_prompt=system_prompt,
        user_prompt=f"{user_prompt}\n\nText: {text}"
    )
    
    # Call LiteLLM
    response = await call_litellm(
        config=config,
        messages=messages,
        session_id=session_id,
        api_endpoint="/analyze-text",
        prompt=prompt_obj
    )
    
    # Parse and return
    return parse_response(response, has_functions=False)
```

### Pattern 2: Replace Image Analysis

**BEFORE**:
```python
import openai
import base64

async def analyze_image(image_bytes: bytes):
    img_base64 = base64.b64encode(image_bytes).decode('utf-8')
    response = await openai.ChatCompletion.acreate(
        model="gpt-4-vision-preview",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze this image"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
            ]
        }]
    )
    return response.choices[0].message.content
```

**AFTER**:
```python
from utils.langfuse_helper import get_prompts_from_langfuse, build_messages, call_litellm, parse_response

async def analyze_image(image_bytes: bytes, session_id: str):
    # Get prompts from Langfuse
    system_prompt, user_prompt, config, prompt_obj = get_prompts_from_langfuse(
        prompt_path="company/image-analysis",
        session_id=session_id
    )
    
    # Build messages with image
    messages = build_messages(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        images=[image_bytes]  # Automatic base64 encoding!
    )
    
    # Call LiteLLM
    response = await call_litellm(
        config=config,
        messages=messages,
        session_id=session_id,
        api_endpoint="/analyze-image",
        prompt=prompt_obj
    )
    
    return parse_response(response, has_functions=False)
```

### Pattern 3: Replace Function Calling

**BEFORE**:
```python
import openai

async def extract_data(text: str):
    function_schema = [{
        "name": "extract_data",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string"}
            },
            "required": ["name", "email"]
        }
    }]
    
    response = await openai.ChatCompletion.acreate(
        model="gpt-4o",
        messages=[{"role": "user", "content": text}],
        functions=function_schema,
        function_call={"name": "extract_data"}
    )
    
    return json.loads(response.choices[0].message.function_call.arguments)
```

**AFTER**:
```python
from utils.langfuse_helper import get_prompts_from_langfuse, build_messages, call_litellm, parse_response

async def extract_data(text: str, session_id: str):
    # Define function schema
    function_schema = [{
        "name": "extract_data",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string"}
            },
            "required": ["name", "email"]
        }
    }]
    
    # Get prompts from Langfuse
    system_prompt, user_prompt, config, prompt_obj = get_prompts_from_langfuse(
        prompt_path="company/data-extraction",
        session_id=session_id
    )
    
    # Build messages
    messages = build_messages(
        system_prompt=system_prompt,
        user_prompt=f"{user_prompt}\n\n{text}"
    )
    
    # Call LiteLLM with functions
    response = await call_litellm(
        config=config,
        messages=messages,
        session_id=session_id,
        api_endpoint="/extract-data",
        functions=function_schema,
        prompt=prompt_obj
    )
    
    # Parse function response
    return parse_response(response, has_functions=True)
```

---

## 📊 STEP 7: CREATE LANGFUSE CONFIGURATIONS

### Where to Configure

1. Go to **Langfuse Dashboard** → **Prompts**
2. Create a new prompt or edit existing
3. Add **Config** section (JSON format)

### Configuration Examples

#### For OpenAI GPT-4o (Normal Models)

```json
{
  "model": "gpt-4o",
  "provider": "openai",
  "temperature": 0.2,
  "max_tokens": 4000,
  "seed": 42
}
```

**System Prompt Example**:
```
You are a helpful AI assistant that analyzes documents accurately.
```

**User Prompt Example**:
```
Analyze the following document and extract key information.
```

---

#### For OpenAI O3 (Reasoning Models)

```json
{
  "model": "o3",
  "provider": "openai",
  "max_completion_tokens": 8000,
  "reasoning_effort": "medium",
  "seed": 42
}
```

**⚠️ CRITICAL**: 
- ❌ Do NOT include `temperature` (not supported)
- ✅ Use `max_completion_tokens` (not `max_tokens`)
- ✅ Use `reasoning_effort` ("low", "medium", "high")

---

#### For OpenAI GPT-5 Series

```json
{
  "model": "gpt-5-nano",
  "provider": "openai",
  "max_completion_tokens": 4000,
  "seed": 42
}
```

**⚠️ CRITICAL**:
- ❌ Do NOT include `temperature` (not supported)
- ✅ Use `max_completion_tokens`
- ❌ Do NOT include `reasoning_effort`

---

#### For Google Gemini (via LiteLLM Proxy)

```json
{
  "model": "litellm_proxy/gemini-2.5-flash",
  "provider": "gemini",
  "temperature": 0.2,
  "max_tokens": 2000
}
```

**⚠️ CRITICAL**:
- ✅ MUST use `litellm_proxy/` prefix
- ✅ Use `max_tokens` (not `max_completion_tokens`)

---

#### For Google Gemini (Direct Vertex AI)

```json
{
  "model": "gemini-2.5-flash",
  "provider": "gemini",
  "temperature": 0.2,
  "max_tokens": 2000
}
```

**⚠️ CRITICAL**:
- Requires Google Cloud credentials configured
- Better to use proxy version above

---

#### For OpenRouter Models

```json
{
  "model": "litellm_proxy/openrouter/anthropic/claude-3.5-sonnet",
  "provider": "openrouter",
  "temperature": 0.7,
  "max_tokens": 4000
}
```

**Popular OpenRouter Models**:
- `litellm_proxy/openrouter/anthropic/claude-3.5-sonnet` - Best quality
- `litellm_proxy/openrouter/openai/gpt-4-turbo` - OpenAI via OpenRouter
- `litellm_proxy/openrouter/meta-llama/llama-3.1-70b-instruct` - Cost-effective
- `litellm_proxy/openrouter/google/gemini-pro` - Google via OpenRouter
- `litellm_proxy/openrouter/z-ai/glm-4.7` - GLM-4.7 model

**⚠️ CRITICAL**:
- ✅ MUST use `litellm_proxy/openrouter/` prefix
- ✅ Requires `OPENROUTER_API_KEY` environment variable
- ✅ Requires `LITELLM_PROXY_API_KEY` environment variable

---

## 🚨 STEP 8: CRITICAL ERROR PREVENTION

### Error 1: Temperature with O-Series/GPT-5

**ERROR**: `gpt-5 models don't support temperature=0.X`

**CAUSE**: Including `temperature` in config for O-series or GPT-5 models

**FIX**: Remove `temperature` from Langfuse config

```json
❌ WRONG:
{
  "model": "o3",
  "temperature": 0.2,
  "max_completion_tokens": 8000
}

✅ CORRECT:
{
  "model": "o3",
  "max_completion_tokens": 8000,
  "reasoning_effort": "medium",
  "seed": 42
}
```

---

### Error 2: Gemini Credentials Error

**ERROR**: `Your default credentials were not found`

**CAUSE**: Using direct Gemini model name without proxy prefix

**FIX**: Add `litellm_proxy/` prefix

```json
❌ WRONG:
{
  "model": "gemini-2.5-flash",
  "provider": "gemini"
}

✅ CORRECT:
{
  "model": "litellm_proxy/gemini-2.5-flash",
  "provider": "gemini"
}
```

---

### Error 3: Function Schema Validation (Gemini)

**ERROR**: `functionDeclaration parameters.xxx schema didn't specify the schema type field`

**CAUSE**: Missing `"type"` field in nested objects/arrays

**FIX**: Always include `"type"` field

```python
❌ WRONG:
{
    "parameters": {
        "type": "object",
        "properties": {
            "items": {  # Missing "type": "array"
                "items": {"type": "string"}
            }
        }
    }
}

✅ CORRECT:
{
    "parameters": {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",  # ← Added this!
                "items": {"type": "string"}
            }
        }
    }
}
```

---

### Error 4: Function Call Parameter (Gemini)

**ERROR**: `vertex_ai does not support parameters: ['function_call']`

**SOLUTION**: Already handled in `langfuse_helper.py`! The code only adds `function_call` for OpenAI providers.

---

## ✅ STEP 9: TESTING CHECKLIST

### Test 1: Basic Text Completion

```python
async def test_text():
    result = await get_and_call_litellm(
        prompt_path="test/simple-text",
        session_id="test-text-001",
        api_endpoint="/test-text"
    )
    assert result, "Result should not be empty"
    print(f"✅ Text test passed: {result[:50]}...")
```

### Test 2: Image Analysis

```python
async def test_image():
    # Load test image
    with open("test_image.png", "rb") as f:
        image_bytes = f.read()
    
    result = await get_and_call_litellm(
        prompt_path="test/image-analysis",
        session_id="test-image-001",
        api_endpoint="/test-image",
        images=[image_bytes]
    )
    assert result, "Result should not be empty"
    print(f"✅ Image test passed: {result[:50]}...")
```

### Test 3: PDF Analysis

```python
async def test_pdf():
    # Load test PDF
    with open("test_document.pdf", "rb") as f:
        pdf_bytes = f.read()
    
    result = await get_and_call_litellm(
        prompt_path="test/pdf-analysis",
        session_id="test-pdf-001",
        api_endpoint="/test-pdf",
        pdf_file=pdf_bytes
    )
    assert result, "Result should not be empty"
    print(f"✅ PDF test passed: {result[:50]}...")
```

### Test 4: Function Calling

```python
async def test_function():
    function_schema = [{
        "name": "extract_test",
        "parameters": {
            "type": "object",
            "properties": {
                "field1": {"type": "string"},
                "field2": {"type": "number"}
            },
            "required": ["field1", "field2"]
        }
    }]
    
    result = await get_and_call_litellm(
        prompt_path="test/function-calling",
        session_id="test-function-001",
        api_endpoint="/test-function",
        functions=function_schema,
        has_functions=True
    )
    assert isinstance(result, dict), "Result should be a dict"
    assert "field1" in result, "field1 should be in result"
    print(f"✅ Function test passed: {result}")
```

### Test 5: Provider Switching

```python
async def test_provider_switch():
    # Test with OpenAI
    result_openai = await get_and_call_litellm(
        prompt_path="test/openai-prompt",  # Config: {"model": "gpt-4o"}
        session_id="test-openai-001",
        api_endpoint="/test-openai"
    )
    print(f"✅ OpenAI test passed")
    
    # Test with Gemini (just change Langfuse config!)
    result_gemini = await get_and_call_litellm(
        prompt_path="test/gemini-prompt",  # Config: {"model": "litellm_proxy/gemini-2.5-flash"}
        session_id="test-gemini-001",
        api_endpoint="/test-gemini"
    )
    print(f"✅ Gemini test passed")
```

---

## 🎯 STEP 10: PRODUCTION DEPLOYMENT

### Pre-Deployment Checklist

- [ ] All tests passing
- [ ] Environment variables configured
- [ ] Langfuse prompts created with correct configs
- [ ] Error handling added
- [ ] Logging configured
- [ ] Langfuse tracing verified
- [ ] Load testing completed
- [ ] Documentation updated

### Deployment Steps

1. **Staging Deployment**
   ```bash
   # Deploy to staging
   git checkout staging
   git merge feature/llm-integration
   git push origin staging
   
   # Run smoke tests
   pytest tests/test_llm_integration.py
   ```

2. **Monitor Langfuse**
   - Check Langfuse dashboard for traces
   - Verify all calls are logged
   - Check for errors

3. **Production Deployment**
   ```bash
   # Deploy to production
   git checkout main
   git merge staging
   git push origin main
   
   # Monitor closely
   ```

4. **Post-Deployment Monitoring**
   - Monitor error rates
   - Check Langfuse traces
   - Verify response times
   - Check cost metrics

---

## 📚 STEP 11: DOCUMENTATION

### Update Project README

Add this section to your project's README:

```markdown
## LLM Integration

This project uses an ultra-simple Langfuse + LiteLLM integration for all AI/LLM calls.

### Quick Usage

```python
from utils.langfuse_helper import get_prompts_from_langfuse, build_messages, call_litellm, parse_response

# Get prompts and config from Langfuse
system_prompt, user_prompt, config, prompt_obj = get_prompts_from_langfuse(
    prompt_path="company/your-prompt",
    session_id="unique-session-id"
)

# Build messages
messages = build_messages(
    system_prompt=system_prompt,
    user_prompt=user_prompt
)

# Call LiteLLM
response = await call_litellm(
    config=config,
    messages=messages,
    session_id="unique-session-id",
    api_endpoint="/your-endpoint",
    prompt=prompt_obj
)

# Parse response
result = parse_response(response, has_functions=False)
```

### Switching Providers

To switch from OpenAI to Gemini (or vice versa):
1. Go to Langfuse Dashboard
2. Update the prompt config
3. Change `model` field
4. No code changes needed!

### Supported Models

- OpenAI: GPT-4o, O3, O4-mini, GPT-5-nano
- Google: Gemini 2.5-flash, Gemini 2.5-pro
- Anthropic: Claude (via LiteLLM)
```

---

## 🎉 COMPLETION VERIFICATION

### Final Checklist

- [ ] `langfuse_helper.py` created and working
- [ ] All dependencies installed
- [ ] Environment variables configured
- [ ] At least one test prompt created in Langfuse
- [ ] Basic test passing
- [ ] Existing code migrated (if applicable)
- [ ] Error handling implemented
- [ ] Langfuse tracing verified
- [ ] Documentation updated
- [ ] Team notified

### Success Criteria

✅ **You have successfully implemented the integration if:**

1. You can call `get_prompts_from_langfuse()` without errors
2. You can switch between OpenAI and Gemini by only changing Langfuse config
3. All LLM calls appear in Langfuse dashboard with traces
4. You can update prompts in Langfuse without code changes
5. Images, PDFs, and function calling work correctly

---

## 📞 TROUBLESHOOTING

### Issue: Import Error

**Error**: `ModuleNotFoundError: No module named 'utils.langfuse_helper'`

**Fix**:
1. Verify file exists at `src/utils/langfuse_helper.py`
2. Verify `src/utils/__init__.py` exists
3. Check Python path: `import sys; print(sys.path)`

---

### Issue: Langfuse Connection Error

**Error**: `Failed to load prompt from Langfuse`

**Fix**:
1. Verify environment variables:
   ```python
   import os
   print(os.getenv('LANGFUSE_PUBLIC_KEY'))
   print(os.getenv('LANGFUSE_SECRET_KEY'))
   print(os.getenv('LANGFUSE_HOST'))
   ```
2. Check Langfuse dashboard is accessible
3. Verify prompt exists in Langfuse

---

### Issue: LiteLLM API Error

**Error**: `litellm.APIConnectionError`

**Fix**:
1. Verify API keys:
   ```python
   print(os.getenv('OPENAI_API_KEY'))
   ```
2. Check network connectivity
3. Verify model name is correct in Langfuse config

---

### Issue: Function Schema Error

**Error**: `schema didn't specify the schema type field`

**Fix**: Add `"type"` field to all nested objects/arrays in function schemas

---

## 🎓 BEST PRACTICES

1. **Always use session IDs** for proper tracing
2. **Never hardcode model configs** - always use Langfuse
3. **Test with multiple providers** before production
4. **Monitor Langfuse traces** regularly
5. **Use fallback configs** for critical paths
6. **Document all Langfuse prompts** with clear names
7. **Version your prompts** in Langfuse
8. **Add error handling** around all LLM calls
9. **Log errors** to Sentry or similar
10. **Monitor costs** via Langfuse dashboard

---

## ✅ IMPLEMENTATION COMPLETE

**Congratulations!** You have successfully implemented the ultra-simple Langfuse + LiteLLM integration.

Your codebase now supports:
- ✅ Dynamic provider switching (OpenAI ↔ Gemini)
- ✅ Zero-code prompt updates
- ✅ Automatic Langfuse tracing
- ✅ Multimodal inputs (images, PDFs)
- ✅ Function calling
- ✅ Clean, maintainable code

**Next Steps**:
1. Share this implementation with your team
2. Create more Langfuse prompts for different use cases
3. Monitor usage and costs in Langfuse
4. Iterate and improve based on traces

---

**Questions or Issues?**
- Check Langfuse documentation: https://langfuse.com/docs
- Check LiteLLM documentation: https://docs.litellm.ai/
- Review this guide again

---

**Last Updated**: 2026-02-09  
**Version**: 1.0.0  
**Status**: Production Ready ✅
