This guide explains how to use the ultra-simple Langfuse + LiteLLM integration in your projects.

---

## 📁 Files to Copy

### 1. **Core Helper Module** (Required)
```
src/utils/langfuse_helper.py
```
**Purpose**: Contains all the functions for Langfuse + LiteLLM integration.

### 2. **Model Schemas** (If using function calling)
```
src/models/extractMetaDataAndSourceModel.py
```
**Purpose**: Contains Pydantic function/tool schemas. Copy only if you need function calling.

### 3. **Documentation** (Recommended)
```
FINAL_MIGRATION_COMPLETE.md
LANGFUSE_CONFIGS.md
CONFIG_QUICK_REFERENCE.json
TEAM_ONBOARDING.md (this file)
```

---

## 🎯 Quick Start (3 Steps)

### Step 1: Copy the Helper Module

Copy `src/utils/langfuse_helper.py` to your project's utils folder.

### Step 2: Install Dependencies

```bash
pip install litellm langfuse openai
```

### Step 3: Set Environment Variables

```bash
# Langfuse credentials
export LANGFUSE_PUBLIC_KEY="pk-lf-..."
export LANGFUSE_SECRET_KEY="sk-lf-..."
export LANGFUSE_HOST="https://cloud.langfuse.com"

# LiteLLM proxy (if using)
export LITELLM_PROXY_URL="http://your-litellm-proxy:4000"

# OpenAI API key
export OPENAI_API_KEY="sk-..."
```

---

## 💻 Usage Examples

### Example 1: Simple Text Completion

```python
from utils.langfuse_helper import get_prompts_from_langfuse, build_messages, call_litellm, parse_response

async def analyze_document(session_id: str, api_endpoint: str):
    # 1. Get prompts and config from Langfuse
    system_prompt, user_prompt, config, prompt_obj = get_prompts_from_langfuse(
        prompt_path="your-company/your-prompt-name",
        session_id=session_id
    )
    
    # 2. Build messages
    messages = build_messages(
        system_prompt=system_prompt,
        user_prompt=user_prompt
    )
    
    # 3. Call LiteLLM
    response = await call_litellm(
        config=config,
        messages=messages,
        session_id=session_id,
        api_endpoint=api_endpoint,
        prompt=prompt_obj
    )
    
    # 4. Parse response
    result = parse_response(response, has_functions=False)
    return result
```

### Example 2: With Images

```python
from utils.langfuse_helper import get_prompts_from_langfuse, build_messages, call_litellm, parse_response

async def analyze_image(image_bytes: bytes, session_id: str, api_endpoint: str):
    # Get prompts from Langfuse
    system_prompt, user_prompt, config, prompt_obj = get_prompts_from_langfuse(
        prompt_path="your-company/image-analysis",
        session_id=session_id
    )
    
    # Build messages with image
    messages = build_messages(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        images=[image_bytes]  # List of image bytes
    )
    
    # Call LiteLLM
    response = await call_litellm(
        config=config,
        messages=messages,
        session_id=session_id,
        api_endpoint=api_endpoint,
        prompt=prompt_obj
    )
    
    return parse_response(response, has_functions=False)
```

### Example 3: With PDF

```python
from utils.langfuse_helper import get_prompts_from_langfuse, build_messages, call_litellm, parse_response

async def analyze_pdf(pdf_bytes: bytes, session_id: str, api_endpoint: str):
    # Get prompts from Langfuse
    system_prompt, user_prompt, config, prompt_obj = get_prompts_from_langfuse(
        prompt_path="your-company/pdf-analysis",
        session_id=session_id
    )
    
    # Build messages with PDF
    messages = build_messages(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        pdf_file=pdf_bytes
    )
    
    # Call LiteLLM
    response = await call_litellm(
        config=config,
        messages=messages,
        session_id=session_id,
        api_endpoint=api_endpoint,
        prompt=prompt_obj
    )
    
    return parse_response(response, has_functions=False)
```

### Example 4: With Function Calling

```python
from utils.langfuse_helper import get_prompts_from_langfuse, build_messages, call_litellm, parse_response

async def extract_structured_data(text: str, session_id: str, api_endpoint: str):
    # Define function schema
    extraction_function = [{
        "name": "extract_data",
        "description": "Extract structured data from text",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Company name"},
                "email": {"type": "string", "description": "Email address"},
                "phone": {"type": "string", "description": "Phone number"}
            },
            "required": ["name", "email", "phone"]
        }
    }]
    
    # Get prompts from Langfuse
    system_prompt, user_prompt, config, prompt_obj = get_prompts_from_langfuse(
        prompt_path="your-company/data-extraction",
        session_id=session_id
    )
    
    # Build messages
    messages = build_messages(
        system_prompt=system_prompt,
        user_prompt=user_prompt
    )
    
    # Call LiteLLM with functions
    response = await call_litellm(
        config=config,
        messages=messages,
        session_id=session_id,
        api_endpoint=api_endpoint,
        functions=extraction_function,
        prompt=prompt_obj
    )
    
    # Parse response (has_functions=True)
    result = parse_response(response, has_functions=True)
    return result
```

### Example 5: With Langfuse Variables

```python
from utils.langfuse_helper import get_prompts_from_langfuse, build_messages, call_litellm, parse_response

async def personalized_analysis(user_name: str, data: dict, session_id: str, api_endpoint: str):
    # Prepare variables for Langfuse template
    variables = {
        "user_name": user_name,
        "data": json.dumps(data, ensure_ascii=False)
    }
    
    # Get prompts with variables
    system_prompt, user_prompt, config, prompt_obj = get_prompts_from_langfuse(
        prompt_path="your-company/personalized-prompt",
        session_id=session_id,
        variables=variables  # {{user_name}} and {{data}} will be replaced
    )
    
    # Build and call
    messages = build_messages(system_prompt=system_prompt, user_prompt=user_prompt)
    response = await call_litellm(config=config, messages=messages, session_id=session_id, api_endpoint=api_endpoint, prompt=prompt_obj)
    
    return parse_response(response, has_functions=False)
```

---

## 🔧 Langfuse Configuration Guide

### Where to Configure

1. Go to Langfuse Dashboard
2. Navigate to **Prompts**
3. Create a new prompt or edit existing one
4. Add **Config** section

### Configuration Format

The config section in Langfuse should be a JSON object with these fields:

```json
{
  "model": "model-name",
  "provider": "openai|gemini",
  "temperature": 0.2,
  "max_tokens": 2000,
  "seed": 42
}
```

---

## 📋 Model-Specific Configs

### OpenAI GPT-4o (Normal Models)

```json
{
  "model": "gpt-4o",
  "provider": "openai",
  "temperature": 0.2,
  "max_tokens": 4000,
  "seed": 42
}
```

### OpenAI O-Series (O3, O4-mini) - Reasoning Models

```json
{
  "model": "o3",
  "provider": "openai",
  "max_completion_tokens": 8000,
  "reasoning_effort": "medium",
  "seed": 42
}
```

**⚠️ Important**: O-series models:
- ❌ Do NOT support `temperature`
- ✅ Use `max_completion_tokens` (not `max_tokens`)
- ✅ Use `reasoning_effort` ("low", "medium", "high")

### OpenAI GPT-5 Series

```json
{
  "model": "gpt-5-nano",
  "provider": "openai",
  "max_completion_tokens": 4000,
  "seed": 42
}
```

**⚠️ Important**: GPT-5 models:
- ❌ Do NOT support `temperature`
- ✅ Use `max_completion_tokens`
- ❌ Do NOT support `reasoning_effort`

### Google Gemini (via LiteLLM Proxy)

```json
{
  "model": "litellm_proxy/gemini-2.5-flash",
  "provider": "gemini",
  "temperature": 0.2,
  "max_tokens": 2000
}
```

**⚠️ Important**: Always use `litellm_proxy/` prefix for Gemini models!

### Google Gemini (Direct Vertex AI)

```json
{
  "model": "gemini-2.5-flash",
  "provider": "gemini",
  "temperature": 0.2,
  "max_tokens": 2000
}
```

**⚠️ Requires**: Google Cloud credentials configured

---

## 🎯 Function Reference

### `get_prompts_from_langfuse()`

Fetches prompts and config from Langfuse.

```python
system_prompt, user_prompt, config, prompt_obj = get_prompts_from_langfuse(
    prompt_path="your-company/prompt-name",
    session_id="unique-session-id",
    variables={"key": "value"}  # Optional: for {{key}} replacement
)
```

**Returns**:
- `system_prompt` (str): System prompt text
- `user_prompt` (str): User prompt text
- `config` (dict): Model configuration
- `prompt_obj`: Langfuse prompt object (for tracing)

---

### `build_messages()`

Builds OpenAI-compatible message array.

```python
messages = build_messages(
    system_prompt="You are a helpful assistant",
    user_prompt="Analyze this document",
    images=[image_bytes1, image_bytes2],  # Optional: list of image bytes
    pdf_file=pdf_bytes  # Optional: PDF bytes
)
```

**Returns**: List of message dicts compatible with LiteLLM

---

### `call_litellm()`

Calls LiteLLM with automatic provider handling.

```python
response = await call_litellm(
    config=config,  # From get_prompts_from_langfuse()
    messages=messages,  # From build_messages()
    session_id="session-123",
    api_endpoint="/your-endpoint",
    functions=[function_schema],  # Optional: for function calling
    tools=[tool_schema],  # Optional: for tool calling
    prompt=prompt_obj  # Optional: for Langfuse tracing
)
```

**Returns**: LiteLLM response object

---

### `parse_response()`

Parses LiteLLM response to extract content or function calls.

```python
result = parse_response(
    response=litellm_response,
    has_functions=False  # Set to True if using function calling
)
```

**Returns**: 
- If `has_functions=False`: String content
- If `has_functions=True`: Dict with function call arguments

---

## 🚨 Common Pitfalls & Solutions

### 1. Temperature Error with O-Series/GPT-5

**Error**: `gpt-5 models don't support temperature=0.X`

**Solution**: Remove `temperature` from Langfuse config for O-series and GPT-5 models.

```json
❌ WRONG:
{
  "model": "o3",
  "temperature": 0.2
}

✅ CORRECT:
{
  "model": "o3",
  "max_completion_tokens": 8000,
  "reasoning_effort": "medium"
}
```

---

### 2. Gemini Credentials Error

**Error**: `Your default credentials were not found`

**Solution**: Use `litellm_proxy/` prefix in model name.

```json
❌ WRONG:
{
  "model": "gemini-2.5-flash"
}

✅ CORRECT:
{
  "model": "litellm_proxy/gemini-2.5-flash"
}
```

---

### 3. Function Schema Validation Error (Gemini)

**Error**: `functionDeclaration parameters.xxx schema didn't specify the schema type field`

**Solution**: Always include `"type"` field for nested objects/arrays in function schemas.

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

### 4. Function Call Parameter Error (Gemini)

**Error**: `vertex_ai does not support parameters: ['function_call']`

**Solution**: The helper already handles this! The `function_call` parameter is only added for OpenAI providers.

---

## 🎓 Best Practices

### 1. **Always Use Langfuse for Configuration**

❌ **Don't hardcode** model configs in code:
```python
# Bad
response = litellm.completion(
    model="gpt-4o",
    temperature=0.2,
    messages=messages
)
```

✅ **Do fetch** from Langfuse:
```python
# Good
system_prompt, user_prompt, config, prompt_obj = get_prompts_from_langfuse(...)
response = await call_litellm(config=config, messages=messages, ...)
```

**Why**: Change models without code changes!

---

### 2. **Use Session IDs for Tracing**

Always pass unique session IDs for proper Langfuse tracing:

```python
session_id = f"user-{user_id}-{timestamp}"
```

---

### 3. **Handle Errors Gracefully**

```python
try:
    system_prompt, user_prompt, config, prompt_obj = get_prompts_from_langfuse(...)
except Exception as e:
    # Fallback to default config
    config = {
        "model": "gpt-4o",
        "provider": "openai",
        "temperature": 0.2,
        "max_tokens": 2000
    }
    system_prompt = "You are a helpful assistant"
    user_prompt = "..."
    prompt_obj = None
```

---

### 4. **Optimize for Cost**

- Use `gpt-4o-mini` for simple tasks
- Use `o4-mini` instead of `o3` when possible
- Use `gemini-2.5-flash` for fast, cheap inference
- Set `reasoning_effort: "low"` for simple O-series tasks

---

### 5. **Test with Multiple Providers**

Your code should work with any provider! Test configs:

```json
// OpenAI
{"model": "gpt-4o", "provider": "openai", "temperature": 0.2, "max_tokens": 2000}

// Gemini
{"model": "litellm_proxy/gemini-2.5-flash", "provider": "gemini", "temperature": 0.2, "max_tokens": 2000}

// O-series
{"model": "o3", "provider": "openai", "max_completion_tokens": 8000, "reasoning_effort": "medium"}
```

---

## 📊 Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Your Application                      │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│              langfuse_helper.py Functions                │
│  • get_prompts_from_langfuse()                          │
│  • build_messages()                                      │
│  • call_litellm()                                        │
│  • parse_response()                                      │
└─────────────────────────────────────────────────────────┘
                           ↓
         ┌─────────────────┴─────────────────┐
         ↓                                    ↓
┌─────────────────┐                  ┌─────────────────┐
│    Langfuse     │                  │    LiteLLM      │
│  (Config + Log) │                  │  (Provider SDK) │
└─────────────────┘                  └─────────────────┘
                                              ↓
                        ┌─────────────────────┼─────────────────────┐
                        ↓                     ↓                     ↓
                ┌──────────────┐      ┌──────────────┐     ┌──────────────┐
                │    OpenAI    │      │    Gemini    │     │   Claude     │
                └──────────────┘      └──────────────┘     └──────────────┘
```

**Key Benefit**: Change providers by updating Langfuse config only!

---

## 🔐 Security Best Practices

### 1. **Never Commit API Keys**

Use environment variables:
```bash
export OPENAI_API_KEY="sk-..."
export LANGFUSE_SECRET_KEY="sk-lf-..."
```

### 2. **Use .env Files**

```bash
# .env (add to .gitignore!)
OPENAI_API_KEY=sk-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

Load with:
```python
from dotenv import load_dotenv
load_dotenv()
```

### 3. **Rotate Keys Regularly**

Change API keys every 90 days.

---

## 📞 Support & Questions

### Internal Resources
- **Langfuse Dashboard**: [Your Langfuse URL]
- **LiteLLM Proxy**: [Your LiteLLM Proxy URL]
- **Documentation**: See `FINAL_MIGRATION_COMPLETE.md`

### External Resources
- **LiteLLM Docs**: https://docs.litellm.ai/
- **Langfuse Docs**: https://langfuse.com/docs
- **OpenAI API Docs**: https://platform.openai.com/docs

---

## ✅ Checklist for Team Members

Before implementing:
- [ ] Copy `langfuse_helper.py` to your project
- [ ] Install dependencies (`litellm`, `langfuse`, `openai`)
- [ ] Set up environment variables
- [ ] Create Langfuse prompts with configs
- [ ] Test with a simple example
- [ ] Test with your specific use case
- [ ] Add error handling
- [ ] Deploy to staging
- [ ] Monitor Langfuse traces
- [ ] Deploy to production

---

AI team can now:
- ✅ Switch between OpenAI, Gemini, Claude without code changes
- ✅ Update prompts in Langfuse without deployments
- ✅ Track all LLM calls in Langfuse
- ✅ Use multimodal inputs (images, PDFs)
- ✅ Use function calling
- ✅ Maintain clean, simple code

