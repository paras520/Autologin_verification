"""
Ultra-Simplified Langfuse + LiteLLM Integration

No wrapper classes. No abstractions. Just direct LiteLLM calls.

This module provides simple helper functions to:
1. Fetch prompts and config from Langfuse
2. Build messages for LiteLLM
3. Call LiteLLM directly

That's it!
"""

import base64
import json
import os
import re
from urllib.parse import quote

import litellm
import requests

DEFAULT_LANGFUSE_LABEL = os.getenv("LANGFUSE_PROMPT_LABEL", "stage2")
LANGFUSE_TIMEOUT_SECONDS = 30
PLACEHOLDER_PATTERN = re.compile(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}")


def _replace_template_variables(template: str, variables: dict | None) -> str:
    if not variables:
        return template

    def replacer(match: re.Match[str]) -> str:
        variable_name = match.group(1)
        value = variables.get(variable_name, match.group(0))
        return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)

    return PLACEHOLDER_PATTERN.sub(replacer, template)


def _compile_prompt(raw_prompt, variables: dict | None):
    if isinstance(raw_prompt, str):
        return _replace_template_variables(raw_prompt, variables)

    if isinstance(raw_prompt, list):
        compiled_messages = []
        for message in raw_prompt:
            if not isinstance(message, dict):
                continue

            if message.get("type") == "placeholder":
                placeholder_name = message.get("name")
                placeholder_value = variables.get(placeholder_name, "") if variables else ""
                if isinstance(placeholder_value, list):
                    compiled_messages.extend(placeholder_value)
                elif isinstance(placeholder_value, dict):
                    compiled_messages.append(placeholder_value)
                elif placeholder_value:
                    compiled_messages.append(
                        {
                            "role": "user",
                            "content": str(placeholder_value),
                        }
                    )
                continue

            compiled_message = dict(message)
            if "content" in compiled_message and isinstance(compiled_message["content"], str):
                compiled_message["content"] = _replace_template_variables(
                    compiled_message["content"],
                    variables,
                )
            compiled_messages.append(compiled_message)
        return compiled_messages

    return raw_prompt


def _fetch_prompt_from_langfuse(prompt_path: str):
    host = os.getenv("LANGFUSE_HOST")
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")

    if not host or not public_key or not secret_key:
        raise ValueError(
            "LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY, and LANGFUSE_SECRET_KEY must be set."
        )

    encoded_prompt_name = quote(prompt_path, safe="")
    request_url = f"{host.rstrip('/')}/api/public/v2/prompts/{encoded_prompt_name}"
    response = requests.get(
        request_url,
        params={"label": DEFAULT_LANGFUSE_LABEL},
        auth=(public_key, secret_key),
        timeout=LANGFUSE_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def get_prompts_from_langfuse(
    prompt_path: str,
    session_id: str,
    variables: dict = None
):
    """
    Fetch prompts and config from Langfuse.
    
    Args:
        prompt_path: Path to Langfuse prompt (e.g., "source-extraction")
        session_id: Session ID for tracking
        variables: Optional variables for prompt compilation
        
    Returns:
        tuple: (system_prompt, user_prompt, config, prompt_obj)
        
    Example:
        system_prompt, user_prompt, config, _ = get_prompts_from_langfuse(
            prompt_path="source-extraction",
            session_id="session-123"
        )
    """
    prompt = _fetch_prompt_from_langfuse(prompt_path)
    compiled = _compile_prompt(prompt.get("prompt"), variables)
    
    # Extract prompts
    system_prompt = None
    user_prompt = None
    
    # Handle different prompt formats
    if isinstance(compiled, list):
        # Messages format
        for msg in compiled:
            if msg.get("role") == "system":
                system_prompt = msg.get("content")
            elif msg.get("role") == "user":
                user_prompt = msg.get("content")
    elif isinstance(compiled, str):
        # Simple string format
        user_prompt = compiled
    
    # Get config from prompt metadata
    config = prompt.get("config") or {}
    
    # Note: Model name should already include litellm_proxy/ prefix in Langfuse config
    # No auto-prefixing to avoid breaking openrouter/*, gemini-*, etc.
    
    return system_prompt, user_prompt, config, prompt


def build_messages(
    system_prompt: str = None,
    user_prompt: str = None,
    pdf_file: bytes = None,
    images: list = None,
    additional_text: str = None
):
    """
    Build messages array for LiteLLM in universal format.
    
    This format works for ALL providers (OpenAI, Gemini, Claude, etc.)
    LiteLLM handles the provider-specific translation.
    
    Args:
        system_prompt: System prompt text
        user_prompt: User prompt text
        pdf_file: PDF file as bytes
        images: List of image bytes
        additional_text: Additional text to append
        
    Returns:
        list: Messages array ready for LiteLLM
        
    Example:
        messages = build_messages(
            system_prompt="You are a helpful assistant",
            user_prompt="Extract metadata",
            pdf_file=pdf_bytes
        )
    """
    messages = []
    
    # Add system prompt
    if system_prompt:
        messages.append({
            "role": "system",
            "content": system_prompt
        })
    
    # Build user message content
    user_content = []
    
    # Add text
    text_parts = []
    if user_prompt:
        text_parts.append(user_prompt)
    if additional_text:
        text_parts.append(additional_text)
    
    if text_parts:
        user_content.append({
            "type": "text",
            "text": "\n".join(text_parts)
        })
    
    # Add PDF (LiteLLM universal format)
    if pdf_file:
        pdf_b64 = base64.b64encode(pdf_file).decode('utf-8')
        user_content.append({
            "type": "file",
            "file": {
                "file_data": f"data:application/pdf;base64,{pdf_b64}"
            }
        })
    
    # Add images (LiteLLM universal format)
    if images:
        for img in images:
            img_b64 = base64.b64encode(img).decode('utf-8')
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_b64}"
                }
            })
    
    # Add user message
    if user_content:
        if len(user_content) == 1 and user_content[0]["type"] == "text":
            # Simple text message
            messages.append({
                "role": "user",
                "content": user_content[0]["text"]
            })
        else:
            # Multimodal message
            messages.append({
                "role": "user",
                "content": user_content
            })
    
    return messages


async def call_litellm(
    config: dict,
    messages: list,
    session_id: str,
    api_endpoint: str = None,
    functions: list = None,
    tools: list = None,
    prompt: str = None
):
    """
    Call LiteLLM directly with config from Langfuse.
    
    This is the main function that replaces all the wrapper classes!
    
    Args:
        config: Config dict from Langfuse (model, temperature, etc.)
        messages: Messages array
        session_id: Session ID for tracking
        api_endpoint: API endpoint for metadata
        functions: Optional function schemas (OpenAI format)
        tools: Optional tool schemas (OpenAI format)
        prompt: Optional prompt for metadata
        
    Returns:
        response: LiteLLM response object
        
    Example:
        response = await call_litellm(
            config={"model": "litellm_proxy/gemini-2.5-flash", "temperature": 0.4},
            messages=messages,
            session_id="session-123",
            functions=extractMetaSourceFunction
        )
    """
    # Build parameters
    params = {
        "model": config["model"],
        "messages": messages,
        "seed": config.get("seed", 42)
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
    
    # Add temperature (if specified)
    if "temperature" in config:
        params["temperature"] = config["temperature"]
    
    # Add token limits
    if "max_tokens" in config:
        params["max_tokens"] = config["max_tokens"]
    elif "max_completion_tokens" in config:
        params["max_completion_tokens"] = config["max_completion_tokens"]
    
    # Add function calling
    if functions:
        params["functions"] = functions
        # Only add function_call for OpenAI (Gemini doesn't support it)
        if config.get("provider") == "openai":
            params["function_call"] = {"name": functions[0]["name"]}
    
    # Add tool calling
    if tools:
        params["tools"] = tools
        params["tool_choice"] = "auto"
    
    # Add API base if using proxy
    if os.getenv("LITELLM_PROXY_URL"):
        params["api_base"] = os.getenv("LITELLM_PROXY_URL")
    
    # Call LiteLLM!
    response = await litellm.acompletion(**params)
    
    return response


def parse_response(response, has_functions: bool = False, has_tools: bool = False):
    """
    Parse LiteLLM response.
    
    Args:
        response: LiteLLM response object
        has_functions: Whether function calling was used
        has_tools: Whether tool calling was used
        
    Returns:
        Parsed result (dict or str)
        
    Example:
        result = parse_response(response, has_functions=True)
    """
    message = response.choices[0].message
    
    # Function calling response
    if has_functions and hasattr(message, 'function_call') and message.function_call:
        return json.loads(message.function_call.arguments)
    
    # Tool calling response
    if has_tools and hasattr(message, 'tool_calls') and message.tool_calls:
        return json.loads(message.tool_calls[0].function.arguments)
    
    # Regular text response
    content = message.content
    
    # Try to parse as JSON
    if content:
        try:
            return json.loads(content)
        except (json.JSONDecodeError, TypeError):
            return content
    
    return content


# Convenience function that combines everything
async def get_and_call_litellm(
    prompt_path: str,
    session_id: str,
    api_endpoint: str = None,
    variables: dict = None,
    pdf_file: bytes = None,
    images: list = None,
    functions: list = None,
    tools: list = None,
    additional_text: str = None
):
    """
    All-in-one convenience function.
    
    Fetches prompt from Langfuse, builds messages, calls LiteLLM, and parses response.
    
    Args:
        prompt_path: Langfuse prompt path
        session_id: Session ID
        api_endpoint: API endpoint for tracking
        variables: Prompt variables
        pdf_file: Optional PDF bytes
        images: Optional image bytes list
        functions: Optional function schemas
        tools: Optional tool schemas
        additional_text: Optional additional text
        
    Returns:
        Parsed result
        
    Example:
        result = await get_and_call_litellm(
            prompt_path="source-extraction",
            session_id="session-123",
            pdf_file=pdf_bytes,
            functions=extractMetaSourceFunction
        )
    """
    # Get prompts and config from Langfuse
    system_prompt, user_prompt, config, _ = get_prompts_from_langfuse(
        prompt_path=prompt_path,
        session_id=session_id,
        variables=variables
    )
    
    # Build messages
    messages = build_messages(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        pdf_file=pdf_file,
        images=images,
        additional_text=additional_text
    )
    
    # Call LiteLLM
    response = await call_litellm(
        config=config,
        messages=messages,
        session_id=session_id,
        api_endpoint=api_endpoint,
        functions=functions,
        tools=tools,
        prompt=user_prompt
    )
    
    # Parse and return
    result = parse_response(
        response,
        has_functions=bool(functions),
        has_tools=bool(tools)
    )
    
    return result


# Legacy compatibility - for gradual migration
def get_prompt_and_client_from_langfuse(
    prompt_path: str,
    session_id: str,
    variables: dict = None,
    function=None,
    tool=None
):
    """
    DEPRECATED: Legacy compatibility function.
    
    This mimics the old interface but returns a simple dict instead of a client object.
    Use get_prompts_from_langfuse() + call_litellm() directly instead.
    
    Returns:
        tuple: (None, user_prompt, config, prompt_obj)
        
    Note: First element is None since we don't use client objects anymore!
    """
    system_prompt, user_prompt, config, prompt_obj = get_prompts_from_langfuse(
        prompt_path=prompt_path,
        session_id=session_id,
        variables=variables
    )
    
    # Store function/tool in config for later use
    if function:
        config["_function"] = function
    if tool:
        config["_tool"] = tool
    
    # Store prompts in config
    config["_system_prompt"] = system_prompt
    config["_user_prompt"] = user_prompt
    
    # Return None as first element (no client object!)
    return None, user_prompt, config, prompt_obj
