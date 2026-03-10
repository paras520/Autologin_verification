from langfuse import get_client                
import os
from utils.sendLogs import send_logs
# Legacy imports - no longer needed with ultra-simple approach
# from utils.model_factory import create_client_from_langfuse
# from utils.model_config_validator import ConfigValidationError
# from utils.unified_client import UnifiedModelClient

langfuse = get_client()

def get_prompt_langfuse(prompt_path, session_id, variables=None):
    """
    Retrieve and process a prompt from Langfuse with comprehensive error handling and logging.
    
    This function fetches a prompt from Langfuse using the specified path and label,
    then processes it to extract system/user prompts, model configuration, and metadata.
    It handles both TextPrompt (string) and ChatPrompt (list) formats from Langfuse.
    
    Args:
        prompt_path (str): The path/identifier of the prompt in Langfuse
        session_id (str): Session identifier for logging and tracking purposes
        variables (dict, optional): Variables for Langfuse template substitution (e.g., {{first_pass}}, {{search_result}})
        
    Returns:
        tuple: A 4-element tuple containing:
            - system_prompt (str): The system prompt text
            - user_prompt (str or None): The user prompt text (if available)
            - model (str): The model name from prompt config or default
            - prompt_dict (dict): The prompt object's metadata dictionary
            
    Raises:
        Exception: If prompt cannot be loaded from Langfuse or compilation fails
        
    Example:
        >>> sys_prompt, user_prompt, model, metadata = get_prompt_langfuse(
        ...     "m75/sourceCheck/extraction", "session_123", 
        ...     variables={"first_pass": json.dumps(data)}
        ... )
        >>> print(f"Using model: {model}")
        >>> print(f"System prompt: {sys_prompt[:50]}...")
    """
    prompt = None
    
    # Step 1: Fetch prompt from Langfuse
    try:
        prompt = langfuse.get_prompt(prompt_path, label=os.getenv("LANGFUSE_PROMPT_LABEL"))
        send_logs(session_id, "LANGFUSE_PROMPT_LOADED",
                 f"• Prompt: {prompt_path}\n• Label: {os.getenv('LANGFUSE_PROMPT_LABEL')}\n• Status: Success",
                 facility="getprompts")
    except Exception as e:
        send_logs(session_id, "LANGFUSE_PROMPT_LOAD_FAILED",
                 f"• Prompt: {prompt_path}\n• Label: {os.getenv('LANGFUSE_PROMPT_LABEL')}\n• Error: {str(e)}\n• Status: Failed",
                 facility="getprompts")
        raise Exception(f"Could not load prompt {prompt_path}: {e}")
    
    # Step 2: Extract model configuration from prompt
    model = None
    try:
        if hasattr(prompt, 'config') and prompt.config:
            model = prompt.config.get("model")
            send_logs(session_id, "PROMPT_CONFIG_EXTRACTED",
                     f"• Prompt: {prompt_path}\n• Model: {model}\n• Config: {prompt.config}\n• Status: Success",
                     facility="getprompts")
        else:
            send_logs(session_id, "PROMPT_CONFIG_MISSING",
                     f"• Prompt: {prompt_path}\n• Config: None\n• Will use default model",
                     facility="getprompts")
    except Exception as e:
        send_logs(session_id, "PROMPT_CONFIG_ERROR",
                 f"• Prompt: {prompt_path}\n• Error: {str(e)}",
                 facility="getprompts")
    
    # Step 3: Use default model if none found in config
    if model is None:
        model = "gpt-5-nano"
        send_logs(session_id, "USING_DEFAULT_MODEL",
                 f"• Prompt: {prompt_path}\n• Default model: {model}\n• Reason: No model in config",
                 facility="getprompts")
    
    # Step 4: Compile and process prompt content with variable substitution
    try:
        # Pass variables to compile() for Langfuse template substitution
        if variables:
            compiled_prompt = prompt.compile(**variables)
            send_logs(session_id, "PROMPT_VARIABLES_SUBSTITUTED",
                     f"• Prompt: {prompt_path}\n• Variables: {list(variables.keys())}\n• Status: Success",
                     facility="getprompts")
        else:
            compiled_prompt = prompt.compile()
            
        system_prompt = None
        user_prompt = None
        
        # Handle TextPromptClient (returns string)
        if isinstance(compiled_prompt, str):
            system_prompt = compiled_prompt
            send_logs(session_id, "PROMPT_COMPILED",
                     f"• Prompt: {prompt_path}\n• Type: TextPrompt (string)\n• Model: {model}\n• Status: Success",
                     facility="getprompts")
            
        # Handle ChatPromptClient (returns list of messages)
        elif isinstance(compiled_prompt, list) and len(compiled_prompt) > 0:
            # Extract system and user prompts from message list
            for message in compiled_prompt:
                if message.get('role') == 'system':
                    system_prompt = message.get('content', '')
                elif message.get('role') == 'user':
                    user_prompt = message.get('content', '')
            
            send_logs(session_id, "PROMPT_COMPILED",
                     f"• Prompt: {prompt_path}\n• Type: ChatPrompt (list)\n• Model: {model}\n• Messages: {len(compiled_prompt)}\n• Status: Success",
                     facility="getprompts")
            
            # Validate that we found a system prompt
            if system_prompt is None:
                raise Exception("No system prompt found in Langfuse response")
        else:
            send_logs(session_id, "PROMPT_COMPILE_ERROR",
                     f"• Prompt: {prompt_path}\n• Unexpected format: {type(compiled_prompt)}\n• Status: Failed",
                     facility="getprompts")
            raise Exception(f"Unexpected compiled prompt format: {type(compiled_prompt)}")
            
    except Exception as e:
        send_logs(session_id, "PROMPT_COMPILE_FAILED",
                 f"• Prompt: {prompt_path}\n• Error: {str(e)}\n• Status: Failed",
                 facility="getprompts")
        raise Exception(f"Failed to compile prompt: {e}")
    
    # Step 5: Safely extract prompt metadata
    try:
        prompt_dict = prompt.__dict__
    except:
        prompt_dict = {"error": "Could not access prompt.__dict__"}
        send_logs(session_id, "get_prompt", f"Could not access prompt.__dict__")
    
    return system_prompt, user_prompt, model, prompt_dict

def build_combined_prompt_object(system_tuple, user_tuple, system_path: str | None = None, user_path: str | None = None):
    """
    Create a combined prompt object suitable for passing to get_gpt_response as metadata.

    Parameters:
    - system_tuple: Tuple returned from get_prompt for the system prompt: (text, model, prompt_dict)
    - user_tuple:   Tuple returned from get_prompt for the user prompt:   (text, model, prompt_dict)
    - system_path:  Optional path/id you used for the system prompt in Langfuse (for tagging)
    - user_path:    Optional path/id you used for the user prompt in Langfuse (for tagging)

    Returns:
    - combined_prompt_dict: A dict with both prompts and a messages array [{role, content}, ...].

    Notes:
    - This does NOT alter model behavior by itself; it is intended for logging/trace metadata.
    - You must still add the system and user prompts to GPTConfig via add_system_prompt/add_user_prompt_text.
    """
    try:
        sys_text, sys_model, sys_meta = system_tuple
    except Exception:
        sys_text, sys_model, sys_meta = "", None, {}
    try:
        usr_text, usr_model, usr_meta = user_tuple
    except Exception:
        usr_text, usr_model, usr_meta = "", None, {}

    combined = {
        "type": "combined",
        "messages": [
            {"role": "system", "content": sys_text},
            {"role": "user", "content": usr_text},
        ],
        "components": [
            {
                "label": "system",
                "path": system_path,
                "model": sys_model,
                "prompt": sys_meta,
            },
            {
                "label": "user",
                "path": user_path,
                "model": usr_model,
                "prompt": usr_meta,
            },
        ],
    }

    # Convenience duplicates for easy filtering in dashboards
    combined["system"] = {"path": system_path, "model": sys_model}
    combined["user"] = {"path": user_path, "model": usr_model}

    return combined


def get_prompt_and_client_from_langfuse(
    prompt_path: str,
    session_id: str,
    variables: dict = None,
    function=None,
    tool=None
):
    """
    NEW RECOMMENDED WAY: Get prompt from Langfuse AND automatically create the appropriate client.
    
    This function:
    1. Fetches the prompt from Langfuse
    2. Extracts system/user prompts and config
    3. Validates the config
    4. Automatically creates GPTConfig or LiteLLMConfig based on provider
    5. Pre-initializes the client with system prompt
    
    Args:
        prompt_path: The path/identifier of the prompt in Langfuse
        session_id: Session identifier for logging and tracking
        variables: Optional variables for Langfuse template substitution
        function: Optional function calling schema (for OpenAI, kept in code)
        tool: Optional tool schema (for OpenAI, kept in code)
        
    Returns:
        tuple: (client, user_prompt, config_dict, prompt_metadata)
            - client: Ready-to-use GPTConfig or LiteLLMConfig instance (with system prompt already added)
            - user_prompt: User prompt text (if available, None otherwise)
            - config_dict: The validated config dict from Langfuse
            - prompt_metadata: The prompt object's metadata dictionary
            
    Raises:
        ConfigValidationError: If Langfuse config is invalid or missing required fields
        Exception: If prompt cannot be loaded from Langfuse
        
    Example usage:
        # Get client and prompts in one call
        client, user_prompt, config, metadata = get_prompt_and_client_from_langfuse(
            prompt_path="m75/sourceCheck/extraction",
            session_id="session_123",
            function=extractMetaSourceFunction  # Keep in code
        )
        
        # Client is ready with system prompt already added
        if user_prompt:
            client.add_user_prompt_text(user_prompt)
        
        # Add more prompts as needed
        client.add_user_prompt_text("Additional context...")
        
        # Call the model
        response = await client.get_gpt_response(session_id, api_endpoint)
    """
    # Step 1: Get prompts and config from Langfuse using existing function
    system_prompt, user_prompt, model_name, prompt_dict = get_prompt_langfuse(
        prompt_path=prompt_path,
        session_id=session_id,
        variables=variables
    )
    
    # Step 2: Extract config from prompt
    try:
        prompt_obj = langfuse.get_prompt(prompt_path, label=os.getenv("LANGFUSE_PROMPT_LABEL"))
        
        if not hasattr(prompt_obj, 'config') or not prompt_obj.config:
            raise ConfigValidationError(
                f"Prompt '{prompt_path}' is missing 'config' field in Langfuse. "
                "Please add a config object with provider, model, and parameters."
            )
        
        config = prompt_obj.config
        
        send_logs(session_id, "LANGFUSE_CONFIG_EXTRACTED",
                 f"• Prompt: {prompt_path}\n• Config: {config}\n• Status: Success",
                 facility="getprompts")
        
    except ConfigValidationError:
        raise
    except Exception as e:
        send_logs(session_id, "LANGFUSE_CONFIG_EXTRACTION_FAILED",
                 f"• Prompt: {prompt_path}\n• Error: {str(e)}\n• Status: Failed",
                 facility="getprompts")
        raise ConfigValidationError(f"Failed to extract config from prompt '{prompt_path}': {e}")
    
    # Step 3: Validate and create client using factory function
    try:
        client = create_client_from_langfuse(
            langfuse_config=config,
            system_prompt=system_prompt or "",
            function=function,
            tool=tool
        )
        
        send_logs(session_id, "CLIENT_CREATED_FROM_LANGFUSE",
                 f"• Prompt: {prompt_path}\n• Provider: {config.get('provider')}\n"
                 f"• Model: {config.get('model')}\n• Client Type: {type(client).__name__}\n• Status: Success",
                 facility="getprompts")
        
    except ConfigValidationError as e:
        send_logs(session_id, "CLIENT_CREATION_FAILED",
                 f"• Prompt: {prompt_path}\n• Config: {config}\n• Error: {str(e)}\n• Status: Failed",
                 facility="getprompts")
        raise
    
    return client, user_prompt, config, prompt_dict
