import time, tiktoken
import os, json
import logging
from typing import Dict, List, Optional, Any

from llm_providers import get_llm_manager, LLMResponse

# Set up logging
logger = logging.getLogger(__name__)

# Token tracking for cost estimation
TOKENS_IN = dict()
TOKENS_OUT = dict()

# Use cl100k_base encoding for token counting
encoding = tiktoken.get_encoding("cl100k_base")

def curr_cost_est():
    """
    Calculate the current cost estimate based on token usage.
    
    Returns:
        float: The estimated cost in USD
    """
    costmap_in = {
        "gpt-4o": 2.50 / 1000000,
        "gpt-4o-mini": 0.150 / 1000000,
        "o1-preview": 15.00 / 1000000,
        "o1-mini": 3.00 / 1000000,
        "claude-3-5-sonnet": 3.00 / 1000000,
        "deepseek-chat": 1.00 / 1000000,
        "o1": 15.00 / 1000000,
        "gemini-pro": 0.125 / 1000000,
        "gemini-ultra": 1.25 / 1000000,
        "gemini-1.5-pro": 0.25 / 1000000,
        "gemini-1.5-flash": 0.125 / 1000000,
        "gemini-2.0-flash": 0.175 / 1000000,
    }
    costmap_out = {
        "gpt-4o": 10.00/ 1000000,
        "gpt-4o-mini": 0.6 / 1000000,
        "o1-preview": 60.00 / 1000000,
        "o1-mini": 12.00 / 1000000,
        "claude-3-5-sonnet": 12.00 / 1000000,
        "deepseek-chat": 5.00 / 1000000,
        "o1": 60.00 / 1000000,
        "gemini-pro": 0.375 / 1000000,
        "gemini-ultra": 3.75 / 1000000,
        "gemini-1.5-pro": 0.75 / 1000000,
        "gemini-1.5-flash": 0.375 / 1000000,
        "gemini-2.0-flash": 0.525 / 1000000,
    }
    
    total_cost = 0.0
    for model in TOKENS_IN:
        # Use a default cost of 0 for models not in the cost maps
        in_cost = costmap_in.get(model, 0.0)
        out_cost = costmap_out.get(model, 0.0)
        total_cost += in_cost * TOKENS_IN[model] + out_cost * TOKENS_OUT[model]
    
    return total_cost

def query_model(
    model_str: str, 
    prompt: str, 
    system_prompt: str, 
    openai_api_key: Optional[str] = None, 
    anthropic_api_key: Optional[str] = None, 
    google_api_key: Optional[str] = None,
    tries: int = 5, 
    timeout: float = 5.0, 
    temp: Optional[float] = None, 
    print_cost: bool = True,
    version: str = "1.5"
) -> str:
    """
    Query an LLM model using the unified provider architecture.
    
    Args:
        model_str: The name of the model to use
        prompt: The user prompt
        system_prompt: The system prompt
        openai_api_key: Optional OpenAI API key
        anthropic_api_key: Optional Anthropic API key
        google_api_key: Optional Google API key
        tries: Number of retries on failure
        timeout: Timeout between retries in seconds
        temp: Optional temperature parameter
        print_cost: Whether to print cost estimates
        version: API version (legacy parameter, kept for compatibility)
        
    Returns:
        str: The model's response
        
    Raises:
        Exception: If the query fails after max retries
    """
    # Set API keys in environment if provided
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
    if anthropic_api_key:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
    if google_api_key:
        os.environ["GOOGLE_API_KEY"] = google_api_key
    
    # Normalize model name
    model = normalize_model_name(model_str)
    
    # Format messages
    messages = [{"role": "user", "content": prompt}]
    
    # Try to query the model with retries
    for attempt in range(tries):
        try:
            # Get the LLM manager
            llm_manager = get_llm_manager()
            
            # Query the model
            logger.info(f"Querying model {model} (attempt {attempt+1}/{tries})")
            response = llm_manager.query(
                model=model,
                messages=messages,
                system_prompt=system_prompt,
                temperature=temp,
                max_tokens=None,  # Let the provider use its default
                timeout=None,     # Let the provider use its default
            )
            
            logger.info(f"Received response of type {type(response)}")
            logger.info(f"Response content: {response.content[:100]}...")
            
            # Update token tracking for cost estimation
            update_token_tracking(model, response)
            
            # Print cost estimate if requested
            if print_cost:
                print(f"Current experiment cost = ${curr_cost_est()}, ** Approximate values, may not reflect true cost")
            
            # Return the response content
            return response.content
        except Exception as e:
            logger.error(f"Inference attempt {attempt+1}/{tries} failed: {e}")
            if attempt < tries - 1:
                time.sleep(timeout)
                continue
            else:
                raise Exception(f"Max retries: timeout after {tries} attempts")

def normalize_model_name(model_str: str) -> str:
    """
    Normalize model name to a standard format.
    
    Args:
        model_str: The input model name
        
    Returns:
        str: The normalized model name
    """
    # Map of aliases to standard model names
    model_aliases = {
        "gpt4omini": "gpt-4o-mini",
        "gpt-4omini": "gpt-4o-mini",
        "gpt4o-mini": "gpt-4o-mini",
        "gpt4o": "gpt-4o",
        "claude-3.5-sonnet": "claude-3-5-sonnet",
        "gemini-pro": "gemini-pro",
        "gemini": "gemini-pro",
        "gemini-ultra": "gemini-ultra",
        "gemini-1.5-pro": "gemini-1.5-pro",
        "gemini-1.5-flash": "gemini-1.5-flash",
        "gemini-2.0-flash": "gemini-2.0-flash",
        "gemini-2.0": "gemini-2.0-flash",
    }
    
    # Normalize to lowercase
    model_str = model_str.lower()
    
    # Return the standard name if it's an alias
    return model_aliases.get(model_str, model_str)

def update_token_tracking(model: str, response: LLMResponse) -> None:
    """
    Update token tracking for cost estimation.
    
    Args:
        model: The model name
        response: The LLMResponse object
    """
    # Initialize token tracking for the model if not already done
    if model not in TOKENS_IN:
        TOKENS_IN[model] = 0
        TOKENS_OUT[model] = 0
    
    # Update token counts
    TOKENS_IN[model] += response.usage.get("prompt_tokens", 0)
    TOKENS_OUT[model] += response.usage.get("completion_tokens", 0)

def stream_model(
    model_str: str, 
    prompt: str, 
    system_prompt: str, 
    openai_api_key: Optional[str] = None, 
    anthropic_api_key: Optional[str] = None, 
    google_api_key: Optional[str] = None,
    temp: Optional[float] = None
):
    """
    Stream responses from an LLM model using the unified provider architecture.
    
    Args:
        model_str: The name of the model to use
        prompt: The user prompt
        system_prompt: The system prompt
        openai_api_key: Optional OpenAI API key
        anthropic_api_key: Optional Anthropic API key
        google_api_key: Optional Google API key
        temp: Optional temperature parameter
        
    Returns:
        Generator: A generator yielding chunks of the response
        
    Raises:
        Exception: If the query fails
    """
    # Set API keys in environment if provided
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
    if anthropic_api_key:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
    if google_api_key:
        os.environ["GOOGLE_API_KEY"] = google_api_key
    
    # Normalize model name
    model = normalize_model_name(model_str)
    
    # Format messages
    messages = [{"role": "user", "content": prompt}]
    
    # Get the LLM manager
    llm_manager = get_llm_manager()
    
    # Stream the response
    return llm_manager.stream(
        model=model,
        messages=messages,
        system_prompt=system_prompt,
        temperature=temp,
        max_tokens=None,  # Let the provider use its default
        timeout=None,     # Let the provider use its default
    )

#print(query_model(model_str="o1-mini", prompt="hi", system_prompt="hey"))