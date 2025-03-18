"""
Anthropic provider implementation.

This module provides an implementation of the LLMProvider interface for Anthropic's Claude models.
"""

import os
import time
import json
import logging
from typing import Dict, List, Optional, Generator, Any
import tiktoken
import anthropic

from .base import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)

class AnthropicProvider(LLMProvider):
    """
    LLM provider implementation for Anthropic.
    
    This provider interacts with the Anthropic API to provide access to
    Claude models like Claude 3.5 Sonnet.
    """
    
    # Map of model names to their actual API identifiers
    MODEL_MAPPING = {
        "claude-3-5-sonnet": "claude-3-5-sonnet-latest",
        "claude-3-opus": "claude-3-opus-20240229",
        "claude-3-sonnet": "claude-3-sonnet-20240229",
        "claude-3-haiku": "claude-3-haiku-20240307",
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Anthropic provider.
        
        Args:
            api_key: Optional API key for Anthropic (default: from environment)
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._client = anthropic.Anthropic(api_key=self.api_key) if self.api_key else None
        self._available_models = None
    
    def is_available(self) -> bool:
        """
        Check if Anthropic is available.
        
        Returns:
            bool: True if Anthropic is available, False otherwise
        """
        if not self.api_key:
            logger.warning("Anthropic API key is not set")
            return False
        
        try:
            # Try to list models to check if Anthropic is available
            # Anthropic doesn't have a list models endpoint, so we'll just check if we can create a client
            return self._client is not None
        except Exception as e:
            logger.warning(f"Anthropic is not available: {e}")
            return False
    
    def supports_model(self, model_name: str) -> bool:
        """
        Check if Anthropic supports the specified model.
        
        Args:
            model_name: The name of the model to check
            
        Returns:
            bool: True if Anthropic supports the model, False otherwise
        """
        # Normalize model name
        model_name = model_name.lower()
        
        # Check if the model is in our mapping
        if model_name in self.MODEL_MAPPING:
            return True
        
        # Check if the model is available directly
        try:
            models = self.get_supported_models()
            return model_name in models
        except Exception:
            return False
    
    def get_supported_models(self) -> List[str]:
        """
        Get a list of models supported by Anthropic.
        
        Returns:
            List[str]: A list of model names
        """
        if self._available_models is None:
            # Anthropic doesn't have a list models endpoint, so we'll just return our known models
            self._available_models = list(self.MODEL_MAPPING.keys())
        
        return self._available_models
    
    def _get_model_id(self, model_name: str) -> str:
        """
        Get the actual model ID to use with the API.
        
        Args:
            model_name: The name of the model
            
        Returns:
            str: The actual model ID
        """
        return self.MODEL_MAPPING.get(model_name, model_name)
    
    def _count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text.
        
        Args:
            text: The text to count tokens for
            
        Returns:
            int: The number of tokens
        """
        try:
            # Use cl100k_base as a reasonable approximation for Claude models
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception:
            # Fall back to a simple approximation if tiktoken fails
            return len(text.split())
    
    def _format_messages(self, messages: List[Dict[str, str]], system_prompt: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Format messages for Anthropic.
        
        Args:
            messages: A list of message dictionaries with "role" and "content" keys
            system_prompt: An optional system prompt
            
        Returns:
            List[Dict[str, str]]: The formatted messages
        """
        formatted_messages = []
        
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "user":
                formatted_messages.append({"role": "user", "content": content})
            elif role == "assistant":
                formatted_messages.append({"role": "assistant", "content": content})
            # Ignore system messages, as they are handled separately
        
        return formatted_messages
    
    def query(
        self,
        model: str,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a completion from Anthropic.
        
        Args:
            model: The name of the model to use
            messages: A list of message dictionaries with "role" and "content" keys
            system_prompt: An optional system prompt
            temperature: An optional temperature parameter
            max_tokens: An optional maximum number of tokens to generate
            timeout: An optional timeout in seconds
            **kwargs: Additional provider-specific parameters
            
        Returns:
            LLMResponse: The response from the LLM
            
        Raises:
            Exception: If the query fails
        """
        start_time = time.time()
        
        # Add retry mechanism with exponential backoff for rate limits
        max_retries = 5
        base_delay = 2  # seconds
        
        for retry in range(max_retries):
            try:
                # Get the actual model ID
                model_id = self._get_model_id(model)
                
                # Format messages
                formatted_messages = self._format_messages(messages, system_prompt)
                
                # Prepare parameters
                params = {
                    "model": model_id,
                    "messages": formatted_messages,
                    "max_tokens": max_tokens or 4096,  # Ensure max_tokens is always provided with a default value
                }
                
                # Add system prompt if provided
                if system_prompt:
                    params["system"] = system_prompt
                
                if temperature is not None:
                    params["temperature"] = temperature
                
                # Add any additional parameters
                params.update(kwargs)
                
                # Generate completion
                response = self._client.messages.create(**params)
                
                # Extract content
                content = json.loads(response.to_json())["content"][0]["text"]
                
                # Estimate token usage (Anthropic doesn't provide this directly)
                prompt_text = system_prompt or ""
                for message in formatted_messages:
                    prompt_text += message.get("content", "")
                
                prompt_tokens = self._count_tokens(prompt_text)
                completion_tokens = self._count_tokens(content)
                
                # Create response
                return LLMResponse(
                    content=content,
                    usage={
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                    model=model,
                    provider="anthropic",
                    raw_response=response
                )
            except Exception as e:
                error_message = str(e)
                logger.error(f"Anthropic query failed: {e}")
                
                # Check if it's a rate limit error
                if "rate_limit_error" in error_message and retry < max_retries - 1:
                    delay = base_delay * (2 ** retry)  # Exponential backoff
                    logger.warning(f"Rate limit exceeded. Retrying in {delay} seconds...")
                    time.sleep(delay)
                    continue
                else:
                    # If it's not a rate limit error or we've exhausted retries, raise the exception
                    raise
            finally:
                elapsed_time = time.time() - start_time
                logger.debug(f"Anthropic query took {elapsed_time:.2f} seconds")
    
    def stream(
        self,
        model: str,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        Stream a completion from Anthropic.
        
        Args:
            model: The name of the model to use
            messages: A list of message dictionaries with "role" and "content" keys
            system_prompt: An optional system prompt
            temperature: An optional temperature parameter
            max_tokens: An optional maximum number of tokens to generate
            timeout: An optional timeout in seconds
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Generator[str, None, None]: A generator yielding chunks of the response
            
        Raises:
            Exception: If the query fails
        """
        # Add retry mechanism with exponential backoff for rate limits
        max_retries = 5
        base_delay = 2  # seconds
        
        for retry in range(max_retries):
            try:
                # Get the actual model ID
                model_id = self._get_model_id(model)
                
                # Format messages
                formatted_messages = self._format_messages(messages, system_prompt)
                
                # Prepare parameters
                params = {
                    "model": model_id,
                    "messages": formatted_messages,
                    "stream": True,
                    "max_tokens": max_tokens or 4096,  # Ensure max_tokens is always provided with a default value
                }
                
                # Add system prompt if provided
                if system_prompt:
                    params["system"] = system_prompt
                
                if temperature is not None:
                    params["temperature"] = temperature
                
                # Remove redundant max_tokens assignment
                # if max_tokens is not None:
                #     params["max_tokens"] = max_tokens
                
                # Add any additional parameters
                params.update(kwargs)
                
                # Stream completion
                with self._client.messages.stream(**params) as stream:
                    for text in stream.text_stream:
                        yield text
                        
                # If we get here, the streaming was successful, so break out of the retry loop
                break
                
            except Exception as e:
                error_message = str(e)
                logger.error(f"Anthropic stream failed: {e}")
                
                # Check if it's a rate limit error
                if "rate_limit_error" in error_message and retry < max_retries - 1:
                    delay = base_delay * (2 ** retry)  # Exponential backoff
                    logger.warning(f"Rate limit exceeded. Retrying in {delay} seconds...")
                    time.sleep(delay)
                    continue
                else:
                    # If it's not a rate limit error or we've exhausted retries, raise the exception
                    raise 