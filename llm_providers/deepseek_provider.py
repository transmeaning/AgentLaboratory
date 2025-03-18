"""
DeepSeek provider implementation.

This module provides an implementation of the LLMProvider interface for DeepSeek models.
"""

import os
import time
import logging
from typing import Dict, List, Optional, Generator, Any
import tiktoken
from openai import OpenAI

from .base import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)

class DeepSeekProvider(LLMProvider):
    """
    LLM provider implementation for DeepSeek.
    
    This provider interacts with the DeepSeek API to provide access to
    DeepSeek's models.
    """
    
    # DeepSeek API base URL
    API_BASE_URL = "https://api.deepseek.com/v1"
    
    # Map of model names to their actual API identifiers
    MODEL_MAPPING = {
        "deepseek-chat": "deepseek-chat",
        "deepseek-coder": "deepseek-coder",
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the DeepSeek provider.
        
        Args:
            api_key: Optional API key for DeepSeek (default: from environment)
        """
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self._client = OpenAI(api_key=self.api_key, base_url=self.API_BASE_URL) if self.api_key else None
        self._available_models = None
    
    def is_available(self) -> bool:
        """
        Check if DeepSeek is available.
        
        Returns:
            bool: True if DeepSeek is available, False otherwise
        """
        if not self.api_key:
            logger.warning("DeepSeek API key is not set")
            return False
        
        try:
            # Try to list models to check if DeepSeek is available
            # DeepSeek doesn't have a list models endpoint, so we'll just check if we can create a client
            return self._client is not None
        except Exception as e:
            logger.warning(f"DeepSeek is not available: {e}")
            return False
    
    def supports_model(self, model_name: str) -> bool:
        """
        Check if DeepSeek supports the specified model.
        
        Args:
            model_name: The name of the model to check
            
        Returns:
            bool: True if DeepSeek supports the model, False otherwise
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
        Get a list of models supported by DeepSeek.
        
        Returns:
            List[str]: A list of model names
        """
        if self._available_models is None:
            # DeepSeek doesn't have a list models endpoint, so we'll just return our known models
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
            # Use cl100k_base as a reasonable approximation for DeepSeek models
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception:
            # Fall back to a simple approximation if tiktoken fails
            return len(text.split())
    
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
        Generate a completion from DeepSeek.
        
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
        
        try:
            # Get the actual model ID
            model_id = self._get_model_id(model)
            
            # Prepare messages
            formatted_messages = []
            
            # Add system prompt if provided
            if system_prompt:
                formatted_messages.append({"role": "system", "content": system_prompt})
            
            # Add messages
            for message in messages:
                # If the first message is a system message and we already added a system prompt,
                # skip it to avoid duplication
                if (
                    message.get("role") == "system" 
                    and system_prompt 
                    and len(formatted_messages) == 1 
                    and formatted_messages[0]["role"] == "system"
                ):
                    continue
                
                formatted_messages.append(message)
            
            # Prepare parameters
            params = {
                "model": model_id,
                "messages": formatted_messages,
            }
            
            if temperature is not None:
                params["temperature"] = temperature
            
            if max_tokens is not None:
                params["max_tokens"] = max_tokens
            
            if timeout is not None:
                params["timeout"] = timeout
            
            # Add any additional parameters
            params.update(kwargs)
            
            # Generate completion
            response = self._client.chat.completions.create(**params)
            
            # Extract content
            content = response.choices[0].message.content
            
            # Get token usage
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
            
            # Create response
            return LLMResponse(
                content=content,
                usage=usage,
                model=model,
                provider="deepseek",
                raw_response=response
            )
        except Exception as e:
            logger.error(f"DeepSeek query failed: {e}")
            raise
        finally:
            elapsed_time = time.time() - start_time
            logger.debug(f"DeepSeek query took {elapsed_time:.2f} seconds")
    
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
        Stream a completion from DeepSeek.
        
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
        try:
            # Get the actual model ID
            model_id = self._get_model_id(model)
            
            # Prepare messages
            formatted_messages = []
            
            # Add system prompt if provided
            if system_prompt:
                formatted_messages.append({"role": "system", "content": system_prompt})
            
            # Add messages
            for message in messages:
                # If the first message is a system message and we already added a system prompt,
                # skip it to avoid duplication
                if (
                    message.get("role") == "system" 
                    and system_prompt 
                    and len(formatted_messages) == 1 
                    and formatted_messages[0]["role"] == "system"
                ):
                    continue
                
                formatted_messages.append(message)
            
            # Prepare parameters
            params = {
                "model": model_id,
                "messages": formatted_messages,
                "stream": True,
            }
            
            if temperature is not None:
                params["temperature"] = temperature
            
            if max_tokens is not None:
                params["max_tokens"] = max_tokens
            
            if timeout is not None:
                params["timeout"] = timeout
            
            # Add any additional parameters
            params.update(kwargs)
            
            # Stream completion
            for chunk in self._client.chat.completions.create(**params):
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"DeepSeek stream failed: {e}")
            raise 