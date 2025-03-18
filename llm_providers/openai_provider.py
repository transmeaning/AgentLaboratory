"""
OpenAI provider implementation.

This module provides an implementation of the LLMProvider interface for OpenAI.
"""

import os
import time
import logging
from typing import Dict, List, Optional, Generator, Any
import tiktoken
from openai import OpenAI

from .base import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)

class OpenAIProvider(LLMProvider):
    """
    LLM provider implementation for OpenAI.
    
    This provider interacts with the OpenAI API to provide access to
    OpenAI's models like GPT-4o, GPT-4o-mini, etc.
    """
    
    # Map of model names to their actual API identifiers
    MODEL_MAPPING = {
        "gpt-4o": "gpt-4o-2024-08-06",
        "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
        "o1-preview": "o1-preview",
        "o1-mini": "o1-mini-2024-09-12",
        "o1": "o1-2024-12-17",
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the OpenAI provider.
        
        Args:
            api_key: Optional API key for OpenAI (default: from environment)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._client = OpenAI(api_key=self.api_key) if self.api_key else None
        self._available_models = None
    
    def is_available(self) -> bool:
        """
        Check if OpenAI is available.
        
        Returns:
            bool: True if OpenAI is available, False otherwise
        """
        if not self.api_key:
            logger.warning("OpenAI API key is not set")
            return False
        
        try:
            # Try to list models to check if OpenAI is available
            self._client.models.list()
            return True
        except Exception as e:
            logger.warning(f"OpenAI is not available: {e}")
            return False
    
    def supports_model(self, model_name: str) -> bool:
        """
        Check if OpenAI supports the specified model.
        
        Args:
            model_name: The name of the model to check
            
        Returns:
            bool: True if OpenAI supports the model, False otherwise
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
        Get a list of models supported by OpenAI.
        
        Returns:
            List[str]: A list of model names
        """
        if self._available_models is None:
            try:
                # Start with our known models
                self._available_models = list(self.MODEL_MAPPING.keys())
                
                # Add any additional models from the API
                response = self._client.models.list()
                for model in response.data:
                    model_id = model.id
                    # Only add models that are not already in our list
                    if model_id not in self._available_models and model_id not in self.MODEL_MAPPING.values():
                        self._available_models.append(model_id)
            except Exception as e:
                logger.warning(f"Failed to get OpenAI models: {e}")
                # Fall back to just our known models
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
    
    def _count_tokens(self, text: str, model: str) -> int:
        """
        Count the number of tokens in a text.
        
        Args:
            text: The text to count tokens for
            model: The model to count tokens for
            
        Returns:
            int: The number of tokens
        """
        try:
            # Try to get the encoding for the model
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
        except Exception:
            try:
                # Fall back to cl100k_base for newer models
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
        Generate a completion from OpenAI.
        
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
                provider="openai",
                raw_response=response
            )
        except Exception as e:
            logger.error(f"OpenAI query failed: {e}")
            raise
        finally:
            elapsed_time = time.time() - start_time
            logger.debug(f"OpenAI query took {elapsed_time:.2f} seconds")
    
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
        Stream a completion from OpenAI.
        
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
            logger.error(f"OpenAI stream failed: {e}")
            raise 