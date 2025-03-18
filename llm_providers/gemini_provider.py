"""
Google Gemini provider implementation.

This module provides an implementation of the LLMProvider interface for Google's Gemini models.
"""

import os
import time
import json
import logging
from typing import Dict, List, Optional, Union, Any, Generator

import google.generativeai as genai
from .base import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)

class GeminiProvider(LLMProvider):
    """
    LLM provider implementation for Google Gemini.
    
    This provider interacts with the Google Generative AI API to provide access to
    Gemini models like Gemini Pro and Gemini Ultra.
    """
    
    # Map of model names to their actual API identifiers
    MODEL_MAPPING = {
        "gemini-pro": "models/gemini-1.5-pro",
        "gemini-pro-vision": "models/gemini-pro-vision",
        "gemini-ultra": "models/gemini-1.5-pro",  # Using 1.5-pro as fallback since Ultra isn't available
        "gemini-1.5-pro": "models/gemini-1.5-pro",
        "gemini-1.5-flash": "models/gemini-1.5-flash",
        "gemini-2.0-flash": "models/gemini-2.0-flash",
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Gemini provider.
        
        Args:
            api_key: Optional API key for Google Gemini (default: from environment)
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self._available_models = None
        
        if self.api_key:
            genai.configure(api_key=self.api_key)
    
    def is_available(self) -> bool:
        """
        Check if Gemini is available.
        
        Returns:
            bool: True if Gemini is available, False otherwise
        """
        if not self.api_key:
            logger.warning("Google API key is not set")
            return False
        
        try:
            # Try to list models to check if Gemini is available
            genai.list_models()
            return True
        except Exception as e:
            logger.warning(f"Gemini is not available: {e}")
            return False
    
    def supports_model(self, model: str) -> bool:
        """
        Check if Gemini supports the specified model.
        
        Args:
            model: The name of the model
            
        Returns:
            bool: True if Gemini supports the model, False otherwise
        """
        return model in self.MODEL_MAPPING
    
    def get_available_models(self) -> List[str]:
        """
        Get a list of models supported by Gemini.
        
        Returns:
            List[str]: A list of model names
        """
        if self._available_models is None:
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
    
    def _format_messages(
        self, 
        messages: List[Dict[str, str]], 
        system_prompt: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Format messages for Gemini.
        
        Args:
            messages: A list of message dictionaries with "role" and "content" keys
            system_prompt: An optional system prompt
            
        Returns:
            List[Dict[str, str]]: Formatted messages for Gemini
        """
        formatted_messages = []
        
        # Add system prompt as a system message if provided
        if system_prompt:
            formatted_messages.append({
                "role": "system",
                "parts": [{"text": system_prompt}]
            })
        
        # Add user and assistant messages
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            # Map OpenAI roles to Gemini roles
            if role == "user":
                gemini_role = "user"
            elif role == "assistant":
                gemini_role = "model"
            elif role == "system":
                gemini_role = "system"
            else:
                # Skip unknown roles
                continue
            
            formatted_messages.append({
                "role": gemini_role,
                "parts": [{"text": content}]
            })
        
        return formatted_messages
    
    def _count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text.
        
        Args:
            text: The text to count tokens for
            
        Returns:
            int: The number of tokens
        """
        # Gemini doesn't provide a token counting function, so we'll estimate
        # Rough estimate: 1 token ≈ 4 characters
        return len(text) // 4
    
    def generate_completion(
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
        Generate a completion from Gemini.
        
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
                
                # Create a Gemini model instance
                gemini_model = genai.GenerativeModel(model_id)
                
                # Extract the last user message for the prompt
                user_messages = [msg for msg in messages if msg["role"] == "user"]
                if not user_messages:
                    raise ValueError("No user messages found in the conversation")
                
                prompt = user_messages[-1]["content"]
                
                # Add context from previous messages
                if len(messages) > 1:
                    context = ""
                    for msg in messages[:-1]:
                        role_prefix = "User: " if msg["role"] == "user" else "Assistant: "
                        context += f"{role_prefix}{msg['content']}\n\n"
                    
                    prompt = f"{context}\nUser: {prompt}\n\nAssistant:"
                
                # Add system prompt if provided
                if system_prompt:
                    prompt = f"{system_prompt}\n\n{prompt}"
                
                # Prepare parameters
                params = {}
                
                # Create generation config
                generation_config = {}
                
                if temperature is not None:
                    generation_config["temperature"] = temperature
                
                if max_tokens is not None:
                    generation_config["max_output_tokens"] = max_tokens
                
                # Add generation config if not empty
                if generation_config:
                    params["generation_config"] = generation_config
                
                # Add any additional parameters
                params.update(kwargs)
                
                # Generate completion
                response = gemini_model.generate_content(
                    prompt,
                    **params
                )
                
                # Extract content
                content = response.text
                
                # Estimate token usage
                prompt_tokens = self._count_tokens(prompt)
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
                    provider="gemini",
                    raw_response=response
                )
            except Exception as e:
                error_message = str(e)
                logger.error(f"Gemini query failed: {e}")
                
                # Check if it's a rate limit error
                if "quota" in error_message.lower() and retry < max_retries - 1:
                    delay = base_delay * (2 ** retry)  # Exponential backoff
                    logger.warning(f"Rate limit exceeded. Retrying in {delay} seconds...")
                    time.sleep(delay)
                    continue
                else:
                    # If it's not a rate limit error or we've exhausted retries, raise the exception
                    raise
    
    def stream_completion(
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
        Stream a completion from Gemini.
        
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
                
                # Create a Gemini model instance
                gemini_model = genai.GenerativeModel(model_id)
                
                # Extract the last user message for the prompt
                user_messages = [msg for msg in messages if msg["role"] == "user"]
                if not user_messages:
                    raise ValueError("No user messages found in the conversation")
                
                prompt = user_messages[-1]["content"]
                
                # Add context from previous messages
                if len(messages) > 1:
                    context = ""
                    for msg in messages[:-1]:
                        role_prefix = "User: " if msg["role"] == "user" else "Assistant: "
                        context += f"{role_prefix}{msg['content']}\n\n"
                    
                    prompt = f"{context}\nUser: {prompt}\n\nAssistant:"
                
                # Add system prompt if provided
                if system_prompt:
                    prompt = f"{system_prompt}\n\n{prompt}"
                
                # Prepare parameters
                params = {
                    "stream": True,
                }
                
                # Create generation config
                generation_config = {}
                
                if temperature is not None:
                    generation_config["temperature"] = temperature
                
                if max_tokens is not None:
                    generation_config["max_output_tokens"] = max_tokens
                
                # Add generation config if not empty
                if generation_config:
                    params["generation_config"] = generation_config
                
                # Add any additional parameters
                params.update(kwargs)
                
                # Stream completion
                response = gemini_model.generate_content(
                    prompt,
                    **params
                )
                
                for chunk in response:
                    if chunk.text:
                        yield chunk.text
                
                # If we get here, the streaming was successful, so break out of the retry loop
                break
                
            except Exception as e:
                error_message = str(e)
                logger.error(f"Gemini stream failed: {e}")
                
                # Check if it's a rate limit error
                if "quota" in error_message.lower() and retry < max_retries - 1:
                    delay = base_delay * (2 ** retry)  # Exponential backoff
                    logger.warning(f"Rate limit exceeded. Retrying in {delay} seconds...")
                    time.sleep(delay)
                    continue
                else:
                    # If it's not a rate limit error or we've exhausted retries, raise the exception
                    raise 
    
    def get_supported_models(self) -> List[str]:
        """
        Get a list of models supported by Gemini.
        
        Returns:
            List[str]: A list of model names
        """
        return self.get_available_models()
    
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
        Query the Gemini model.
        
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
        return self.generate_completion(
            model=model,
            messages=messages,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            **kwargs
        )
    
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
        Stream responses from the Gemini model.
        
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
        yield from self.stream_completion(
            model=model,
            messages=messages,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            **kwargs
        ) 