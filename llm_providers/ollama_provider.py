"""
Ollama provider implementation.

This module provides an implementation of the LLMProvider interface for Ollama.
"""

import os
import time
import logging
from typing import Dict, List, Optional, Generator, Any
import tiktoken
import ollama

from .base import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)

class OllamaProvider(LLMProvider):
    """
    LLM provider implementation for Ollama.
    
    This provider interacts with a local Ollama instance to provide access to
    locally running LLM models.
    """
    
    def __init__(self, base_url: Optional[str] = None):
        """
        Initialize the Ollama provider.
        
        Args:
            base_url: Optional base URL for the Ollama API (default: http://localhost:11434)
        """
        self.base_url = base_url or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self._client = ollama.Client(host=self.base_url)
        self._available_models = None
    
    def is_available(self) -> bool:
        """
        Check if Ollama is available.
        
        Returns:
            bool: True if Ollama is available, False otherwise
        """
        try:
            # Try to list models to check if Ollama is available
            self._client.list()
            return True
        except Exception as e:
            logger.warning(f"Ollama is not available: {e}")
            return False
    
    def supports_model(self, model_name: str) -> bool:
        """
        Check if Ollama supports the specified model.
        
        Args:
            model_name: The name of the model to check
            
        Returns:
            bool: True if Ollama supports the model, False otherwise
        """
        try:
            models = self.get_supported_models()
            logger.info(f"Checking if Ollama supports model: {model_name}")
            logger.info(f"Available models: {models}")
            result = model_name in models
            logger.info(f"Model {model_name} is {'supported' if result else 'not supported'}")
            return result
        except Exception as e:
            logger.warning(f"Error checking if Ollama supports model {model_name}: {e}")
            return False
    
    def get_supported_models(self) -> List[str]:
        """
        Get a list of models supported by Ollama.
        
        Returns:
            List[str]: A list of model names
        """
        if self._available_models is None:
            try:
                response = self._client.list()
                self._available_models = []
                
                # Handle the ListResponse object from the Ollama Python client
                if hasattr(response, 'models'):
                    # Extract model names from the models attribute
                    for model in response.models:
                        if hasattr(model, 'model'):
                            self._available_models.append(model.model)
                # Fallback for other response formats
                elif isinstance(response, dict) and "models" in response:
                    self._available_models = [model["name"] for model in response["models"]]
                elif isinstance(response, list):
                    self._available_models = [model.get("name") for model in response if isinstance(model, dict) and "name" in model]
                else:
                    logger.warning(f"Unexpected Ollama response format: {type(response)}")
                    logger.warning(f"Response content: {response}")
            except Exception as e:
                logger.warning(f"Failed to get Ollama models: {e}")
                self._available_models = []
        
        return self._available_models
    
    def _count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text.
        
        Args:
            text: The text to count tokens for
            
        Returns:
            int: The number of tokens
        """
        try:
            # Use cl100k_base as a reasonable default for most models
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception:
            # Fallback to a simple approximation if tiktoken fails
            return len(text.split())
    
    def _format_messages(self, messages: List[Dict[str, str]], system_prompt: Optional[str] = None) -> str:
        """
        Format messages for Ollama.
        
        Args:
            messages: A list of message dictionaries with "role" and "content" keys
            system_prompt: An optional system prompt
            
        Returns:
            str: The formatted prompt
        """
        formatted_prompt = ""
        
        # Add system prompt if provided
        if system_prompt:
            formatted_prompt += f"<system>\n{system_prompt}\n</system>\n\n"
        
        # Add messages
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system" and not system_prompt:
                formatted_prompt += f"<system>\n{content}\n</system>\n\n"
            elif role == "user":
                formatted_prompt += f"<human>\n{content}\n</human>\n\n"
            elif role == "assistant":
                formatted_prompt += f"<assistant>\n{content}\n</assistant>\n\n"
        
        # Add final assistant prompt
        formatted_prompt += "<assistant>\n"
        
        return formatted_prompt
    
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
        Generate a completion from Ollama.
        
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
            # Format messages for Ollama
            prompt = self._format_messages(messages, system_prompt)
            logger.info(f"Querying Ollama model: {model}")
            
            # Prepare options
            options = {}
            
            if temperature is not None:
                options["temperature"] = temperature
            
            if max_tokens is not None:
                options["num_predict"] = max_tokens
            else:
                # Use a default max_tokens value from environment or a reasonable default
                options["num_predict"] = int(os.getenv("OLLAMA_MAX_TOKENS", "2048"))
            
            # Add any additional options
            if "options" in kwargs:
                options.update(kwargs.pop("options"))
            
            # Prepare parameters
            params = {
                "model": model,
                "prompt": prompt,
                "options": options,
            }
            
            # Add system prompt if provided
            if system_prompt:
                params["system"] = system_prompt
            
            # Add any additional parameters
            params.update(kwargs)
            
            logger.info(f"Ollama query parameters: {params}")
            
            # Generate completion
            try:
                response = self._client.generate(**params)
                logger.info(f"Ollama response type: {type(response)}")
                
                # Log response details
                if hasattr(response, "__dict__"):
                    logger.info(f"Response __dict__: {response.__dict__}")
                
                # Extract response content based on response format
                if hasattr(response, "response"):
                    content = response.response
                    logger.info(f"Response content from attribute: {content[:100]}...")
                elif isinstance(response, dict) and "response" in response:
                    content = response["response"]
                    logger.info(f"Response content from dict: {content[:100]}...")
                else:
                    logger.warning(f"Unexpected response type: {type(response)}")
                    content = str(response)
                    logger.info(f"Response content from str: {content[:100]}...")
                
                # Count tokens
                prompt_tokens = self._count_tokens(prompt)
                completion_tokens = self._count_tokens(content)
                
                # Create response
                llm_response = LLMResponse(
                    content=content,
                    usage={
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                    model=model,
                    provider="ollama",
                    raw_response=response
                )
                
                logger.info(f"Created LLMResponse with content: {llm_response.content[:100]}...")
                return llm_response
            except Exception as e:
                logger.error(f"Ollama generate call failed: {e}")
                raise
        except Exception as e:
            logger.error(f"Ollama query failed: {e}")
            logger.error(f"Exception type: {type(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
        finally:
            elapsed_time = time.time() - start_time
            logger.info(f"Ollama query took {elapsed_time:.2f} seconds")
    
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
        Stream a completion from Ollama.
        
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
            # Format messages for Ollama
            prompt = self._format_messages(messages, system_prompt)
            logger.info(f"Streaming from Ollama model: {model}")
            
            # Prepare options
            options = {}
            
            if temperature is not None:
                options["temperature"] = temperature
            
            if max_tokens is not None:
                options["num_predict"] = max_tokens
            else:
                # Use a default max_tokens value from environment or a reasonable default
                options["num_predict"] = int(os.getenv("OLLAMA_MAX_TOKENS", "2048"))
            
            # Add any additional options
            if "options" in kwargs:
                options.update(kwargs.pop("options"))
            
            # Prepare parameters
            params = {
                "model": model,
                "prompt": prompt,
                "stream": True,
                "options": options,
            }
            
            # Add system prompt if provided
            if system_prompt:
                params["system"] = system_prompt
            
            # Add any additional parameters
            params.update(kwargs)
            
            logger.info(f"Ollama stream parameters: {params}")
            
            # Stream completion
            for chunk in self._client.generate(**params):
                if hasattr(chunk, "response"):
                    yield chunk.response
                elif isinstance(chunk, dict) and "response" in chunk:
                    yield chunk["response"]
                else:
                    logger.warning(f"Unexpected chunk type: {type(chunk)}")
        except Exception as e:
            logger.error(f"Ollama stream failed: {e}")
            raise 