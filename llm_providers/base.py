"""
Base classes for LLM providers.

This module defines the abstract interface that all LLM providers must implement,
as well as common utility classes and functions.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any, Generator


@dataclass
class LLMResponse:
    """
    Data class representing a response from an LLM.
    
    Attributes:
        content: The text content of the response
        usage: Token usage statistics
        model: The name of the model used
        provider: The name of the provider (e.g., "openai", "anthropic", "ollama")
        raw_response: The raw response object from the provider
    """
    content: str
    usage: Dict[str, int]
    model: str
    provider: str
    raw_response: Optional[Any] = None


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    All LLM providers must implement this interface.
    """
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if this provider is available (API keys set, service reachable, etc.)
        
        Returns:
            bool: True if the provider is available, False otherwise
        """
        pass
    
    @abstractmethod
    def supports_model(self, model_name: str) -> bool:
        """
        Check if this provider supports the specified model.
        
        Args:
            model_name: The name of the model to check
            
        Returns:
            bool: True if the provider supports the model, False otherwise
        """
        pass
    
    @abstractmethod
    def get_supported_models(self) -> List[str]:
        """
        Get a list of models supported by this provider.
        
        Returns:
            List[str]: A list of model names
        """
        pass
    
    @abstractmethod
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
        Generate a completion from the LLM.
        
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
        pass
    
    @abstractmethod
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
        Stream a completion from the LLM.
        
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
        pass 