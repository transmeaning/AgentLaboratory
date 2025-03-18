"""
LLM Manager for coordinating different LLM providers.

This module provides a manager class that coordinates different LLM providers
and provides a unified interface for the rest of the codebase.
"""

import os
import logging
from typing import Dict, List, Optional, Union, Any, Type, Generator

from .base import LLMProvider, LLMResponse
from .ollama_provider import OllamaProvider
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider
from .deepseek_provider import DeepSeekProvider
from .gemini_provider import GeminiProvider

logger = logging.getLogger(__name__)

class LLMManager:
    """
    Manager class for coordinating different LLM providers.
    
    This class provides a unified interface for interacting with different LLM providers,
    handles provider selection, and manages fallbacks.
    """
    
    def __init__(self):
        """Initialize the LLM manager."""
        self._providers = {}
        self._provider_classes = {}
        self._model_to_provider = {}
        
        # Register built-in providers
        self.register_provider("ollama", OllamaProvider)
        self.register_provider("openai", OpenAIProvider)
        self.register_provider("anthropic", AnthropicProvider)
        self.register_provider("deepseek", DeepSeekProvider)
        self.register_provider("gemini", GeminiProvider)
    
    def register_provider(self, provider_name: str, provider_class: Type[LLMProvider]) -> None:
        """
        Register a provider class.
        
        Args:
            provider_name: The name of the provider
            provider_class: The provider class
        """
        self._provider_classes[provider_name] = provider_class
    
    def get_provider(self, provider_name: str) -> LLMProvider:
        """
        Get a provider instance.
        
        Args:
            provider_name: The name of the provider
            
        Returns:
            LLMProvider: The provider instance
            
        Raises:
            ValueError: If the provider is not registered or not available
        """
        if provider_name not in self._providers:
            if provider_name not in self._provider_classes:
                raise ValueError(f"Provider {provider_name} is not registered")
            
            provider_class = self._provider_classes[provider_name]
            provider = provider_class()
            
            if not provider.is_available():
                raise ValueError(f"Provider {provider_name} is not available")
            
            self._providers[provider_name] = provider
            
            # Register models
            for model in provider.get_supported_models():
                self._model_to_provider[model] = provider_name
        
        return self._providers[provider_name]
    
    def get_provider_for_model(self, model_name: str) -> LLMProvider:
        """
        Get the provider for a specific model.
        
        Args:
            model_name: The name of the model
            
        Returns:
            LLMProvider: The provider instance
            
        Raises:
            ValueError: If no provider supports the model
        """
        # Check if we already know which provider supports this model
        if model_name in self._model_to_provider:
            provider_name = self._model_to_provider[model_name]
            return self.get_provider(provider_name)
        
        # Try to find a provider that supports this model
        for provider_name, provider_class in self._provider_classes.items():
            try:
                provider = self.get_provider(provider_name)
                if provider.supports_model(model_name):
                    self._model_to_provider[model_name] = provider_name
                    return provider
            except ValueError:
                continue
        
        raise ValueError(f"No provider supports model {model_name}")
    
    def list_available_providers(self) -> List[str]:
        """
        List available providers.
        
        Returns:
            List[str]: A list of available provider names
        """
        available_providers = []
        
        for provider_name, provider_class in self._provider_classes.items():
            try:
                provider = provider_class()
                if provider.is_available():
                    available_providers.append(provider_name)
            except Exception as e:
                logger.warning(f"Provider {provider_name} is not available: {e}")
        
        return available_providers
    
    def list_available_models(self) -> Dict[str, List[str]]:
        """
        List available models grouped by provider.
        
        Returns:
            Dict[str, List[str]]: A dictionary mapping provider names to lists of model names
        """
        available_models = {}
        
        for provider_name in self.list_available_providers():
            try:
                provider = self.get_provider(provider_name)
                models = provider.get_supported_models()
                if models:
                    available_models[provider_name] = models
            except Exception as e:
                logger.warning(f"Failed to get models for provider {provider_name}: {e}")
        
        return available_models
    
    def query(
        self,
        model: str,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
        provider: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a completion from an LLM.
        
        Args:
            model: The name of the model to use
            messages: A list of message dictionaries with "role" and "content" keys
            system_prompt: An optional system prompt
            temperature: An optional temperature parameter
            max_tokens: An optional maximum number of tokens to generate
            timeout: An optional timeout in seconds
            provider: An optional provider name to use
            **kwargs: Additional provider-specific parameters
            
        Returns:
            LLMResponse: The response from the LLM
            
        Raises:
            Exception: If the query fails
        """
        if provider:
            # Use the specified provider
            provider_instance = self.get_provider(provider)
            if not provider_instance.supports_model(model):
                raise ValueError(f"Provider {provider} does not support model {model}")
        else:
            # Auto-detect provider based on model
            provider_instance = self.get_provider_for_model(model)
        
        return provider_instance.query(
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
        provider: Optional[str] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        Stream a completion from an LLM.
        
        Args:
            model: The name of the model to use
            messages: A list of message dictionaries with "role" and "content" keys
            system_prompt: An optional system prompt
            temperature: An optional temperature parameter
            max_tokens: An optional maximum number of tokens to generate
            timeout: An optional timeout in seconds
            provider: An optional provider name to use
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Generator[str, None, None]: A generator yielding chunks of the response
            
        Raises:
            Exception: If the query fails
        """
        if provider:
            # Use the specified provider
            provider_instance = self.get_provider(provider)
            if not provider_instance.supports_model(model):
                raise ValueError(f"Provider {provider} does not support model {model}")
        else:
            # Auto-detect provider based on model
            provider_instance = self.get_provider_for_model(model)
        
        yield from provider_instance.stream(
            model=model,
            messages=messages,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            **kwargs
        )


# Singleton instance
_llm_manager = None

def get_llm_manager() -> LLMManager:
    """
    Get the singleton LLM manager instance.
    
    Returns:
        LLMManager: The LLM manager instance
    """
    global _llm_manager
    if _llm_manager is None:
        _llm_manager = LLMManager()
    return _llm_manager

def get_llm_provider(provider_name: Optional[str] = None, model_name: Optional[str] = None) -> LLMProvider:
    """
    Get a provider instance.
    
    Args:
        provider_name: The name of the provider
        model_name: The name of the model
        
    Returns:
        LLMProvider: The provider instance
        
    Raises:
        ValueError: If the provider is not registered or not available
    """
    manager = get_llm_manager()
    
    if provider_name:
        return manager.get_provider(provider_name)
    elif model_name:
        return manager.get_provider_for_model(model_name)
    else:
        raise ValueError("Either provider_name or model_name must be specified")

def list_available_providers() -> List[str]:
    """
    List available providers.
    
    Returns:
        List[str]: A list of available provider names
    """
    return get_llm_manager().list_available_providers()

def list_available_models() -> Dict[str, List[str]]:
    """
    List available models grouped by provider.
    
    Returns:
        Dict[str, List[str]]: A dictionary mapping provider names to lists of model names
    """
    return get_llm_manager().list_available_models() 