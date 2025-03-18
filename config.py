import os
from dotenv import load_dotenv
from typing import Optional

# Load environment variables from .env file
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# Ollama Configuration
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
OLLAMA_MAX_TOKENS = int(os.getenv('OLLAMA_MAX_TOKENS', '2048'))

# Function to validate API keys
def validate_api_keys():
    """Validate that required API keys are set."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set in .env file")
    return True

# Function to get OpenAI API key
def get_openai_api_key():
    """Get the OpenAI API key from environment variables."""
    validate_api_keys()
    return OPENAI_API_KEY

# Function to get DeepSeek API key
def get_deepseek_api_key():
    """Get the DeepSeek API key from environment variables."""
    return DEEPSEEK_API_KEY

# Function to get Anthropic API key
def get_anthropic_api_key():
    """Get the Anthropic API key from environment variables."""
    return ANTHROPIC_API_KEY

# Function to get Google API key
def get_google_api_key():
    """Get the Google API key from environment variables."""
    return GOOGLE_API_KEY

# Function to get Ollama host
def get_ollama_host():
    """Get the Ollama host from environment variables."""
    return OLLAMA_HOST

# Function to get Ollama max tokens
def get_ollama_max_tokens():
    """Get the Ollama max tokens from environment variables."""
    return OLLAMA_MAX_TOKENS 