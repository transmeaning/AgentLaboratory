import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')

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