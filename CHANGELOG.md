# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Flexible LLM provider architecture with support for multiple providers
- Ollama integration for using local LLM models
- Support for Anthropic Claude models
- Improved DeepSeek provider implementation
- Test script for verifying LLM provider functionality
- Streaming support for all LLM providers
- Documentation for setting up and using different LLM providers
- New `test_arxiv_search.py` file implementing comprehensive unit tests for ArxivSearch functionality
  - Tests for successful paper retrieval
  - Tests for cleanup after error conditions
- New `config.py` for centralized configuration management
  - Environment variable loading using python-dotenv
  - API key validation and safe access functions
  - Structured configuration approach for better maintainability

### Changed
- Refactored inference.py to use the new LLM provider architecture
- Updated configuration system to support multiple providers
- Improved error handling and retry logic for LLM queries
- Enhanced token counting and cost estimation
- Updated parameter ordering in `mlesolver.py` get_score function calls for consistency
  - Changed from `openai_api_key, REWARD_MODEL_LLM` to `REWARD_MODEL_LLM, openai_api_key`
  - Affects three locations in the file
  - No functional changes, improves code readability
- Enhanced `.gitignore` configuration
  - Added patterns for sensitive files (.env, .env.*)
  - Added IDE-specific directories (.cursor/, .vscode/)
  - Added logs directory and log files (logs/, *.log)
  - Reorganized patterns into logical sections

### Security
- Improved sensitive file handling through .gitignore updates
  - Prevents accidental commit of API keys in .env files
  - Excludes log files that might contain sensitive information
  - Blocks IDE-specific files that might contain local configurations

## [0.1.0] - 2024-01-15

### Added
- Initial release of Agent Laboratory
- Support for OpenAI and DeepSeek models
- Basic research workflow with literature review, experimentation, and report writing
- ArXiv integration for paper retrieval
- Python execution environment for experiments
- LaTeX support for report generation

### Changed
- Updated parameter ordering in `mlesolver.py` get_score function calls for consistency
  - Changed from `