# Changelog
All notable changes to AgentLaboratory will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- New `test_arxiv_search.py` file implementing comprehensive unit tests for ArxivSearch functionality
  - Tests for successful paper retrieval
  - Tests for invalid paper ID handling
  - Tests for connection error retry mechanism
  - Tests for cleanup after error conditions
- New `config.py` for centralized configuration management
  - Environment variable loading using python-dotenv
  - API key validation and safe access functions
  - Structured configuration approach for better maintainability

### Changed
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