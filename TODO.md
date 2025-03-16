# TODO List

## High Priority

### ArXiv Search Retry Mechanism
- [ ] Review interaction between requests.Session retry mechanism and manual retries
- [ ] Consider separating retry logic for different failure modes (HTTP vs Connection)
- [ ] Add more detailed logging in the retry mechanism
- [ ] Implement a more robust mock setup for the entire request chain

### Research Session
- [ ] Run one after installing Gemma
- [ ]
- [ ]

## Medium Priority

### Documentation
- [ ] Complete the Table of Contents in documentation.md
- [ ] Add examples of common research workflows to End-User Guide Section
- [ ] Add detailed API documentation for LLM Providers Module
- [ ] Document the new modular agent architecture

### Testing
- [ ] Add integration tests for the complete research workflow
- [ ] Implement performance benchmarks for LLM operations
- [ ] Add test coverage reporting

## Low Priority

### Features
- [ ] Implement progress tracking for long-running operations
- [ ] Add support for custom LLM provider configurations
- [ ] Create visualization tools for research metrics

### Maintenance
- [ ] Review and update dependencies
- [ ] Optimize token usage in LLM calls
- [ ] Clean up deprecated code in inference.py 