# NOVA Retrieval VLM Test Suite

This directory contains comprehensive tests for the NOVA Retrieval VLM system.

## Test Categories

### 🧪 **Unit Tests** (No API key required)
- **Type System**: `test_types.py` - Type safety and jaxtyping integration
- **Processors**: `test_processors.py` - Core processor functionality
- **Adapters**: `test_adapters.py` - Model adapter interfaces
- **Metrics**: `test_metrics_*.py` - Evaluation metrics

### 🔗 **Integration Tests** (Optional API key)
- **OpenRouter API**: `test_openrouter_integration.py` - Real API testing
- **Prompt Consistency**: `test_prompt_consistency.py` - Template validation
- **End-to-End**: `test_integration.py` - Complete workflow testing

### ⚡ **Performance Tests**
- **Edge Cases**: `test_edge_cases_and_stress.py` - Stress testing
- **Batch Processing**: `test_batch_processing_utils.py` - Performance benchmarks

## Quick Start

### Run All Tests (No API Key)
```bash
uv run python scripts/run_api_tests.py --test-type unit
```

### Run With OpenRouter API Key
```bash
# Set environment variable
export OPENROUTER_API_KEY="your_api_key_here"
uv run python scripts/run_api_tests.py --test-type all

# Or pass key directly
uv run python scripts/run_api_tests.py --api-key "your_api_key_here" --test-type integration
```

### Run Specific Test Categories
```bash
# Unit tests only
uv run python scripts/run_api_tests.py --test-type unit

# Prompt consistency tests
uv run python scripts/run_api_tests.py --test-type prompt

# Integration tests (requires API key)
uv run python scripts/run_api_tests.py --test-type integration

# Performance tests
uv run python scripts/run_api_tests.py --test-type performance
```

### Run Individual Test Files
```bash
# Run specific test file
uv run pytest tests/test_processors.py -v

# Run specific test method
uv run pytest tests/test_processors.py::TestProcessorConfig::test_processor_config_creation -v
```

## Real API Testing

The OpenRouter integration tests can run against the real API to verify:

1. **Model Compatibility**: Different OpenRouter models work correctly
2. **Clinical History Integration**: Models properly use clinical context
3. **Error Handling**: API errors are handled gracefully
4. **Performance**: Response times and token usage

### Supported Models for Testing
- `anthropic/claude-3.5-sonnet`
- `openai/gpt-4o`
- `google/gemini-pro-1.5`
- And other OpenRouter-compatible models

## Test Coverage

### ✅ **Core Functionality**
- [x] Dataset loading and data integrity
- [x] Model adapter interfaces
- [x] Processor configuration and execution
- [x] Type safety with jaxtyping/beartype
- [x] Prompt template consistency
- [x] Clinical-radiological correlation

### ✅ **API Integration**
- [x] OpenRouter API connectivity
- [x] Multiple model testing
- [x] Error handling and recovery
- [x] Token usage and rate limiting
- [x] Clinical history utilization

### ✅ **Performance & Reliability**
- [x] Batch processing efficiency
- [x] Memory management
- [x] Error boundaries
- [x] Edge case handling
- [x] Stress testing

## Test Configuration

### Environment Variables
```bash
# Required for real API tests
OPENROUTER_API_KEY=your_key_here

# Optional: Enable verbose logging
LOGURU_LEVEL=DEBUG

# Optional: Set test data directory
NOVA_DATA_DIR=./data/nova
```

### Pytest Configuration
See `pytest.ini` for default test configuration.

## Troubleshooting

### API Key Issues
```bash
# Check if API key is set
echo $OPENROUTER_API_KEY

# Test API connectivity
uv run python -c "
import os
from nova_retrieval_vlm.models.openai_adapter import OpenAIAdapter
print('API key found:', bool(os.getenv('OPENROUTER_API_KEY')))
"
```

### Test Failures
1. **Check API quota**: Ensure your OpenRouter account has available credits
2. **Verify model access**: Some models may require specific permissions
3. **Network connectivity**: Ensure internet access for API calls
4. **Dataset access**: Verify NOVA dataset files are available for integration tests

### Performance Issues
- Run tests with `--test-type unit` for fastest execution
- Use `--test-type performance` for specific performance testing
- Check system resources for stress tests

## Contributing

When adding new tests:

1. **Follow naming conventions**: Use descriptive test method names
2. **Include both mock and real API tests** where applicable
3. **Add comprehensive assertions** for expected behavior
4. **Test error conditions** and edge cases
5. **Update this README** with new test categories

## Continuous Integration

These tests are designed to run in CI/CD environments:

- **Unit tests**: Always run, no external dependencies
- **Integration tests**: Run when API key is available
- **Performance tests**: Run on schedule for performance monitoring

## Test Results Interpretation

- **✅ PASSED**: Test completed successfully
- **❌ FAILED**: Test failed - check output for details
- **⚠️ SKIPPED**: Test skipped (usually missing API key or dependencies)
- **🔄 TIMEOUT**: Test took too long - check performance issues