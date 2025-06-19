# Contributing to NOVA Retrieval VLM

Thank you for your interest in contributing to the NOVA Retrieval VLM framework! This project aims to advance research in medical imaging analysis through retrieval-augmented vision-language models.

## Getting Started

### Development Setup

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/your-username/nova_retrieval_vlm.git
   cd nova_retrieval_vlm
   ```

2. **Set up development environment:**
   ```bash
   # Install dependencies
   poetry install
   
   # Install pre-commit hooks
   poetry run pre-commit install
   
   # Verify setup
   python scripts/setup_check.py --verbose
   ```

3. **Configure environment:**
   ```bash
   # Copy example environment file
   cp .env.example .env
   
   # Add your API keys to .env
   # (see README.md for details)
   ```

### Development Workflow

1. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes and test:**
   ```bash
   # Run tests
   poetry run pytest
   
   # Check code quality
   poetry run black .
   poetry run isort .
   poetry run ruff check .
   poetry run mypy src/
   ```

3. **Commit your changes:**
   ```bash
   git add .
   git commit -m "feat: description of your changes"
   ```

## Types of Contributions

### 🐛 Bug Reports

- Use the GitHub issue template
- Include steps to reproduce
- Provide system information
- Include relevant logs/error messages

### ✨ Feature Requests

- Describe the problem you're solving
- Explain your proposed solution
- Consider backwards compatibility
- Discuss performance implications

### 📝 Documentation

- Fix typos or unclear explanations
- Add examples and tutorials
- Improve API documentation
- Update README or guides

### 🧪 Adding New Models

We welcome contributions of new model adapters! To add support for a new model:

1. **Create model adapter:**
   ```python
   # src/nova_retrieval_vlm/models/your_model_adapter.py
   from .base import BaseAdapter
   
   class YourModelAdapter(BaseAdapter):
       # Implement required methods
       pass
   ```

2. **Add configuration:**
   - Update `ModelConfig` if needed
   - Add model to supported models list
   - Update documentation

3. **Add tests:**
   ```python
   # tests/test_your_model.py
   def test_your_model_adapter():
       # Test implementation
       pass
   ```

### 🔧 Evaluation Metrics

To add new evaluation metrics:

1. **Implement metric:**
   ```python
   # src/nova_retrieval_vlm/evaluation/your_metric.py
   def your_metric(predictions, references):
       # Implement metric calculation
       return score
   ```

2. **Update evaluator:**
   - Add to evaluation pipeline
   - Include in configuration options
   - Add tests and documentation

### 📊 Retrieval Methods

To add new retrieval methods:

1. **Implement retriever:**
   ```python
   # src/nova_retrieval_vlm/retrieval/your_retriever.py
   from .base import BaseRetriever
   
   class YourRetriever(BaseRetriever):
       # Implement retrieval logic
       pass
   ```

2. **Update configuration:**
   - Add to `RetrievalConfig`
   - Update CLI options
   - Add documentation

## Code Standards

### Python Style

- Follow PEP 8
- Use type hints
- Write docstrings for all public functions
- Maximum line length: 88 characters

### Commit Messages

Use conventional commit format:
- `feat:` new features
- `fix:` bug fixes
- `docs:` documentation changes
- `test:` test additions/changes
- `refactor:` code refactoring
- `perf:` performance improvements

### Testing

- Write tests for new functionality
- Maintain >80% code coverage
- Use pytest fixtures for common setups
- Mock external API calls

### Documentation

- Update README.md for user-facing changes
- Add docstrings with examples
- Update type hints
- Include references to papers/methods

## Research Contributions

### Experimental Results

When contributing experimental results:

1. **Include methodology:**
   - Dataset splits used
   - Model configurations
   - Evaluation metrics
   - Hardware specifications

2. **Provide reproducibility:**
   - Configuration files
   - Random seeds
   - Dependency versions
   - Results logs

3. **Statistical significance:**
   - Multiple runs
   - Confidence intervals
   - Statistical tests

### Benchmarking

For benchmark contributions:

1. **Fair comparison:**
   - Same evaluation protocol
   - Consistent preprocessing
   - Proper baselines

2. **Comprehensive evaluation:**
   - Multiple metrics
   - Error analysis
   - Ablation studies

## Review Process

### Pull Request Guidelines

1. **Description:**
   - Clear title and description
   - Link related issues
   - List changes made
   - Include testing information

2. **Code Review:**
   - All CI checks must pass
   - At least one approving review
   - Address reviewer feedback
   - Maintain clean commit history

3. **Testing:**
   - All tests pass
   - New tests for new functionality
   - No decrease in coverage

### Merging

- Use squash and merge for feature branches
- Maintain linear history on main branch
- Delete merged branches

## Community Guidelines

### Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Provide constructive feedback
- Focus on technical merit

### Communication

- Use GitHub issues for bugs/features
- Discussions for questions/ideas
- Be patient and helpful
- Share knowledge and resources

## Getting Help

### Support Channels

- 📖 Check the [Documentation](./docs/)
- 🐛 Search [GitHub Issues](https://github.com/your-org/nova_retrieval_vlm/issues)
- 💬 Join our [Discord Community](https://discord.gg/your-server)
- 📧 Email maintainers for private issues

### Development Questions

- Ask in GitHub Discussions
- Include relevant context
- Share minimal reproducible examples
- Be specific about your environment

## Recognition

Contributors will be:
- Listed in AUTHORS.md
- Mentioned in release notes
- Acknowledged in papers (for significant contributions)
- Invited to co-author on relevant publications

## License

By contributing, you agree that your contributions will be licensed under the same MIT License that covers the project.

---

Thank you for contributing to advancing medical AI research! 🧠✨ 