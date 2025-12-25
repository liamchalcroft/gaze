# Radiant Harness - Makefile
# Convenient shortcuts for development and evaluation

.PHONY: help install test check clean format lint eval analyze

# Default target
help: ## Show this help message
	@echo "Radiant Harness - Available Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Quick Start:"
	@echo "  1. make install    # Install dependencies"
	@echo "  2. make eval      # Run baseline evaluation"
	@echo "  3. make analyze   # Analyze results"

# Environment setup
install: ## Install dependencies
	@echo "📦 Installing dependencies with uv..."
	uv sync

# Configuration and verification
check: ## Run quality checks
	@echo "🔍 Running quality checks..."
	@./scripts/check_quality.sh

# Testing and development
test: ## Run test suite
	@echo "🧪 Running tests..."
	uv run pytest tests/ -x --tb=short

test-cov: ## Run tests with coverage
	@echo "🧪 Running tests with coverage..."
	uv run pytest tests/ --cov=radiant_harness --cov-report=html

# Code quality
format: ## Format code with ruff
	@echo "🎨 Formatting code..."
	uv run ruff format .

lint: ## Check code quality
	@echo "🔍 Linting code..."
	uv run ruff check .
	uv run pyright src/

quality: format lint ## Format and lint code

# Evaluation workflow
eval: ## Run baseline evaluation
	@echo "🔬 Running baseline evaluation..."
	@./scripts/eval_nova.sh config/baseline.yaml

eval-agentic: ## Run agentic evaluation
	@echo "🤖 Running agentic evaluation..."
	@./scripts/eval_nova.sh config/agentic.yaml

analyze: ## Analyze evaluation results
	@echo "📊 Analyzing evaluation results..."
	@./scripts/eval_nova.sh analyze

full-paper-workflow: eval eval-agentic analyze ## Run complete paper evaluation workflow

# Cleanup
clean: ## Clean temporary files and caches
	@echo "🧹 Cleaning up..."
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -delete
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Cleanup complete"

clean-all: clean ## Clean everything including outputs
	@echo "🧹 Deep cleaning..."
	@rm -rf runs/ outputs/ paper_results/ .hydra/
	@echo "⚠️  Removed all experiment outputs"

# Development utilities
dev-setup: ## Setup development environment with pre-commit hooks
	@echo "👨‍💻 Setting up development environment..."
	uv sync
	pre-commit install

status: ## Show project status
	@echo "📊 Radiant Harness Status"
	@echo "========================="
	@echo "📁 Project Directory: $(PWD)"
	@echo "🐍 Python Version: $(shell python --version 2>/dev/null || echo 'Not found')"
	@echo "📦 uv: $(shell command -v uv >/dev/null 2>&1 && echo 'Available' || echo 'Not found')"
	@echo "🔑 .env file: $(shell [ -f .env ] && echo 'Exists' || echo 'Missing')"
	@echo "📊 Config files: $(shell ls config/*.yaml 2>/dev/null | wc -l) found"
	@echo "📝 Scripts: $(shell ls scripts/*.py 2>/dev/null | wc -l) available"
	@echo ""
	@echo "Available configs:"
	@ls -1 config/*.yaml 2>/dev/null | sed 's/^/  - /' || echo "  No config files found"