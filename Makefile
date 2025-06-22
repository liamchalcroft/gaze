# NOVA Retrieval VLM - Makefile
# Convenient shortcuts for development and experimentation

.PHONY: help install test check setup clean experiments quick-test format lint docs

# Default target
help: ## Show this help message
	@echo "NOVA Retrieval VLM - Available Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Quick Start:"
	@echo "  1. make setup    # Initial setup"
	@echo "  2. make check    # Verify configuration"
	@echo "  3. make quick    # Run quick test"
	@echo "  4. make exp      # Run full experiments"

# Environment setup
setup: ## Complete project setup (install deps, create .env, etc.)
	@echo "🚀 Setting up NOVA Retrieval VLM..."
	@if command -v uv >/dev/null 2>&1; then \
	echo "📦 Installing dependencies with uv..."; \
	uv pip install -e .; \
	else \
	echo "📦 Installing dependencies with pip..."; \
	pip install -e .; \
	fi
	@if [ ! -f .env ]; then \
		echo "⚙️  Creating .env file from template..."; \
		cp .env.example .env || echo "Please create .env file manually"; \
		echo "🔑 Please edit .env and add your API keys"; \
	fi
	@echo "✅ Setup complete! Run 'make check' to verify."

install: ## Install dependencies only
	@if command -v uv >/dev/null 2>&1; then \
	uv pip install -e .; \
	else \
	pip install -e .; \
	fi

# Configuration and verification
check: ## Check setup and configuration
	@echo "🔍 Checking NOVA Retrieval VLM setup..."
	@python scripts/setup_check.py --verbose

check-fix: ## Check setup and attempt to fix issues automatically
	@echo "🔧 Checking and fixing NOVA Retrieval VLM setup..."
	@python scripts/setup_check.py --verbose --fix

env: ## Create .env file from template
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "✅ Created .env file. Please edit it with your API keys."; \
	else \
		echo "⚠️  .env file already exists."; \
	fi

# Data management
download: ## Download NOVA dataset
	@echo "📥 Downloading NOVA dataset..."
	@python scripts/download_nova.py --data-dir ${DATA_DIR:-./data/nova}

index: ## Build retrieval indexes
	@echo "🔍 Building retrieval indexes..."
	@python scripts/build_index.py

data: download index ## Download dataset and build indexes

# Testing and development
test: ## Run test suite
	@echo "🧪 Running tests..."
	@if command -v uv >/dev/null 2>&1; then \
	pytest; \
	else \
	pytest; \
	fi

test-cov: ## Run tests with coverage
	@echo "🧪 Running tests with coverage..."
	@if command -v uv >/dev/null 2>&1; then \
	pytest --cov=nova_retrieval_vlm --cov-report=html; \
	else \
	pytest --cov=nova_retrieval_vlm --cov-report=html; \
	fi
	
quick: ## Run quick test with free model
	@echo "⚡ Running quick test..."
	@bash scripts/run_experiments.sh quick

# Code quality
format: ## Format code with black and isort
	@echo "🎨 Formatting code..."
	@if command -v uv >/dev/null 2>&1; then \
	black .; \
	isort .; \
	else \
	black .; \
	isort .; \
	fi

lint: ## Check code quality
	@echo "🔍 Linting code..."
	@if command -v uv >/dev/null 2>&1; then \
	ruff check .; \
	mypy src/; \
	else \
	ruff check .; \
	mypy src/; \
	fi

quality: format lint ## Format and lint code

# Experiments
exp: ## Run full experiment suite
	@echo "🧪 Running full experiment suite..."
	@bash scripts/run_experiments.sh full

experiments: exp ## Alias for exp

exp-viz: ## Run visualization experiments
	@echo "🎨 Running visualization experiments..."
	@bash scripts/run_experiments.sh viz

# Individual tasks
localization: ## Run localization task
	@echo "🎯 Running localization task..."
	@python -m nova_retrieval_vlm.cli task=localization model.name=openai/gpt-4o-mini:free

caption: ## Run caption task
	@echo "💬 Running caption task..."
	@python -m nova_retrieval_vlm.cli task=caption model.name=openai/gpt-4o-mini:free

diagnosis: ## Run diagnosis task
	@echo "🏥 Running diagnosis task..."
	@python -m nova_retrieval_vlm.cli task=diagnosis model.name=openai/gpt-4o-mini:free

# Cleanup
clean: ## Clean temporary files and caches
	@echo "🧹 Cleaning up..."
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -delete
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Cleanup complete"

clean-all: clean ## Clean everything including outputs and data
	@echo "🧹 Deep cleaning..."
	@rm -rf runs/ outputs/ indexes/ .hydra/
	@echo "⚠️  Removed all experiment outputs and indexes"

# Documentation
docs: ## Generate documentation
	@echo "📚 Generating documentation..."
	@echo "📖 README.md is the main documentation"
	@echo "📖 See CONTRIBUTING.md for development guidelines"

# Development utilities
dev-setup: ## Setup development environment with pre-commit hooks
	@echo "👨‍💻 Setting up development environment..."
	@if command -v uv >/dev/null 2>&1; then \
	uv pip install -e .[dev]; \
	pre-commit install; \
	else \
	pip install -e .[dev]; \
	pre-commit install; \
	fi

demo: ## Run a quick demo of all tasks
	@echo "🎭 Running demonstration of all tasks..."
	@echo "📍 Localization..."
	@python -m nova_retrieval_vlm.cli task=localization model.name=openai/gpt-4o-mini:free max_iterations=1
	@echo "💬 Caption..."
	@python -m nova_retrieval_vlm.cli task=caption model.name=openai/gpt-4o-mini:free max_iterations=1
	@echo "🏥 Diagnosis..."
	@python -m nova_retrieval_vlm.cli task=diagnosis model.name=openai/gpt-4o-mini:free max_iterations=1

status: ## Show project status
	@echo "📊 NOVA Retrieval VLM Status"
	@echo "============================"
	@echo "📁 Project Directory: $(PWD)"
	@echo "🐍 Python Version: $(shell python --version 2>/dev/null || echo 'Not found')"
@echo "📦 uv: $(shell command -v uv >/dev/null 2>&1 && echo 'Available' || echo 'Not found')"
	@echo "🔑 .env file: $(shell [ -f .env ] && echo 'Exists' || echo 'Missing')"
	@echo "📊 Data directory: $(shell [ -d data ] && echo 'Exists' || echo 'Missing')"
	@echo "🔍 Indexes: $(shell [ -d indexes ] && echo 'Built' || echo 'Not built')"
	@echo ""
	@echo "Run 'make check' for detailed verification"

# Variable examples for users
show-vars: ## Show available environment variables
	@echo "🔧 Environment Variables:"
	@echo "DATA_DIR=${DATA_DIR:-./data/nova}"
	@echo "OUTPUT_DIR=${OUTPUT_DIR:-./runs}"
	@echo "OPENROUTER_API_KEY=${OPENROUTER_API_KEY:-not_set}"
	@echo "OPENAI_API_KEY=${OPENAI_API_KEY:-not_set}" 