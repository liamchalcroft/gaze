# Radiant Harness - Makefile

.PHONY: help install test check clean format lint quality dev-setup status

help: ## Show this help message
	@echo "Radiant Harness - Available Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Quick Start:"
	@echo "  1. make install    # Install dependencies"
	@echo "  2. make check      # Run quality checks"
	@echo "  3. make test       # Run test suite"

install: ## Install dependencies
	uv sync

check: ## Run all quality checks (lint, typecheck, test)
	uv run ruff check src/
	uv run ruff format --check src/
	uv run pyright src/
	uv run pytest tests/ -x --tb=short

test: ## Run test suite
	uv run pytest tests/ -x --tb=short

test-cov: ## Run tests with coverage
	uv run pytest tests/ --cov=radiant_harness --cov-report=html

format: ## Format code with ruff
	uv run ruff format .

lint: ## Check code quality
	uv run ruff check .
	uv run pyright src/

quality: format lint ## Format and lint code

clean: ## Clean temporary files and caches
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -delete
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true

clean-all: clean ## Clean everything including outputs
	@rm -rf runs/ outputs/ paper_results/

dev-setup: ## Setup development environment with pre-commit hooks
	uv sync
	pre-commit install

status: ## Show project status
	@echo "Radiant Harness Status"
	@echo "======================"
	@echo "Project Directory: $(PWD)"
	@echo "Python Version: $(shell python --version 2>/dev/null || echo 'Not found')"
	@echo "uv: $(shell command -v uv >/dev/null 2>&1 && echo 'Available' || echo 'Not found')"
	@echo ".env file: $(shell [ -f .env ] && echo 'Exists' || echo 'Missing')"
