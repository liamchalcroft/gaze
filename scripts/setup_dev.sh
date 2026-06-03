#!/usr/bin/env bash
# Development setup script for GAZE

set -euo pipefail

echo "🚀 Setting up GAZE development environment..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Please install it first:"
    echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Install dependencies
echo "📦 Installing dependencies..."
uv sync

# Install pre-commit hooks
echo "🔧 Installing pre-commit hooks..."
pre-commit install

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    cat > .env << EOF
# API Keys (set these for model access)
OPENAI_API_KEY=your_openai_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Optional: NCBI API key for PubMed search
NCBI_API_KEY=your_ncbi_api_key_here
NCBI_EMAIL=your_email@example.com
EOF
    echo "✅ Created .env file. Please add your API keys."
fi

echo ""
echo "✅ Development environment setup complete!"
echo ""
echo "Next steps:"
echo "1. Add your API keys to .env"
echo "2. Run tests: uv run pytest"
echo "3. Run linting: uv run ruff check ."
echo "4. Run type checking: uv run pyright src/"
echo ""
echo "Happy coding! 🎉"