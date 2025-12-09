FROM python:3.11-slim

WORKDIR /app

# Install uv for fast dependency management
RUN pip install uv

# Copy project
COPY . /app

# Install dependencies without dev
RUN uv pip install -e .

# Build guideline indexes at build time (stub)
RUN mkdir -p /app/indexes

# Make NOVA example CLI the container entrypoint
ENV PYTHONPATH=/app/examples/nova:/app
WORKDIR /app/examples/nova
ENTRYPOINT ["python", "-m", "src.cli"]
