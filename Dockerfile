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

ENTRYPOINT ["python", "-m", "nova_retrieval_vlm.cli"]
