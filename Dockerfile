# Container for the NOVA example CLI.
# For reproducible, supply-chain-resistant builds, pin the base image by digest:
#   FROM python:3.11-slim@sha256:<digest>
FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

# Copy only dependency metadata + the package first for better layer caching.
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/

# The NOVA example entrypoint imports torch et al., so install the nova extra.
RUN uv pip install --system ".[nova]"

# Example code is not part of the wheel; copy it explicitly.
COPY examples/nova/ ./examples/nova/

RUN mkdir -p /app/indexes

# Run as an unprivileged user (defence in depth: this process decodes
# untrusted images and fetches remote content).
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser /app
USER appuser

ENV PYTHONPATH=/app/examples/nova:/app
WORKDIR /app/examples/nova
ENTRYPOINT ["python", "-m", "src.cli"]
