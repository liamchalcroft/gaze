# Container for the NOVA example CLI.
# Base image pinned by digest for reproducible, supply-chain-resistant builds.
# Refresh with: docker manifest inspect python:3.11-slim
FROM python:3.11-slim@sha256:a3ab0b966bc4e91546a033e22093cb840908979487a9fc0e6e38295747e49ac0

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

# Smoke check: confirm the package imports inside the image.
HEALTHCHECK --interval=1m --timeout=10s --retries=3 \
    CMD python -c "import gaze" || exit 1

ENTRYPOINT ["python", "-m", "src.cli"]
