FROM python:3.11-slim

WORKDIR /app

# Install Poetry
RUN pip install poetry

# Copy project
COPY . /app

# Install dependencies without dev
RUN poetry config virtualenvs.create false && \
    poetry install --no-dev --no-interaction --no-ansi

# Build guideline indexes at build time (stub)
RUN mkdir -p /app/indexes

ENTRYPOINT ["python", "-m", "nova_retrieval_vlm.cli"]
