FROM python:3.11-slim

WORKDIR /app

RUN pip install uv

COPY . /app

RUN uv pip install --system .

RUN mkdir -p /app/indexes

ENV PYTHONPATH=/app/examples/nova:/app
WORKDIR /app/examples/nova
ENTRYPOINT ["python", "-m", "src.cli"]
