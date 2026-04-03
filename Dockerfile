# Stage 1: Build dependencies (includes gcc for C extensions)
FROM python:3.13-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt psycopg2-binary

# Stage 2: Runtime (no gcc, smaller image)
FROM python:3.13-slim

# Only runtime libs needed (libpq for psycopg2)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 curl && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r botuser && useradd -r -g botuser -m botuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application
COPY . .

# Create directories for persistent data
RUN mkdir -p data logs && chown -R botuser:botuser /app

USER botuser

EXPOSE 8080

# V12 15.1: Ensure Docker sends SIGINT (not SIGTERM) so Python KeyboardInterrupt
# and our signal handler in main.py trigger a clean graceful shutdown.
STOPSIGNAL SIGINT

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["python", "main.py"]
