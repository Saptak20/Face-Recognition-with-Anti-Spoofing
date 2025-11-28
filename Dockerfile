FROM python:3.12-slim

# Avoid interactive prompts during package install
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System dependencies for OpenCV, image processing, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (better layer caching)
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r /app/requirements.txt

# Copy application source
COPY . /app

# Create required directories
RUN mkdir -p /app/data /app/logs /app/models

# Expose default port (Render injects PORT env var)
EXPOSE 8000

# Health check (optional; Render also uses HTTP health check path)
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD \
    wget -qO- "http://127.0.0.1:${PORT:-8000}/api/v1/health" >/dev/null || exit 1

# Start the API via main orchestrator so all components are initialized
# Render provides PORT; pass it to main.py explicitly
CMD ["python", "main.py", "--host", "0.0.0.0", "--port", "${PORT:-8000}"]
