FROM python:3.12-slim

# Install system dependencies for OpenCV headless operation and FAISS
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for headless operation
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OPENCV_OPENCL_RUNTIME="" \
    QT_QPA_PLATFORM=offscreen

WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install CPU-only PyTorch and dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    torch==2.5.1+cpu \
    torchvision==0.20.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY src/ ./src/
COPY main.py .
COPY config/ ./config/

# Create data directories (will be mounted as volumes in production)
RUN mkdir -p /app/data/embeddings /app/data/backups /app/logs /app/models

# Expose the default port (Render will override via PORT env var)
EXPOSE 8000

# Health check using the API health endpoint
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=30s \
    CMD python -c "import urllib.request; import os; \
    port = os.environ.get('PORT', '8000'); \
    urllib.request.urlopen(f'http://127.0.0.1:{port}/api/v1/health', timeout=5)" || exit 1

# Start the application - use shell to expand PORT env var
CMD python main.py --host 0.0.0.0 --port ${PORT:-8000}