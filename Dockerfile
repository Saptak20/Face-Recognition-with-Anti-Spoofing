FROM python:3.12-slim








# Expose default port (Render injects PORT env var)
EXPOSE 8000

# Health check (optional; Render also uses HTTP health check path)
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD \
    wget -qO- "http://127.0.0.1:${PORT:-8000}/api/v1/health" >/dev/null || exit 1

# Start the API via main orchestrator so all components are initialized
# Use shell form to properly expand PORT environment variable
CMD python main.py --host 0.0.0.0 --port ${PORT:-8000}
