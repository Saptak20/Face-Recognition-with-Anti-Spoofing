FROM python:3.12-slim



# Health check (optional; Render also uses HTTP health check path)
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD \
    wget -qO- "http://127.0.0.1:${PORT:-8000}/api/v1/health" >/dev/null || exit 1


