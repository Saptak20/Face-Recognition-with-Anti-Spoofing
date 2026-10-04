# AGENTS.md - Face Recognition with Anti-Spoofing

## Quick Commands
```bash
# Install dependencies (CPU)
pip install -r requirements.txt

# Install with GPU support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install faiss-gpu

# Run system
python main.py                    # default config
python main.py --config config.yaml
python main.py --host 0.0.0.0 --port 8080 --debug

# Create sample config
python main.py --create-sample-config

# Initialize DB manually
python -c "from src.database_manager import DatabaseManager; DatabaseManager(); print('DB initialized')"
```

## Testing
```bash
# Install test deps
pip install pytest pytest-asyncio pytest-cov

# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test
pytest tests/test_face_capture.py -v
```

## Architecture
- **Entry**: `main.py` → `FaceRecognitionSystem` class
- **Config**: `src/config.py` loads YAML/JSON + env vars (prefix: `FACE_RECOGNITION_`)
- **Models**: Face capture (MTCNN) → Embedding (FaceNet) → Liveness (CNN) → Deepfake (ViT)
- **Storage**: SQLite (metadata) + FAISS (embeddings) in `data/`
- **API**: FastAPI at `/api/v1/*`, docs at `/docs`, health at `/api/v1/health`

## Key Files
| File | Purpose |
|------|---------|
| `main.py` | App entry, CLI args, system orchestration |
| `src/config.py` | ConfigManager, dataclasses, env var mapping |
| `src/authentication.py` | AuthEngine - core pipeline logic |
| `src/api.py` | FastAPI routes, middleware |
| `config/config.yaml` | Default development config |

## Environment Variables
```bash
FACE_RECOGNITION_DEVICE=cuda
FACE_RECOGNITION_EMBEDDING_MODEL=vggface2
FACE_RECOGNITION_HOST=0.0.0.0
FACE_RECOGNITION_PORT=8000
FACE_RECOGNITION_DEBUG=false
FACE_RECOGNITION_ENABLE_MFA=true
FACE_RECOGNITION_SENDER_EMAIL=...
FACE_RECOGNITION_SENDER_PASSWORD=...
```

## Docker / Deploy
- `Dockerfile` uses `python:3.12-slim`, health check on `/api/v1/health`
- `render.yaml` for Render.com (auto-deploy, starter plan)
- Data dirs (`data/embeddings/`, `data/backups/`, `logs/*.log`) are gitignored

## Gotchas
- `sys.path.append("src")` in main.py - imports use `from src.module import ...`
- Models auto-download on first run (Hugging Face, torchvision)
- CUDA OOM → set `FACE_RECOGNITION_DEVICE=cpu`
- No pytest config file - uses defaults
- No lint/typecheck configured (add ruff/mypy if needed)