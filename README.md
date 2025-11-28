# Face Recognition System with Anti-Spoofing

A production-grade, real-time face recognition system with advanced anti-spoofing and deepfake detection capabilities. This system provides secure authentication through comprehensive facial analysis, including liveness detection and deepfake prevention.

## 🚀 Features

### Core Capabilities
- **Real-time Face Recognition**: High-accuracy face detection and matching using FaceNet embeddings
- **Anti-Spoofing Protection**: CNN-based liveness detection to prevent photo/video attacks
- **Deepfake Detection**: Vision Transformer (ViT) based synthetic face detection
- **Large-scale Matching**: Efficient similarity search using FAISS vector database
- **Multi-Factor Authentication**: Optional OTP verification via email/SMS
- **RESTful API**: FastAPI-based endpoints for easy integration

### Security Features
- **Comprehensive Pipeline**: Face capture → Liveness → Deepfake → Identity matching
- **Confidence Scoring**: Multi-layered confidence calculation with configurable thresholds
- **Rate Limiting**: Protection against brute force attacks
- **Audit Logging**: Complete authentication attempt logging
- **Data Protection**: Secure storage of biometric templates

### Technical Highlights
- **Modular Architecture**: Clean, SOLID-principle based design
- **Async Processing**: Non-blocking operations for better performance
- **Configuration Management**: YAML/JSON config with environment variable support
- **Production Ready**: Comprehensive logging, monitoring, and error handling
- **Scalable Design**: Supports horizontal scaling and load balancing

## 📋 System Requirements

### Hardware Requirements
- **Minimum**: 4GB RAM, 2-core CPU
- **Recommended**: 8GB RAM, 4-core CPU, GPU (CUDA compatible)
- **Storage**: 5GB free space for models and data

### Software Requirements
- **Python**: 3.10 or higher
- **Operating System**: Windows 10+, Ubuntu 18.04+, macOS 10.15+
- **Webcam**: For real-time face capture
- **Internet**: For downloading pretrained models (initial setup)

## 🛠️ Installation

### 1. Clone Repository
```bash
git clone https://github.com/your-username/face-recognition-system.git
cd face-recognition-system
```

### 2. Create Virtual Environment
```bash
# Using venv
python -m venv face_recognition_env

# Activate environment
# Windows
face_recognition_env\Scripts\activate
# Linux/Mac
source face_recognition_env/bin/activate
```

### 3. Install Dependencies
```bash
# Install all required packages
pip install -r requirements.txt

# For GPU support (optional)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install faiss-gpu
```

### 4. Create Configuration
```bash
# Generate sample configuration
python main.py --create-sample-config

# Edit configuration file
# config.yaml will be created with default settings
```

### 5. Initialize Database
```bash
# The database will be automatically initialized on first run
# Or manually initialize:
python -c "from src.database_manager import DatabaseManager; db = DatabaseManager(); print('Database initialized')"
```

## ⚙️ Configuration

The system uses a hierarchical configuration system supporting YAML files and environment variables.

### Configuration File Structure
```yaml
models:
  device: "cpu"  # or "cuda"
  embedding_model: "vggface2"
  embedding_dim: 512
  liveness_threshold: 0.5
  deepfake_threshold: 0.5

database:
  sqlite_db_path: "data/face_recognition.db"
  faiss_index_path: "data/embeddings/face_index.faiss"
  backup_enabled: true

authentication:
  face_similarity_threshold: 0.7
  overall_confidence_threshold: 0.6
  enable_mfa: false
  max_attempts_per_hour: 5

api:
  host: "0.0.0.0"
  port: 8000
  debug: false
  allowed_origins: ["*"]
```

### Environment Variables
```bash
# Model Configuration
export FACE_RECOGNITION_DEVICE=cuda
export FACE_RECOGNITION_EMBEDDING_MODEL=vggface2

# API Configuration  
export FACE_RECOGNITION_HOST=0.0.0.0
export FACE_RECOGNITION_PORT=8000
export FACE_RECOGNITION_DEBUG=false

# Authentication
export FACE_RECOGNITION_ENABLE_MFA=true
export FACE_RECOGNITION_SENDER_EMAIL=your-email@gmail.com
export FACE_RECOGNITION_SENDER_PASSWORD=your-app-password
```

## 🚀 Quick Start

### 1. Start the System
```bash
# With default configuration
python main.py

# With custom configuration
python main.py --config config.yaml

# With command line overrides
python main.py --host 0.0.0.0 --port 8080 --debug
```

### 2. Access API Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/api/v1/health

### 3. Register a User
```bash
# Using curl
curl -X POST "http://localhost:8000/api/v1/register" \
     -H "Content-Type: application/json" \
     -d '{
       "user_id": "john_doe_001",
       "name": "John Doe",
       "email": "john.doe@example.com",
       "capture_duration": 5
     }'
```

### 4. Authenticate User
```bash
# Using curl
curl -X POST "http://localhost:8000/api/v1/authenticate" \
     -H "Content-Type: application/json" \
     -d '{
       "capture_duration": 3
     }'
```

## 📚 API Reference

### Authentication Endpoints

#### Register User
```http
POST /api/v1/register
Content-Type: application/json

{
  "user_id": "unique_user_id",
  "name": "User Name",
  "email": "user@example.com",
  "phone": "+1-555-123-4567",
  "capture_duration": 5,
  "min_quality_score": 0.7
}
```

#### Authenticate User
```http
POST /api/v1/authenticate
Content-Type: application/json

{
  "capture_duration": 3
}
```

#### Verify MFA
```http
POST /api/v1/verify-mfa
Content-Type: application/json

{
  "user_id": "unique_user_id",
  "otp": "123456"
}
```

### Management Endpoints

#### Get User Information
```http
GET /api/v1/users/{user_id}
Authorization: Bearer <api_key>
```

#### System Statistics
```http
GET /api/v1/stats
Authorization: Bearer <api_key>
```

#### Upload Image for Analysis
```http
POST /api/v1/upload-image
Content-Type: multipart/form-data

file: <image_file>
```

## 🧪 Testing

### Run Unit Tests
```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-cov

# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test category
pytest tests/test_face_capture.py -v
```

### Test API Endpoints
```bash
# Test registration endpoint
python tests/test_api_endpoints.py

# Manual testing with curl
bash tests/test_api_manual.sh
```

### Performance Benchmarking
```bash
# Benchmark model inference times
python -c "
from src.main import FaceRecognitionSystem
system = FaceRecognitionSystem()
system.initialize()
print('Embedding extraction:', system.embedding_extractor.benchmark_inference_time())
print('Liveness detection:', system.liveness_detector.benchmark_inference_time())
"
```

## 📊 System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Face Capture  │    │  Liveness Det.  │    │ Deepfake Det.   │
│   (MTCNN + CV)  │───▶│  (CNN/MobileNet)│───▶│  (ViT/Transformer)│
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Embedding Ext.  │    │ Authentication  │    │    Database     │
│   (FaceNet)     │───▶│    Engine       │───▶│ (SQLite+FAISS)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI       │    │      MFA        │    │   Monitoring    │
│   (REST API)    │    │   (OTP/Email)   │    │   & Logging     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🔧 Development

### Project Structure
```
face_recognition_system/
├── src/                          # Source code
│   ├── face_capture.py          # Face detection and capture
│   ├── embedding_extraction.py  # Face embedding extraction  
│   ├── liveness_detection.py    # Anti-spoofing detection
│   ├── deepfake_detection.py    # Deepfake detection
│   ├── database_manager.py      # Database operations
│   ├── authentication.py        # Authentication logic
│   ├── api.py                   # FastAPI endpoints
│   ├── config.py                # Configuration management
│   └── utils.py                 # Utility functions
├── tests/                       # Unit tests
├── data/                        # Data storage
│   ├── embeddings/              # FAISS index files
│   └── backups/                 # Database backups
├── models/                      # Pretrained models
├── logs/                        # Application logs
├── config/                      # Configuration files
├── main.py                      # Application entry point
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

### Adding New Features

1. **New Detection Method**: Extend `liveness_detection.py` or `deepfake_detection.py`
2. **New API Endpoint**: Add to `api.py` and update documentation
3. **New Configuration**: Update `config.py` dataclasses
4. **New Database Table**: Modify `database_manager.py`

### Code Style
- Follow PEP 8 guidelines
- Use type hints for all functions
- Write comprehensive docstrings
- Maintain test coverage above 80%

## 🔒 Security Considerations

### Data Protection
- Biometric templates are stored as normalized embeddings (not raw images)
- Database encryption at rest (configure in production)
- Secure API key management
- Rate limiting and DDoS protection

### Privacy Compliance
- GDPR compliant data handling
- User consent mechanisms
- Data retention policies
- Right to erasure implementation

### Production Deployment
- Use HTTPS/TLS encryption
- Implement proper authentication (JWT/OAuth)
- Enable audit logging
- Regular security updates

## 🚀 Production Deployment

### Docker Deployment
```dockerfile
# Dockerfile (create this file)
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "main.py", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build and run
docker build -t face-recognition-system .
docker run -p 8000:8000 face-recognition-system
```

### Production Configuration
```yaml
# production_config.yaml
environment: production
api:
  host: "0.0.0.0"
  port: 8000
  debug: false
  workers: 4
  api_key_required: true

logging:
  level: "INFO"
  file_enabled: true
  file_path: "/var/log/face_recognition.log"

database:
  backup_enabled: true
  backup_interval_hours: 6
  max_backups: 30
```

### Load Balancing
```nginx
# nginx.conf
upstream face_recognition {
    server 127.0.0.1:8000;
    server 127.0.0.1:8001;
    server 127.0.0.1:8002;
}

server {
    listen 80;
    location / {
        proxy_pass http://face_recognition;
    }
}
```

## 📈 Performance Optimization

### Model Optimization
- Use quantized models for mobile deployment
- GPU acceleration for batch processing
- Model pruning for reduced memory usage
- TensorRT optimization for NVIDIA GPUs

### Database Optimization
- FAISS index optimization for large datasets
- SQLite WAL mode for better concurrency
- Regular database maintenance and cleanup
- Distributed storage for horizontal scaling

### API Optimization
- Implement caching for frequent requests
- Use connection pooling
- Enable compression
- Implement request queuing for high load

## 🐛 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```bash
# Solution: Use CPU or reduce batch size
export FACE_RECOGNITION_DEVICE=cpu
```

#### 2. Webcam Not Detected
```bash
# Check camera permissions and drivers
# Test with:
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera works:', cap.isOpened())"
```

#### 3. Model Download Failures
```bash
# Manual model download
python -c "from transformers import ViTModel; ViTModel.from_pretrained('google/vit-base-patch16-224')"
```

#### 4. Permission Errors
```bash
# Fix file permissions
chmod +x main.py
sudo chown -R $USER:$USER data/ models/ logs/
```

### Debug Mode
```bash
# Enable detailed logging
python main.py --debug

# Check system status
curl http://localhost:8000/api/v1/health
```

## 📞 Support

### Getting Help
- **Documentation**: Check this README and API docs
- **Issues**: Create GitHub issues for bugs
- **Discussions**: Use GitHub Discussions for questions
- **Email**: contact@example.com (replace with actual)

### Contributing
1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push branch (`git push origin feature/new-feature`)
5. Create Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **FaceNet**: For face embedding architecture
- **MTCNN**: For face detection
- **Hugging Face**: For transformer models
- **FAISS**: For efficient similarity search
- **FastAPI**: For modern API framework

## 📊 Changelog

### v1.0.0 (2024-01-01)
- Initial release
- Core face recognition functionality
- Anti-spoofing and deepfake detection
- RESTful API with FastAPI
- Comprehensive documentation

### v1.1.0 (Planned)
- Mobile app integration
- Real-time video stream processing
- Enhanced security features
- Performance optimizations

---

**Built with ❤️ for secure and reliable face recognition**