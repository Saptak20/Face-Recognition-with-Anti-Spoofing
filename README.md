# Face Recognition with Anti-Spoofing

A modular **FastAPI-based face recognition and authentication system** that combines face detection, image quality validation, embedding extraction, similarity matching, liveness detection, deepfake detection, MFA, SQLite, and FAISS.

The project has been engineered with a strong focus on **API reliability, deployment readiness, persistent storage, testability, and separation between webcam capture and server-side image processing**.

> **Important:** The current repository is an engineering prototype. The biometric/anti-spoofing ML components still require properly trained and evaluated model weights before this system should be considered suitable for high-security biometric authentication.

---

## ✨ Features

### Face Processing

- Face detection using OpenCV-based detection
- Face extraction and preprocessing
- Face quality validation
- Blur/quality checks
- Single-image processing without requiring a server-side webcam
- Legacy webcam capture support for local use

### Authentication Pipeline

```text
Image / Video Frame
        │
        ▼
┌─────────────────────┐
│   Face Detection    │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  Face Quality Check │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ Face Preprocessing  │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ Embedding Extraction│
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│   FAISS Matching    │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ Authentication      │
│ Decision Engine      │
└──────────┬──────────┘
           ▼
      MFA / Result
````

Optional security stages can additionally include:

```text
Face
 │
 ├── Liveness Detection
 │
 └── Deepfake Detection
```

### Backend

* FastAPI REST API
* Modular service architecture
* Environment-based configuration
* Health-check endpoint
* Structured logging
* Error handling
* API authentication controls
* Rate limiting

### Storage

* SQLite for user and embedding metadata
* FAISS for vector similarity search
* Automatic FAISS consistency validation
* Deterministic FAISS rebuild after deletion
* Atomic FAISS persistence
* Database and FAISS backup support
* Configurable persistent storage paths

### Security

* Configurable face similarity thresholds
* Overall authentication confidence threshold
* Rate limiting
* MFA / OTP support
* Environment-based secrets
* Configurable CORS
* Audit logging
* No secrets committed to source control

### Deployment

* Docker support
* CPU-only production container
* Headless OpenCV
* Render deployment configuration
* Render persistent disk support
* Environment-based production configuration
* Health checks
* Production startup validation
* Configurable memory-saving mode for constrained deployments

---

# 🏗️ Architecture

```text
                         Client
                           │
                           │ Image / Frame
                           ▼
                  ┌───────────────────┐
                  │     FastAPI       │
                  │      REST API     │
                  └─────────┬─────────┘
                            │
              ┌─────────────┴─────────────┐
              │                           │
              ▼                           ▼
      ┌───────────────┐          ┌────────────────┐
      │ Face Processor│          │ Authentication │
      │               │          │     Engine     │
      └───────┬───────┘          └───────┬────────┘
              │                          │
              ▼                          ▼
      ┌───────────────┐          ┌────────────────┐
      │ Face Detection│          │   Liveness     │
      │ & Quality     │          │   Detection    │
      └───────┬───────┘          └───────┬────────┘
              │                          │
              ▼                          ▼
      ┌───────────────┐          ┌────────────────┐
      │  Embedding    │          │    Deepfake    │
      │  Extraction   │          │    Detection   │
      └───────┬───────┘          └───────┬────────┘
              │                          │
              └─────────────┬────────────┘
                            ▼
                  ┌───────────────────┐
                  │ Authentication    │
                  │ Decision Engine    │
                  └─────────┬─────────┘
                            │
                            ▼
                  ┌───────────────────┐
                  │   SQLite + FAISS  │
                  └─────────┬─────────┘
                            │
                            ▼
                  ┌───────────────────┐
                  │ MFA / API Result  │
                  └───────────────────┘
```

---

# 📁 Project Structure

```text
Face-Recognition-with-Anti-Spoofing/
│
├── config/
│   └── config.yaml
│
├── data/
│   ├── embeddings/
│   ├── backups/
│   └── face_recognition.db
│
├── logs/
│
├── models/
│
├── src/
│   ├── api.py
│   ├── authentication.py
│   ├── config.py
│   ├── database_manager.py
│   ├── deepfake_detection.py
│   ├── embedding_extraction.py
│   ├── face_capture.py
│   ├── face_processor.py
│   ├── liveness_detection.py
│   └── utils.py
│
├── tests/
│   ├── test_api.py
│   ├── test_authentication.py
│   ├── test_database_manager.py
│   └── test_face_capture.py
│
├── .env.example
├── .gitignore
├── Dockerfile
├── main.py
├── render.yaml
├── requirements.txt
└── README.md
```

---

# ⚙️ Technology Stack

| Component         | Technology                   |
| ----------------- | ---------------------------- |
| Backend           | FastAPI                      |
| Language          | Python                       |
| Face Processing   | OpenCV                       |
| Deep Learning     | PyTorch                      |
| Embeddings        | MobileNetV2-based extractor  |
| Vector Search     | FAISS                        |
| Database          | SQLite                       |
| Configuration     | YAML + Environment Variables |
| Authentication    | API/JWT/MFA components       |
| Containerization  | Docker                       |
| Deployment        | Render                       |
| Testing           | Pytest                       |
| Production OpenCV | OpenCV Headless              |

---

# 🚀 Quick Start

## 1. Clone the repository

```bash
git clone https://github.com/Saptak20/Face-Recognition-with-Anti-Spoofing.git
cd Face-Recognition-with-Anti-Spoofing
```

---

## 2. Create a virtual environment

Using Python `venv`:

```bash
python3 -m venv .venv
```

Activate it:

### Linux / macOS

```bash
source .venv/bin/activate
```

### Windows

```powershell
.venv\Scripts\activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

For GPU development, install the appropriate PyTorch build separately according to your CUDA environment.

---

## 4. Configure environment variables

Create a local `.env` file based on:

```text
.env.example
```

Never commit `.env`.

Example:

```bash
export FACE_RECOGNITION_ENVIRONMENT=development
export FACE_RECOGNITION_DEVICE=cpu
export FACE_RECOGNITION_DB_PATH=data/face_recognition.db
export FACE_RECOGNITION_FAISS_INDEX_PATH=data/embeddings/face_index.faiss
```

For production, secrets should be supplied through the deployment platform rather than committed to the repository.

---

# ▶️ Running Locally

Start the API:

```bash
python main.py
```

The default API runs on:

```text
http://localhost:8000
```

API documentation:

```text
http://localhost:8000/docs
```

ReDoc:

```text
http://localhost:8000/redoc
```

Health check:

```text
http://localhost:8000/api/v1/health
```

---

# 🔌 API Endpoints

## Health Check

```http
GET /api/v1/health
```

Returns the current health status of the application and its components.

---

## Register Using an Image Frame

```http
POST /api/v1/register-frame
```

Multipart form data:

```text
user_id
name
email
phone
min_quality_score
file
```

Example:

```bash
curl -X POST http://localhost:8000/api/v1/register-frame \
  -F "user_id=test_user_001" \
  -F "name=Test User" \
  -F "email=test@example.com" \
  -F "phone=1234567890" \
  -F "min_quality_score=0.7" \
  -F "file=@face.jpg"
```

---

## Authenticate Using an Image Frame

```http
POST /api/v1/authenticate-frame
```

Example:

```bash
curl -X POST http://localhost:8000/api/v1/authenticate-frame \
  -F "file=@face.jpg"
```

This endpoint is designed around **client-side frame/image capture**, avoiding the architectural problem of trying to access a user's webcam from the cloud server.

---

## Legacy Webcam Endpoints

The project retains webcam-based functionality for local environments.

These endpoints require the machine running the server to have access to a webcam.

For cloud deployment, use the frame-upload endpoints instead.

---

# 🧠 Face Processing

The image-processing pipeline is separated from webcam capture.

```text
Client Frame
     │
     ▼
FaceProcessor
     │
     ├── Detect Face
     ├── Extract Face
     ├── Preprocess
     ├── Validate Quality
     └── Return Processed Result
```

This separation allows the same processing pipeline to work with:

* Webcam frames
* Uploaded images
* Browser camera frames
* API clients
* Automated tests

---

# 🗄️ SQLite + FAISS

The system uses two complementary storage layers.

### SQLite

Stores structured information such as:

* Users
* User metadata
* Embedding metadata
* FAISS vector identifiers
* Authentication-related information

### FAISS

Stores vectors for efficient similarity search.

```text
User
 │
 ├── Metadata ───────────────► SQLite
 │
 └── Face Embedding ────────► FAISS
```

The system maintains synchronization between SQLite and FAISS.

### FAISS consistency

When a user is deleted:

```text
Delete SQLite records
        │
        ▼
Rebuild FAISS index
        │
        ▼
Remap FAISS IDs
        │
        ▼
Persist atomically
```

This prevents stale vectors from remaining searchable after user deletion.

---

# 💾 Persistent Storage

Storage paths are configurable through environment variables.

```bash
FACE_RECOGNITION_DB_PATH
FACE_RECOGNITION_FAISS_INDEX_PATH
FACE_RECOGNITION_BACKUP_PATH
```

Example production configuration:

```text
/var/data/face_recognition.db
/var/data/embeddings/face_index.faiss
/var/data/backups/
```

This allows SQLite and FAISS data to survive container restarts when used with persistent storage.

---

# 🔐 Configuration

Configuration is managed through YAML and environment variables.

Important configuration categories include:

```text
System
Models
Database
Authentication
API
Logging
```

Environment variables take precedence where configured.

Example:

```bash
FACE_RECOGNITION_DEVICE=cpu
FACE_RECOGNITION_DB_PATH=data/face_recognition.db
FACE_RECOGNITION_FAISS_INDEX_PATH=data/embeddings/face_index.faiss
FACE_RECOGNITION_OVERALL_CONFIDENCE_THRESHOLD=0.6
FACE_RECOGNITION_API_KEY_REQUIRED=true
FACE_RECOGNITION_ALLOWED_ORIGINS=http://localhost:3000
```

---

# 🛡️ Memory-Constrained Deployment

The optional liveness and deepfake models can require significant memory.

For constrained deployment environments, the application supports:

```bash
FACE_RECOGNITION_SKIP_OPTIONAL_MODELS=true
```

When enabled, optional model initialization is skipped so that the core face-processing and authentication infrastructure can operate within constrained memory environments.

This mode should **not** be interpreted as providing full anti-spoofing protection.

---

# 🐳 Docker

Build the production image:

```bash
docker build -t face-recognition-with-anti-spoofing .
```

Run locally:

```bash
docker run -p 8000:8000 \
  face-recognition-with-anti-spoofing
```

The production container uses:

* Python 3.12
* CPU-only PyTorch
* Headless OpenCV
* FAISS CPU
* FastAPI
* Environment-driven configuration

The container respects the platform-provided `PORT` environment variable.

---

# ☁️ Render Deployment

The repository includes:

```text
render.yaml
```

The deployment configuration provides:

* Docker-based deployment
* Persistent disk configuration
* Health checks
* Production environment configuration
* Environment-variable based secrets
* Persistent SQLite storage
* Persistent FAISS storage

Health check:

```text
GET /api/v1/health
```

Production secrets should be configured through Render environment variables.

Never commit production credentials to GitHub.

---

# 🧪 Testing

The project currently has a comprehensive automated test suite.

Run:

```bash
pytest -q
```

Current verified result:

```text
89 passed, 1 warning
```

The test suite covers:

* API endpoints
* Authentication
* Face capture
* Face processing
* SQLite operations
* FAISS operations
* FAISS deletion/rebuild behavior
* FAISS ID remapping
* Backup integrity
* Consistency validation
* API error handling
* Persistence behavior

---

# 🧪 API Testing

Interactive API documentation:

```text
http://localhost:8000/docs
```

Health check:

```bash
curl http://localhost:8000/api/v1/health
```

Example image authentication:

```bash
curl -X POST \
  http://localhost:8000/api/v1/authenticate-frame \
  -F "file=@face.jpg"
```

---

# 🔒 Security Architecture

The system includes several defensive layers:

```text
                Authentication Request
                         │
                         ▼
                Face Quality Check
                         │
                         ▼
                 Liveness Check
                         │
                         ▼
                Deepfake Check
                         │
                         ▼
                Face Similarity
                         │
                         ▼
              Overall Confidence
                         │
                         ▼
                    Rate Limit
                         │
                         ▼
                       MFA
                         │
                         ▼
                  Authentication
```

Additional controls include:

* Rate limiting
* Configurable thresholds
* API key support
* JWT-related configuration
* MFA support
* Audit logging
* Environment-based secret management
* Persistent storage controls

---

# ⚠️ Current ML Limitations

The current engineering implementation should **not be represented as a production-grade biometric security system yet**.

The following areas still require proper model training, evaluation, and calibration:

### Face Embeddings

The current embedding pipeline uses a MobileNetV2 backbone with a project-specific projection layer.

The projection layer is not a production identity-recognition model trained specifically for face verification.

### Liveness Detection

The liveness detector currently requires properly trained anti-spoofing weights.

Without trained weights, its output cannot be treated as reliable anti-spoofing evidence.

### Deepfake Detection

The deepfake detector uses a transformer-based architecture, but the classification component requires properly trained/fine-tuned weights before it can be relied upon for security decisions.

### Face Detection

The system can fall back to OpenCV Haar-based detection when the preferred detector assets are unavailable.

---

# 🧭 Development Tracks

The project is being developed in two distinct tracks.

## Track A — Engineering & Deployment

Completed / implemented:

* Modular face processing
* Frame-based API
* Webcam/server separation
* SQLite + FAISS integration
* FAISS consistency handling
* Atomic persistence
* Backup verification
* API error handling
* Automated testing
* Dockerization
* CPU deployment configuration
* Environment-based secrets
* Render deployment configuration
* Persistent storage configuration
* Production health checks

## Track B — ML Security

Planned / ongoing:

* Production-grade face embeddings
* Proper anti-spoofing model
* Proper deepfake detection model
* Model evaluation
* Threshold calibration
* False acceptance / false rejection analysis
* Model versioning
* Security benchmarking

---

# 📈 Future Improvements

* Replace the current embedding extractor with a face-recognition model trained for identity verification
* Integrate a properly trained anti-spoofing model
* Integrate a properly trained deepfake detector
* Add model evaluation and calibration pipelines
* Add ROC/DET evaluation
* Add FAR/FRR measurements
* Add model version management
* Improve face detection with bundled production weights
* Add frontend camera capture application
* Add WebSocket-based real-time verification
* Add stronger persistent database infrastructure for larger deployments
* Add comprehensive observability
* Add CI/CD pipeline
* Add automated deployment verification
* Add security and load testing

---

# 📊 Engineering Milestones

```text
Core Application Repair              ✅
Frame-Based Processing               ✅
Webcam/API Separation                ✅
SQLite + FAISS Integration           ✅
FAISS Consistency & Rebuild          ✅
Atomic Persistence                   ✅
API Error Handling                   ✅
Automated Test Suite                 ✅
Dockerization                        ✅
CPU Production Runtime               ✅
Environment-Based Secrets             ✅
Render Configuration                 ✅
Persistent Storage Configuration      ✅
Production Health Checks             ✅

Production ML Embeddings             ⏳
Real Anti-Spoofing Model              ⏳
Real Deepfake Detection               ⏳
Model Calibration                     ⏳
Security Benchmarking                 ⏳
```

---

# 🤝 Development

Clone the repository:

```bash
git clone https://github.com/Saptak20/Face-Recognition-with-Anti-Spoofing.git
cd Face-Recognition-with-Anti-Spoofing
```

Create a branch:

```bash
git checkout -b feature/your-feature
```

Make your changes and run:

```bash
pytest -q
```

Commit:

```bash
git add .
git commit -m "feat: describe your change"
```

Push:

```bash
git push origin feature/your-feature
```

---

# 📄 License

This project is intended for educational, research, and engineering purposes.

See the repository license file for the applicable licensing terms.

---

# 👨‍💻 Author

**Saptak Mondal**

Computer Science Engineering
AIML & IoT

GitHub:

[https://github.com/Saptak20](https://github.com/Saptak20)

---

# ⭐ Project

**Face Recognition with Anti-Spoofing**

An end-to-end exploration of:

```text
Computer Vision
      +
Deep Learning
      +
Face Recognition
      +
Anti-Spoofing
      +
FastAPI
      +
FAISS
      +
SQLite
      +
Docker
      +
Cloud Deployment
```

Built to explore how a computer-vision authentication system can move from a local prototype toward a reliable, testable, deployable backend architecture.
