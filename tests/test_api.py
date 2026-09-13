"""
Test FastAPI Endpoints

Unit and integration tests for the FastAPI-based REST API endpoints under /api/v1/
including registration, authentication, MFA, user management, and system stats.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import sys
from pathlib import Path
from datetime import datetime
import json
import io
import requests
import asyncio
from PIL import Image

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.api import FaceRecognitionAPI


class ASGIResponse:
    """Lightweight response object wrapping ASGI response output."""
    def __init__(self, status_code: int, headers: dict, content: bytes):
        self.status_code = status_code
        self.headers = headers
        self.content = content
        self.text = content.decode('utf-8', errors='replace')

    def json(self):
        return json.loads(self.text)


class ASGIClient:
    """
    In-process ASGI HTTP test client.
    Uses requests.Request to prepare multipart, urlencoded, or JSON payloads,
    and executes them directly against the ASGI app without requiring httpx or live networking.
    """
    def __init__(self, app):
        self.app = app

    def request(self, method: str, url: str, params=None, data=None, json_data=None, files=None, headers=None):
        if not url.startswith('http'):
            url = f'http://testserver{url}'

        req = requests.Request(
            method=method.upper(),
            url=url,
            params=params,
            data=data,
            json=json_data,
            files=files,
            headers=headers
        ).prepare()

        parsed_url = requests.utils.urlparse(req.url)
        path = parsed_url.path or '/'
        query_string = (parsed_url.query or '').encode('latin1')
        body = req.body or b''
        if isinstance(body, str):
            body = body.encode('utf-8')
        elif hasattr(body, 'read'):
            body = body.read()

        raw_headers = [(k.lower().encode('latin1'), v.encode('latin1')) for k, v in req.headers.items()]

        resp_headers = {}
        resp_status = None
        resp_body = []

        scope = {
            'type': 'http',
            'asgi': {'version': '3.0'},
            'http_version': '1.1',
            'method': method.upper(),
            'path': path,
            'raw_path': path.encode('latin1'),
            'query_string': query_string,
            'headers': raw_headers,
            'client': ('127.0.0.1', 12345),
            'server': ('127.0.0.1', 80),
            'scheme': 'http',
        }

        sent = False
        async def receive():
            nonlocal sent
            if not sent:
                sent = True
                return {'type': 'http.request', 'body': body, 'more_body': False}
            return {'type': 'http.disconnect'}

        async def send(msg):
            nonlocal resp_status, resp_headers, resp_body
            if msg['type'] == 'http.response.start':
                resp_status = msg['status']
                for k, v in msg.get('headers', []):
                    resp_headers[k.decode('latin1')] = v.decode('latin1')
            elif msg['type'] == 'http.response.body':
                resp_body.append(msg.get('body', b''))

        asyncio.run(self.app(scope, receive, send))
        return ASGIResponse(resp_status, resp_headers, b''.join(resp_body))

    def get(self, url: str, **kwargs):
        json_data = kwargs.pop('json', None)
        return self.request('GET', url, json_data=json_data, **kwargs)

    def post(self, url: str, **kwargs):
        json_data = kwargs.pop('json', None)
        return self.request('POST', url, json_data=json_data, **kwargs)

    def delete(self, url: str, **kwargs):
        json_data = kwargs.pop('json', None)
        return self.request('DELETE', url, json_data=json_data, **kwargs)

    def put(self, url: str, **kwargs):
        json_data = kwargs.pop('json', None)
        return self.request('PUT', url, json_data=json_data, **kwargs)


def create_test_image_bytes(color='blue', size=(160, 160)) -> bytes:
    """Helper to create valid JPEG image bytes for upload tests."""
    img = Image.new('RGB', size, color=color)
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    return buf.getvalue()


@pytest.fixture
def mock_components():
    """Create mock components for testing."""
    return {
        'face_capture': Mock(),
        'embedding_extractor': Mock(),
        'liveness_detector': Mock(),
        'deepfake_detector': Mock(),
        'database_manager': Mock(),
        'auth_engine': Mock()
    }


@pytest.fixture
def api_instance(mock_components):
    """Create API instance with mock components configured."""
    api = FaceRecognitionAPI({
        'allowed_origins': ['*'],
        'api_key_required': False
    })
    api.set_components(**mock_components)
    return api


@pytest.fixture
def client(api_instance):
    """Create ASGI client for testing API endpoints."""
    return ASGIClient(api_instance.app)


class TestRootAndHealthEndpoints:
    """Tests for root and health check endpoints."""

    def test_root_endpoint(self, client):
        """Test root endpoint returns API metadata and available endpoints."""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert data["message"] == "Face Recognition System API"
        assert data["version"] == "1.0.0"
        assert data["status"] == "active"
        assert "endpoints" in data
        assert data["endpoints"]["health_check"] == "/api/v1/health"
        assert data["endpoints"]["registration"] == "/api/v1/register"
        assert data["endpoints"]["authentication"] == "/api/v1/authenticate"

    def test_health_check_healthy(self, client):
        """Test health check returns healthy when all components are set."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert all(data["components"].values())
        assert "timestamp" in data
        assert "uptime" in data

    def test_health_check_unhealthy(self):
        """Test health check returns unhealthy when components are missing."""
        api_uninitialized = FaceRecognitionAPI({})
        uninit_client = ASGIClient(api_uninitialized.app)

        response = uninit_client.get("/api/v1/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "unhealthy"
        assert not all(data["components"].values())


class TestUserRegistrationEndpoints:
    """Tests for user registration endpoints (webcam and frame upload)."""

    def test_register_user_webcam_success(self, client, mock_components):
        """Test successful user registration via JSON / webcam endpoint."""
        mock_components['auth_engine'].register_user.return_value = {
            'success': True,
            'message': 'User registered successfully',
            'user_id': 'john_doe_001',
            'embedding_id': 'emb-uuid-1234',
            'quality_score': 0.92,
            'total_faces_captured': 5,
            'valid_faces_processed': 4
        }

        response = client.post(
            "/api/v1/register",
            json={
                "user_id": "john_doe_001",
                "name": "John Doe",
                "email": "john@example.com",
                "capture_duration": 3,
                "min_quality_score": 0.7
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["user_id"] == "john_doe_001"
        assert data["embedding_id"] == "emb-uuid-1234"
        assert data["quality_score"] == 0.92

    def test_register_user_webcam_failure(self, client, mock_components):
        """Test user registration failure when auth engine rejects."""
        mock_components['auth_engine'].register_user.return_value = {
            'success': False,
            'message': 'User already exists or database error',
            'user_id': 'john_doe_001'
        }

        response = client.post(
            "/api/v1/register",
            json={
                "user_id": "john_doe_001",
                "name": "John Doe"
            }
        )

        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "User already exists" in data["detail"]

    def test_register_user_missing_required_fields(self, client):
        """Test registration returns 422 when required fields are missing."""
        response = client.post(
            "/api/v1/register",
            json={"name": "John Doe"}  # Missing user_id
        )
        assert response.status_code == 422

    def test_register_user_engine_not_initialized(self):
        """Test registration returns 500 when auth engine is uninitialized."""
        api = FaceRecognitionAPI({})
        client = ASGIClient(api.app)

        response = client.post(
            "/api/v1/register",
            json={"user_id": "john_doe_001", "name": "John Doe"}
        )
        assert response.status_code == 500

    def test_register_frame_success(self, client, mock_components):
        """Test successful user registration via image frame upload."""
        mock_components['auth_engine'].register_user.return_value = {
            'success': True,
            'message': 'User registered successfully',
            'user_id': 'frame_user_001',
            'embedding_id': 'emb-frame-5678',
            'quality_score': 0.88,
            'total_faces_captured': 1,
            'valid_faces_processed': 1
        }

        image_bytes = create_test_image_bytes(color='green')

        response = client.post(
            "/api/v1/register-frame",
            data={
                "user_id": "frame_user_001",
                "name": "Frame User",
                "email": "frame@example.com",
                "min_quality_score": "0.7"
            },
            files={
                "file": ("face.jpg", image_bytes, "image/jpeg")
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["user_id"] == "frame_user_001"
        assert data["embedding_id"] == "emb-frame-5678"

        # Verify auth_engine was called with frames keyword arg
        mock_components['auth_engine'].register_user.assert_called_once()
        call_kwargs = mock_components['auth_engine'].register_user.call_args[1]
        assert 'frames' in call_kwargs
        assert len(call_kwargs['frames']) == 1

    def test_register_frame_invalid_image(self, client):
        """Test registration frame upload rejects corrupted / non-image files."""
        response = client.post(
            "/api/v1/register-frame",
            data={
                "user_id": "bad_img_user",
                "name": "Bad Image User"
            },
            files={
                "file": ("bad.jpg", b"not-a-valid-image-bytes", "image/jpeg")
            }
        )
        assert response.status_code == 400
        assert "Invalid image file" in response.json()["detail"]


class TestUserAuthenticationEndpoints:
    """Tests for authentication endpoints (webcam and frame upload)."""

    def test_authenticate_user_webcam_success(self, client, mock_components):
        """Test successful authentication via webcam endpoint."""
        mock_components['auth_engine'].authenticate_user.return_value = {
            'success': True,
            'message': 'Authentication successful',
            'user_id': 'john_doe_001',
            'name': 'John Doe',
            'confidence': 0.85,
            'face_similarity': 0.88,
            'liveness_score': 0.82,
            'deepfake_score': 0.80,
            'processing_time': 0.45
        }

        response = client.post(
            "/api/v1/authenticate",
            json={"capture_duration": 3}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["user_id"] == "john_doe_001"
        assert data["confidence"] == 0.85
        assert data["face_similarity"] == 0.88

    def test_authenticate_user_webcam_failure(self, client, mock_components):
        """Test failed authentication returns unsuccessful response."""
        mock_components['auth_engine'].authenticate_user.return_value = {
            'success': False,
            'message': 'No matching user found',
            'confidence': 0.0,
            'liveness_score': 0.85,
            'deepfake_score': 0.90
        }

        response = client.post(
            "/api/v1/authenticate",
            json={"capture_duration": 3}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "No matching user found" in data["message"]

    def test_authenticate_user_mfa_required(self, client, mock_components):
        """Test authentication response when MFA is triggered."""
        mock_components['auth_engine'].authenticate_user.return_value = {
            'success': False,
            'message': 'Multi-factor authentication required',
            'mfa_required': True,
            'user_id': 'john_doe_001',
            'confidence': 0.85,
            'face_similarity': 0.88,
            'liveness_score': 0.82,
            'deepfake_score': 0.80
        }

        response = client.post(
            "/api/v1/authenticate",
            json={"capture_duration": 3}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["mfa_required"] is True
        assert data["user_id"] == "john_doe_001"

    def test_authenticate_frame_success(self, client, mock_components):
        """Test authentication via uploaded image frame."""
        mock_components['auth_engine'].authenticate_user.return_value = {
            'success': True,
            'message': 'Authentication successful',
            'user_id': 'frame_user_001',
            'name': 'Frame User',
            'confidence': 0.91,
            'face_similarity': 0.93,
            'liveness_score': 0.89,
            'deepfake_score': 0.90,
            'processing_time': 0.32
        }

        image_bytes = create_test_image_bytes(color='red')

        response = client.post(
            "/api/v1/authenticate-frame",
            files={
                "file": ("auth_face.jpg", image_bytes, "image/jpeg")
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["user_id"] == "frame_user_001"
        assert data["confidence"] == 0.91

        # Verify auth_engine was called with frame keyword arg
        mock_components['auth_engine'].authenticate_user.assert_called_once()
        call_kwargs = mock_components['auth_engine'].authenticate_user.call_args[1]
        assert 'frame' in call_kwargs

    def test_authenticate_frame_invalid_image(self, client):
        """Test authentication frame upload rejects invalid image bytes with HTTP 400."""
        response = client.post(
            "/api/v1/authenticate-frame",
            files={
                "file": ("auth_face.jpg", b"invalid-bytes", "image/jpeg")
            }
        )
        assert response.status_code == 400
        assert response.status_code != 200
        data = response.json()
        assert "detail" in data
        assert "Invalid image file" in data["detail"]


class TestMFAVerificationEndpoint:
    """Tests for multi-factor authentication verification."""

    def test_verify_mfa_success(self, client, mock_components):
        """Test successful MFA OTP verification."""
        mock_components['auth_engine'].verify_mfa.return_value = {
            'success': True,
            'message': 'Multi-factor authentication successful',
            'user_id': 'john_doe_001'
        }

        response = client.post(
            "/api/v1/verify-mfa",
            json={"user_id": "john_doe_001", "otp": "123456"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["user_id"] == "john_doe_001"

    def test_verify_mfa_failure(self, client, mock_components):
        """Test failed MFA verification returns HTTP 400."""
        mock_components['auth_engine'].verify_mfa.return_value = {
            'success': False,
            'message': 'Invalid or expired OTP'
        }

        response = client.post(
            "/api/v1/verify-mfa",
            json={"user_id": "john_doe_001", "otp": "000000"}
        )

        assert response.status_code == 400
        assert "Invalid or expired OTP" in response.json()["detail"]

    def test_verify_mfa_engine_not_initialized(self):
        """Test MFA verification returns 500 when engine is not initialized."""
        api = FaceRecognitionAPI({})
        client = ASGIClient(api.app)

        response = client.post(
            "/api/v1/verify-mfa",
            json={"user_id": "john_doe_001", "otp": "123456"}
        )
        assert response.status_code == 500


class TestUserManagementEndpoints:
    """Tests for user information retrieval and deletion endpoints."""

    def test_get_user_success(self, client, mock_components):
        """Test retrieving existing user information."""
        mock_components['database_manager'].get_user.return_value = {
            'user_id': 'john_doe_001',
            'name': 'John Doe',
            'email': 'john@example.com',
            'phone': '+1234567890',
            'created_at': '2026-09-01T12:00:00',
            'is_active': True
        }

        response = client.get("/api/v1/users/john_doe_001")
        assert response.status_code == 200

        data = response.json()
        assert data["user_id"] == "john_doe_001"
        assert data["name"] == "John Doe"
        assert data["email"] == "john@example.com"
        assert data["is_active"] is True

    def test_get_user_not_found(self, client, mock_components):
        """Test retrieving non-existent user returns 404 and preserves error detail."""
        mock_components['database_manager'].get_user.return_value = None

        response = client.get("/api/v1/users/nonexistent")
        assert response.status_code == 404
        data = response.json()
        assert data["error"] == "Not Found"
        assert data["message"] == "User not found"
        assert data["detail"] == "User not found"

    def test_delete_user_success(self, client, mock_components):
        """Test deleting existing user."""
        mock_components['database_manager'].delete_user.return_value = True

        response = client.delete("/api/v1/users/john_doe_001")
        assert response.status_code == 200
        assert "deleted successfully" in response.json()["message"]
        mock_components['database_manager'].delete_user.assert_called_once_with('john_doe_001')

    def test_delete_user_not_found(self, client, mock_components):
        """Test deleting non-existent user returns 404."""
        mock_components['database_manager'].delete_user.return_value = False

        response = client.delete("/api/v1/users/nonexistent")
        assert response.status_code == 404


class TestSystemEndpoints:
    """Tests for stats, image analysis, and benchmarking endpoints."""

    def test_get_system_stats(self, client, mock_components):
        """Test retrieving system and database statistics."""
        mock_components['database_manager'].get_statistics.return_value = {
            'total_users': 10,
            'active_users': 9,
            'total_embeddings': 25,
            'avg_embedding_quality': 0.89,
            'total_authentications': 100,
            'successful_authentications': 92,
            'success_rate_percent': 92.0,
            'faiss_index_size': 25
        }

        response = client.get("/api/v1/stats")
        assert response.status_code == 200

        data = response.json()
        assert data["total_users"] == 10
        assert data["active_users"] == 9
        assert data["total_embeddings"] == 25
        assert data["success_rate_percent"] == 92.0

    def test_upload_image_for_analysis(self, client, mock_components):
        """Test uploading image for liveness and deepfake analysis."""
        mock_components['liveness_detector'].comprehensive_liveness_check.return_value = {
            'is_live': True,
            'confidence': 0.88,
            'combined_score': 0.85
        }
        mock_components['deepfake_detector'].comprehensive_deepfake_analysis.return_value = {
            'is_deepfake': False,
            'confidence': 0.90,
            'combined_fake_score': 0.10
        }

        image_bytes = create_test_image_bytes(color='yellow')

        response = client.post(
            "/api/v1/upload-image",
            files={"file": ("check.jpg", image_bytes, "image/jpeg")}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["filename"] == "check.jpg"
        assert "liveness_analysis" in data
        assert data["liveness_analysis"]["is_live"] is True
        assert "deepfake_analysis" in data
        assert data["deepfake_analysis"]["is_deepfake"] is False

    def test_benchmark_system(self, client, mock_components):
        """Test system benchmarking endpoint."""
        mock_components['embedding_extractor'].benchmark_inference_time.return_value = {'mean_time_ms': 12.5}
        mock_components['liveness_detector'].benchmark_inference_time.return_value = {'mean_time_ms': 8.2}
        mock_components['deepfake_detector'].benchmark_inference_time.return_value = {'mean_time_ms': 25.1}

        response = client.get("/api/v1/system/benchmark")
        assert response.status_code == 200

        data = response.json()
        assert "benchmarks" in data
        assert "embedding_extraction" in data["benchmarks"]
        assert "liveness_detection" in data["benchmarks"]
        assert "deepfake_detection" in data["benchmarks"]

    def test_custom_404_handler(self, client):
        """Test global 404 handler returns standardized generic JSON for unmatched routes."""
        response = client.get("/api/v1/route-that-does-not-exist")
        assert response.status_code == 404
        data = response.json()
        assert data["error"] == "Not Found"
        assert data["message"] == "The requested resource was not found"
        assert data["detail"] == "The requested resource was not found"
        assert "path" in data
        assert data["path"] == "/api/v1/route-that-does-not-exist"
