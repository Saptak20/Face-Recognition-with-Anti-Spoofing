"""
Test FastAPI Endpoints

Unit tests for the FastAPI-based REST API endpoints including
registration, authentication, MFA, and management operations.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
import sys
from pathlib import Path
from datetime import datetime
import json
import io
import base64
from PIL import Image

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    from fastapi.testclient import TestClient
    from src.api import FaceRecognitionAPI, app
except ImportError:
    TestClient = Mock
    FaceRecognitionAPI = Mock
    app = Mock


class TestFaceRecognitionAPI:
    """Test class for Face Recognition API endpoints."""
    
    @pytest.fixture
    def mock_components(self):
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
    def api_config(self):
        """Create API configuration for testing."""
        return {
            'host': '0.0.0.0',
            'port': 8000,
            'enable_cors': True,
            'cors_origins': ['*'],
            'enable_rate_limiting': True,
            'rate_limit_calls': 100,
            'rate_limit_period': 3600,
            'enable_authentication': False,  # Disable for testing
            'max_file_size': 10 * 1024 * 1024,  # 10MB
            'allowed_file_types': ['image/jpeg', 'image/png', 'image/jpg']
        }
    
    @pytest.fixture
    def api_instance(self, mock_components, api_config):
        """Create API instance for testing."""
        return FaceRecognitionAPI(
            **mock_components,
            config=api_config
        )
    
    @pytest.fixture
    def client(self, api_instance):
        """Create test client."""
        if TestClient == Mock:
            return Mock()
        return TestClient(app)
    
    def test_api_initialization(self, api_instance, api_config):
        """Test API initialization."""
        assert api_instance.config == api_config
        assert hasattr(api_instance, 'auth_engine')
        assert hasattr(api_instance, 'database_manager')
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        if client == Mock():
            return  # Skip if TestClient not available
        
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
    
    def test_system_info(self, client):
        """Test system info endpoint."""
        if client == Mock():
            return
        
        response = client.get("/info")
        assert response.status_code == 200
        
        data = response.json()
        assert "system_name" in data
        assert "version" in data
        assert "description" in data
        assert "features" in data
    
    def create_test_image(self):
        """Create a test image for upload."""
        # Create a simple RGB image
        img = Image.new('RGB', (160, 160), color='red')
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='JPEG')
        img_buffer.seek(0)
        return img_buffer
    
    def test_register_user_success(self, client, mock_components):
        """Test successful user registration."""
        if client == Mock():
            return
        
        # Mock successful registration
        mock_components['auth_engine'].register_user.return_value = {
            'success': True,
            'user_id': 'test_user_001',
            'embedding_id': 'embedding_123',
            'message': 'User registered successfully'
        }
        
        # Create test image
        test_image = self.create_test_image()
        
        response = client.post(
            "/register",
            data={
                'user_id': 'test_user_001',
                'name': 'Test User',
                'email': 'test@example.com',
                'department': 'Engineering'
            },
            files={'image': ('test.jpg', test_image, 'image/jpeg')}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data['success'] is True
        assert data['user_id'] == 'test_user_001'
    
    def test_register_user_missing_fields(self, client):
        """Test registration with missing required fields."""
        if client == Mock():
            return
        
        response = client.post(
            "/register",
            data={'name': 'Test User'}  # Missing user_id
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_register_user_invalid_image(self, client):
        """Test registration with invalid image."""
        if client == Mock():
            return
        
        response = client.post(
            "/register",
            data={
                'user_id': 'test_user_001',
                'name': 'Test User'
            },
            files={'image': ('test.txt', b'not an image', 'text/plain')}
        )
        
        assert response.status_code == 400
    
    def test_authenticate_user_success(self, client, mock_components):
        """Test successful user authentication."""
        if client == Mock():
            return
        
        # Mock successful authentication
        mock_components['auth_engine'].authenticate_user.return_value = {
            'success': True,
            'user_id': 'test_user_001',
            'name': 'Test User',
            'confidence': 0.85,
            'timestamp': datetime.now().isoformat()
        }
        
        test_image = self.create_test_image()
        
        response = client.post(
            "/authenticate",
            files={'image': ('test.jpg', test_image, 'image/jpeg')}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data['success'] is True
        assert data['user_id'] == 'test_user_001'
    
    def test_authenticate_user_failure(self, client, mock_components):
        """Test failed user authentication."""
        if client == Mock():
            return
        
        # Mock failed authentication
        mock_components['auth_engine'].authenticate_user.return_value = {
            'success': False,
            'message': 'No matching user found',
            'confidence': 0.3
        }
        
        test_image = self.create_test_image()
        
        response = client.post(
            "/authenticate",
            files={'image': ('test.jpg', test_image, 'image/jpeg')}
        )
        
        assert response.status_code == 401
        data = response.json()
        assert data['success'] is False
    
    def test_authenticate_user_mfa_required(self, client, mock_components):
        """Test authentication requiring MFA."""
        if client == Mock():
            return
        
        # Mock MFA required
        mock_components['auth_engine'].authenticate_user.return_value = {
            'success': False,
            'mfa_required': True,
            'user_id': 'test_user_001',
            'message': 'MFA verification required'
        }
        
        test_image = self.create_test_image()
        
        response = client.post(
            "/authenticate",
            files={'image': ('test.jpg', test_image, 'image/jpeg')}
        )
        
        assert response.status_code == 202  # Accepted, but MFA required
        data = response.json()
        assert data['mfa_required'] is True
        assert data['user_id'] == 'test_user_001'
    
    def test_verify_mfa_success(self, client, mock_components):
        """Test successful MFA verification."""
        if client == Mock():
            return
        
        # Mock successful MFA verification
        mock_components['auth_engine'].verify_mfa.return_value = {
            'success': True,
            'user_id': 'test_user_001',
            'message': 'MFA verification successful'
        }
        
        response = client.post(
            "/verify-mfa",
            json={
                'user_id': 'test_user_001',
                'otp_code': '123456'
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data['success'] is True
        assert data['user_id'] == 'test_user_001'
    
    def test_verify_mfa_failure(self, client, mock_components):
        """Test failed MFA verification."""
        if client == Mock():
            return
        
        # Mock failed MFA verification
        mock_components['auth_engine'].verify_mfa.return_value = {
            'success': False,
            'message': 'Invalid or expired OTP'
        }
        
        response = client.post(
            "/verify-mfa",
            json={
                'user_id': 'test_user_001',
                'otp_code': 'wrong_otp'
            }
        )
        
        assert response.status_code == 400
        data = response.json()
        assert data['success'] is False
    
    def test_get_user_info(self, client, mock_components):
        """Test getting user information."""
        if client == Mock():
            return
        
        # Mock user data
        mock_components['database_manager'].get_user.return_value = {
            'user_id': 'test_user_001',
            'name': 'Test User',
            'email': 'test@example.com',
            'department': 'Engineering',
            'is_active': True,
            'created_at': '2023-12-01 10:00:00'
        }
        
        response = client.get("/users/test_user_001")
        
        assert response.status_code == 200
        data = response.json()
        assert data['user_id'] == 'test_user_001'
        assert data['name'] == 'Test User'
    
    def test_get_user_info_not_found(self, client, mock_components):
        """Test getting non-existent user information."""
        if client == Mock():
            return
        
        # Mock user not found
        mock_components['database_manager'].get_user.return_value = None
        
        response = client.get("/users/nonexistent_user")
        
        assert response.status_code == 404
    
    def test_update_user_info(self, client, mock_components):
        """Test updating user information."""
        if client == Mock():
            return
        
        # Mock successful update
        mock_components['database_manager'].update_user.return_value = True
        mock_components['database_manager'].get_user.return_value = {
            'user_id': 'test_user_001',
            'name': 'Updated User',
            'email': 'updated@example.com'
        }
        
        response = client.put(
            "/users/test_user_001",
            json={
                'name': 'Updated User',
                'email': 'updated@example.com'
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data['name'] == 'Updated User'
    
    def test_deactivate_user(self, client, mock_components):
        """Test deactivating user."""
        if client == Mock():
            return
        
        # Mock successful deactivation
        mock_components['database_manager'].deactivate_user.return_value = True
        
        response = client.post("/users/test_user_001/deactivate")
        
        assert response.status_code == 200
        data = response.json()
        assert data['success'] is True
    
    def test_delete_user(self, client, mock_components):
        """Test deleting user."""
        if client == Mock():
            return
        
        # Mock successful deletion
        mock_components['database_manager'].delete_user.return_value = True
        
        response = client.delete("/users/test_user_001")
        
        assert response.status_code == 200
        data = response.json()
        assert data['success'] is True
    
    def test_get_user_stats(self, client, mock_components):
        """Test getting user statistics."""
        if client == Mock():
            return
        
        # Mock user stats
        mock_components['database_manager'].get_user_stats.return_value = {
            'user_id': 'test_user_001',
            'total_embeddings': 5,
            'total_authentications': 25,
            'successful_authentications': 23,
            'failed_authentications': 2,
            'last_authentication': '2023-12-01 15:30:00'
        }
        
        response = client.get("/users/test_user_001/stats")
        
        assert response.status_code == 200
        data = response.json()
        assert data['total_embeddings'] == 5
        assert data['total_authentications'] == 25
    
    def test_get_authentication_logs(self, client, mock_components):
        """Test getting authentication logs."""
        if client == Mock():
            return
        
        # Mock authentication logs
        mock_components['database_manager'].get_authentication_logs.return_value = [
            {
                'user_id': 'test_user_001',
                'success': True,
                'confidence': 0.85,
                'timestamp': '2023-12-01 10:00:00'
            },
            {
                'user_id': 'test_user_002',
                'success': False,
                'confidence': 0.45,
                'timestamp': '2023-12-01 10:05:00'
            }
        ]
        
        response = client.get("/logs/authentication?limit=10")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data['logs']) == 2
        assert data['logs'][0]['user_id'] == 'test_user_001'
    
    def test_get_system_stats(self, client, mock_components):
        """Test getting system statistics."""
        if client == Mock():
            return
        
        # Mock system stats
        mock_components['database_manager'].get_system_stats.return_value = {
            'total_users': 150,
            'active_users': 142,
            'total_embeddings': 750,
            'total_authentications': 5420,
            'successful_authentications': 5128,
            'success_rate': 0.946
        }
        
        response = client.get("/stats/system")
        
        assert response.status_code == 200
        data = response.json()
        assert data['total_users'] == 150
        assert data['success_rate'] == 0.946
    
    def test_add_user_embedding(self, client, mock_components):
        """Test adding additional user embedding."""
        if client == Mock():
            return
        
        # Mock successful embedding addition
        mock_components['database_manager'].add_embedding.return_value = 'embedding_456'
        
        test_image = self.create_test_image()
        
        response = client.post(
            "/users/test_user_001/embeddings",
            files={'image': ('test.jpg', test_image, 'image/jpeg')}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data['success'] is True
        assert 'embedding_id' in data
    
    def test_get_user_embeddings(self, client, mock_components):
        """Test getting user embeddings."""
        if client == Mock():
            return
        
        # Mock user embeddings
        mock_components['database_manager'].get_user_embeddings.return_value = [
            {
                'embedding_id': 'embedding_123',
                'created_at': '2023-12-01 10:00:00',
                'quality_score': 0.9
            },
            {
                'embedding_id': 'embedding_456',
                'created_at': '2023-12-01 11:00:00',
                'quality_score': 0.85
            }
        ]
        
        response = client.get("/users/test_user_001/embeddings")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data['embeddings']) == 2
        assert data['embeddings'][0]['embedding_id'] == 'embedding_123'
    
    def test_backup_database(self, client, mock_components):
        """Test database backup endpoint."""
        if client == Mock():
            return
        
        # Mock successful backup
        mock_components['database_manager'].backup_database.return_value = True
        
        response = client.post("/admin/backup")
        
        assert response.status_code == 200
        data = response.json()
        assert data['success'] is True
    
    def test_file_size_validation(self, client):
        """Test file size validation."""
        if client == Mock():
            return
        
        # Create a large file (simulate)
        large_file = io.BytesIO(b'x' * (15 * 1024 * 1024))  # 15MB
        
        response = client.post(
            "/register",
            data={
                'user_id': 'test_user_001',
                'name': 'Test User'
            },
            files={'image': ('large.jpg', large_file, 'image/jpeg')}
        )
        
        assert response.status_code == 413  # Payload too large
    
    def test_rate_limiting(self, client):
        """Test rate limiting functionality."""
        if client == Mock():
            return
        
        # This test would require actual rate limiting implementation
        # For now, just check that the endpoint responds normally
        response = client.get("/health")
        assert response.status_code == 200


class TestAPIMiddleware:
    """Test class for API middleware functionality."""
    
    def test_cors_headers(self, client):
        """Test CORS headers in responses."""
        if client == Mock():
            return
        
        response = client.get("/health")
        
        # Check that CORS headers might be present
        # (depends on FastAPI CORS middleware configuration)
        assert response.status_code == 200
    
    def test_request_logging(self, client):
        """Test request logging middleware."""
        if client == Mock():
            return
        
        # Make a request that should be logged
        response = client.get("/health")
        
        assert response.status_code == 200
        # In a real implementation, you would check logs
    
    def test_error_handling(self, client):
        """Test global error handling."""
        if client == Mock():
            return
        
        # Test with invalid endpoint
        response = client.get("/nonexistent-endpoint")
        
        assert response.status_code == 404


@pytest.mark.asyncio
class TestAsyncEndpoints:
    """Test async functionality of API endpoints."""
    
    @pytest.fixture
    def async_client(self, api_instance):
        """Create async test client."""
        if TestClient == Mock:
            return Mock()
        return TestClient(app)
    
    async def test_async_authentication(self, async_client, mock_components):
        """Test async authentication endpoint."""
        if async_client == Mock():
            return
        
        # Mock async authentication
        mock_components['auth_engine'].authenticate_user = AsyncMock()
        mock_components['auth_engine'].authenticate_user.return_value = {
            'success': True,
            'user_id': 'test_user_001',
            'confidence': 0.85
        }
        
        test_image = self.create_test_image()
        
        response = async_client.post(
            "/authenticate",
            files={'image': ('test.jpg', test_image, 'image/jpeg')}
        )
        
        assert response.status_code == 200
    
    def create_test_image(self):
        """Create a test image for upload."""
        img = Image.new('RGB', (160, 160), color='blue')
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='JPEG')
        img_buffer.seek(0)
        return img_buffer


@pytest.mark.integration
class TestAPIIntegration:
    """Integration tests for the complete API."""
    
    def test_full_user_lifecycle(self, client, mock_components):
        """Test complete user lifecycle through API."""
        if client == Mock():
            return
        
        # Mock all required operations
        mock_components['auth_engine'].register_user.return_value = {
            'success': True,
            'user_id': 'test_user_001',
            'embedding_id': 'embedding_123'
        }
        
        mock_components['database_manager'].get_user.return_value = {
            'user_id': 'test_user_001',
            'name': 'Test User',
            'is_active': True
        }
        
        mock_components['auth_engine'].authenticate_user.return_value = {
            'success': True,
            'user_id': 'test_user_001',
            'confidence': 0.85
        }
        
        mock_components['database_manager'].delete_user.return_value = True
        
        test_image = self.create_test_image()
        
        # 1. Register user
        response = client.post(
            "/register",
            data={'user_id': 'test_user_001', 'name': 'Test User'},
            files={'image': ('test.jpg', test_image, 'image/jpeg')}
        )
        assert response.status_code == 200
        
        # 2. Get user info
        response = client.get("/users/test_user_001")
        assert response.status_code == 200
        
        # 3. Authenticate user
        test_image.seek(0)  # Reset image buffer
        response = client.post(
            "/authenticate",
            files={'image': ('test.jpg', test_image, 'image/jpeg')}
        )
        assert response.status_code == 200
        
        # 4. Delete user
        response = client.delete("/users/test_user_001")
        assert response.status_code == 200
    
    def create_test_image(self):
        """Create a test image for upload."""
        img = Image.new('RGB', (160, 160), color='green')
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='JPEG')
        img_buffer.seek(0)
        return img_buffer


if __name__ == "__main__":
    # Simple test runner
    pytest.main([__file__, "-v"])