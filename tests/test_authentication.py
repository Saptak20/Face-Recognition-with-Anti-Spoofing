"""
Test Authentication Engine

Unit tests for the authentication engine including registration,
authentication pipeline, and multi-factor authentication.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    from src.authentication import AuthenticationEngine, OTPManager
except ImportError:
    AuthenticationEngine = Mock
    OTPManager = Mock


class TestOTPManager:
    """Test class for OTP Manager functionality."""
    
    @pytest.fixture
    def otp_manager(self):
        """Create OTP Manager instance for testing."""
        return OTPManager(otp_length=6, otp_expiry_minutes=5)
    
    def test_initialization(self, otp_manager):
        """Test OTP Manager initialization."""
        assert otp_manager.otp_length == 6
        assert otp_manager.otp_expiry_minutes == 5
        assert isinstance(otp_manager.active_otps, dict)
    
    def test_generate_otp(self, otp_manager):
        """Test OTP generation."""
        user_id = "test_user_001"
        otp = otp_manager.generate_otp(user_id)
        
        assert len(otp) == 6
        assert otp.isdigit()
        assert user_id in otp_manager.active_otps
        
        otp_data = otp_manager.active_otps[user_id]
        assert otp_data['otp'] == otp
        assert otp_data['attempts'] == 0
        assert otp_data['expiry'] > datetime.now()
    
    def test_verify_otp_success(self, otp_manager):
        """Test successful OTP verification."""
        user_id = "test_user_001"
        otp = otp_manager.generate_otp(user_id)
        
        result = otp_manager.verify_otp(user_id, otp)
        
        assert result is True
        assert user_id not in otp_manager.active_otps  # Should be cleaned up
    
    def test_verify_otp_failure(self, otp_manager):
        """Test OTP verification failure."""
        user_id = "test_user_001"
        otp = otp_manager.generate_otp(user_id)
        
        result = otp_manager.verify_otp(user_id, "wrong_otp")
        
        assert result is False
        assert user_id in otp_manager.active_otps
        assert otp_manager.active_otps[user_id]['attempts'] == 1
    
    def test_verify_otp_no_active_otp(self, otp_manager):
        """Test OTP verification with no active OTP."""
        result = otp_manager.verify_otp("nonexistent_user", "123456")
        
        assert result is False
    
    def test_verify_otp_expired(self, otp_manager):
        """Test OTP verification with expired OTP."""
        user_id = "test_user_001"
        otp = otp_manager.generate_otp(user_id)
        
        # Manually expire the OTP
        past_time = datetime.now() - timedelta(minutes=10)
        otp_manager.active_otps[user_id]['expiry'] = past_time
        
        result = otp_manager.verify_otp(user_id, otp)
        
        assert result is False
        assert user_id not in otp_manager.active_otps  # Should be cleaned up
    
    def test_verify_otp_too_many_attempts(self, otp_manager):
        """Test OTP verification with too many attempts."""
        user_id = "test_user_001"
        otp = otp_manager.generate_otp(user_id)
        
        # Make 3 failed attempts
        for _ in range(3):
            otp_manager.verify_otp(user_id, "wrong_otp")
        
        # Fourth attempt should fail even with correct OTP
        result = otp_manager.verify_otp(user_id, otp)
        
        assert result is False
        assert user_id not in otp_manager.active_otps  # Should be cleaned up
    
    @patch('smtplib.SMTP')
    def test_send_otp_email(self, mock_smtp, otp_manager):
        """Test OTP email sending."""
        mock_server = Mock()
        mock_smtp.return_value = mock_server
        
        result = otp_manager.send_otp_email(
            user_email="test@example.com",
            otp="123456",
            sender_email="sender@example.com",
            sender_password="password",
            user_name="Test User"
        )
        
        assert result is True
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once_with("sender@example.com", "password")
        mock_server.send_message.assert_called_once()
        mock_server.quit.assert_called_once()
    
    def test_cleanup_expired_otps(self, otp_manager):
        """Test cleanup of expired OTPs."""
        # Generate some OTPs
        user1 = "user1"
        user2 = "user2"
        
        otp_manager.generate_otp(user1)
        otp_manager.generate_otp(user2)
        
        # Manually expire one OTP
        past_time = datetime.now() - timedelta(minutes=10)
        otp_manager.active_otps[user1]['expiry'] = past_time
        
        # Cleanup expired OTPs
        cleaned_count = otp_manager.cleanup_expired_otps()
        
        assert cleaned_count == 1
        assert user1 not in otp_manager.active_otps
        assert user2 in otp_manager.active_otps


class TestAuthenticationEngine:
    """Test class for Authentication Engine functionality."""
    
    @pytest.fixture
    def mock_components(self):
        """Create mock components for testing."""
        return {
            'face_capture': Mock(),
            'embedding_extractor': Mock(),
            'liveness_detector': Mock(),
            'deepfake_detector': Mock(),
            'database_manager': Mock()
        }
    
    @pytest.fixture
    def auth_config(self):
        """Create authentication configuration for testing."""
        return {
            'face_similarity_threshold': 0.7,
            'liveness_threshold': 0.5,
            'deepfake_threshold': 0.5,
            'overall_confidence_threshold': 0.6,
            'enable_mfa': False,
            'max_attempts_per_hour': 5,
            'confidence_weights': {
                'face_similarity': 0.5,
                'liveness': 0.3,
                'deepfake': 0.2
            }
        }
    
    @pytest.fixture
    def auth_engine(self, mock_components, auth_config):
        """Create Authentication Engine instance for testing."""
        return AuthenticationEngine(
            face_capture=mock_components['face_capture'],
            embedding_extractor=mock_components['embedding_extractor'],
            liveness_detector=mock_components['liveness_detector'],
            deepfake_detector=mock_components['deepfake_detector'],
            database_manager=mock_components['database_manager'],
            config=auth_config
        )
    
    def test_initialization(self, auth_engine, auth_config):
        """Test Authentication Engine initialization."""
        assert auth_engine.face_similarity_threshold == 0.7
        assert auth_engine.liveness_threshold == 0.5
        assert auth_engine.deepfake_threshold == 0.5
        assert auth_engine.overall_confidence_threshold == 0.6
        assert auth_engine.enable_mfa is False
    
    def test_register_user_success(self, auth_engine, mock_components):
        """Test successful user registration."""
        # Mock database operations
        mock_components['database_manager'].add_user.return_value = True
        mock_components['database_manager'].add_embedding.return_value = "embedding_123"
        
        # Mock face capture
        test_face = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
        mock_components['face_capture'].capture_from_webcam.return_value = [test_face]
        mock_components['face_capture'].validate_face_quality.return_value = {'overall_quality': 0.9}
        
        # Mock embedding extraction
        test_embedding = np.random.randn(512).astype(np.float32)
        mock_components['embedding_extractor'].extract_embedding.return_value = test_embedding
        
        result = auth_engine.register_user(
            user_id="test_user_001",
            name="Test User",
            email="test@example.com"
        )
        
        assert result['success'] is True
        assert result['user_id'] == "test_user_001"
        assert 'embedding_id' in result
        mock_components['database_manager'].add_user.assert_called_once()
        mock_components['database_manager'].add_embedding.assert_called_once()
    
    def test_register_user_existing_user(self, auth_engine, mock_components):
        """Test user registration with existing user."""
        mock_components['database_manager'].add_user.return_value = False
        
        result = auth_engine.register_user(
            user_id="existing_user",
            name="Existing User"
        )
        
        assert result['success'] is False
        assert "already exists" in result['message']
    
    def test_register_user_no_faces_captured(self, auth_engine, mock_components):
        """Test user registration with no faces captured."""
        mock_components['database_manager'].add_user.return_value = True
        mock_components['face_capture'].capture_from_webcam.return_value = []
        
        result = auth_engine.register_user(
            user_id="test_user_001",
            name="Test User"
        )
        
        assert result['success'] is False
        assert "No faces captured" in result['message']
    
    def test_authenticate_user_success(self, auth_engine, mock_components):
        """Test successful user authentication."""
        # Mock face capture
        test_face = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
        mock_components['face_capture'].capture_from_webcam.return_value = [test_face]
        mock_components['face_capture'].validate_face_quality.return_value = {'overall_quality': 0.8}
        
        # Mock liveness detection
        mock_components['liveness_detector'].comprehensive_liveness_check.return_value = {
            'combined_score': 0.8
        }
        
        # Mock deepfake detection
        mock_components['deepfake_detector'].comprehensive_deepfake_analysis.return_value = {
            'combined_fake_score': 0.2  # Low fake score = high real score
        }
        
        # Mock embedding extraction and authentication
        test_embedding = np.random.randn(512).astype(np.float32)
        mock_components['embedding_extractor'].extract_embedding.return_value = test_embedding
        mock_components['database_manager'].authenticate_user.return_value = {
            'user_id': 'test_user_001',
            'name': 'Test User',
            'similarity': 0.85,
            'is_active': True
        }
        
        result = auth_engine.authenticate_user()
        
        assert result['success'] is True
        assert result['user_id'] == 'test_user_001'
        assert result['confidence'] > 0.6
    
    def test_authenticate_user_no_face(self, auth_engine, mock_components):
        """Test authentication with no face detected."""
        mock_components['face_capture'].capture_from_webcam.return_value = []
        
        result = auth_engine.authenticate_user()
        
        assert result['success'] is False
        assert "No face detected" in result['message']
    
    def test_authenticate_user_liveness_failed(self, auth_engine, mock_components):
        """Test authentication with liveness detection failure."""
        test_face = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
        mock_components['face_capture'].capture_from_webcam.return_value = [test_face]
        mock_components['face_capture'].validate_face_quality.return_value = {'overall_quality': 0.8}
        
        # Low liveness score
        mock_components['liveness_detector'].comprehensive_liveness_check.return_value = {
            'combined_score': 0.3
        }
        
        result = auth_engine.authenticate_user()
        
        assert result['success'] is False
        assert "Liveness check failed" in result['message']
    
    def test_authenticate_user_deepfake_detected(self, auth_engine, mock_components):
        """Test authentication with deepfake detection."""
        test_face = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
        mock_components['face_capture'].capture_from_webcam.return_value = [test_face]
        mock_components['face_capture'].validate_face_quality.return_value = {'overall_quality': 0.8}
        
        mock_components['liveness_detector'].comprehensive_liveness_check.return_value = {
            'combined_score': 0.8
        }
        
        # High fake score
        mock_components['deepfake_detector'].comprehensive_deepfake_analysis.return_value = {
            'combined_fake_score': 0.8
        }
        
        result = auth_engine.authenticate_user()
        
        assert result['success'] is False
        assert "Deepfake detected" in result['message']
    
    def test_authenticate_user_no_match(self, auth_engine, mock_components):
        """Test authentication with no matching user."""
        test_face = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
        mock_components['face_capture'].capture_from_webcam.return_value = [test_face]
        mock_components['face_capture'].validate_face_quality.return_value = {'overall_quality': 0.8}
        
        mock_components['liveness_detector'].comprehensive_liveness_check.return_value = {
            'combined_score': 0.8
        }
        mock_components['deepfake_detector'].comprehensive_deepfake_analysis.return_value = {
            'combined_fake_score': 0.2
        }
        
        test_embedding = np.random.randn(512).astype(np.float32)
        mock_components['embedding_extractor'].extract_embedding.return_value = test_embedding
        mock_components['database_manager'].authenticate_user.return_value = None
        
        result = auth_engine.authenticate_user()
        
        assert result['success'] is False
        assert "No matching user found" in result['message']
    
    def test_authenticate_user_with_mfa(self, auth_engine, mock_components, auth_config):
        """Test authentication requiring MFA."""
        # Enable MFA
        auth_config['enable_mfa'] = True
        auth_engine.enable_mfa = True
        auth_engine.otp_manager = Mock()
        auth_engine.otp_manager.generate_otp.return_value = "123456"
        
        # Setup successful authentication up to MFA
        test_face = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
        mock_components['face_capture'].capture_from_webcam.return_value = [test_face]
        mock_components['face_capture'].validate_face_quality.return_value = {'overall_quality': 0.8}
        
        mock_components['liveness_detector'].comprehensive_liveness_check.return_value = {
            'combined_score': 0.8
        }
        mock_components['deepfake_detector'].comprehensive_deepfake_analysis.return_value = {
            'combined_fake_score': 0.2
        }
        
        test_embedding = np.random.randn(512).astype(np.float32)
        mock_components['embedding_extractor'].extract_embedding.return_value = test_embedding
        mock_components['database_manager'].authenticate_user.return_value = {
            'user_id': 'test_user_001',
            'name': 'Test User',
            'email': 'test@example.com',
            'similarity': 0.85,
            'is_active': True
        }
        
        result = auth_engine.authenticate_user()
        
        assert result['success'] is False
        assert result['mfa_required'] is True
        assert result['user_id'] == 'test_user_001'
    
    def test_verify_mfa_success(self, auth_engine):
        """Test successful MFA verification."""
        auth_engine.enable_mfa = True
        auth_engine.otp_manager = Mock()
        auth_engine.otp_manager.verify_otp.return_value = True
        
        result = auth_engine.verify_mfa("test_user_001", "123456")
        
        assert result['success'] is True
        assert result['user_id'] == "test_user_001"
    
    def test_verify_mfa_failure(self, auth_engine):
        """Test MFA verification failure."""
        auth_engine.enable_mfa = True
        auth_engine.otp_manager = Mock()
        auth_engine.otp_manager.verify_otp.return_value = False
        
        result = auth_engine.verify_mfa("test_user_001", "wrong_otp")
        
        assert result['success'] is False
        assert "Invalid or expired OTP" in result['message']
    
    def test_verify_mfa_disabled(self, auth_engine):
        """Test MFA verification when MFA is disabled."""
        result = auth_engine.verify_mfa("test_user_001", "123456")
        
        assert result['success'] is False
        assert "not enabled" in result['message']
    
    def test_calculate_overall_confidence(self, auth_engine):
        """Test overall confidence calculation."""
        confidence = auth_engine._calculate_overall_confidence(
            face_similarity=0.8,
            liveness_score=0.7,
            deepfake_score=0.9
        )
        
        # Should be weighted average: 0.8*0.5 + 0.7*0.3 + 0.9*0.2 = 0.79
        expected = 0.8 * 0.5 + 0.7 * 0.3 + 0.9 * 0.2
        assert abs(confidence - expected) < 0.01
    
    def test_rate_limiting(self, auth_engine):
        """Test rate limiting functionality."""
        ip_address = "192.168.1.100"
        
        # First few attempts should pass
        for _ in range(5):
            assert auth_engine._check_rate_limit(ip_address) is True
        
        # Sixth attempt should fail
        assert auth_engine._check_rate_limit(ip_address) is False
    
    def test_authentication_logging(self, auth_engine, mock_components):
        """Test authentication attempt logging."""
        auth_engine._log_authentication(
            user_id="test_user_001",
            success=True,
            confidence=0.85,
            liveness_score=0.8,
            deepfake_score=0.9,
            ip_address="192.168.1.100"
        )
        
        mock_components['database_manager'].log_authentication.assert_called_once()


@pytest.mark.asyncio
class TestAuthenticationAsync:
    """Test async functionality of Authentication Engine."""
    
    @pytest.fixture
    def auth_engine(self):
        """Create mock authentication engine for async testing."""
        mock_components = {
            'face_capture': Mock(),
            'embedding_extractor': Mock(),
            'liveness_detector': Mock(),
            'deepfake_detector': Mock(),
            'database_manager': Mock()
        }
        
        config = {
            'face_similarity_threshold': 0.7,
            'liveness_threshold': 0.5,
            'deepfake_threshold': 0.5,
            'overall_confidence_threshold': 0.6,
            'enable_mfa': False,
            'max_attempts_per_hour': 5,
            'confidence_weights': {
                'face_similarity': 0.5,
                'liveness': 0.3,
                'deepfake': 0.2
            }
        }
        
        return AuthenticationEngine(**mock_components, config=config)


if __name__ == "__main__":
    # Simple test runner
    pytest.main([__file__, "-v"])