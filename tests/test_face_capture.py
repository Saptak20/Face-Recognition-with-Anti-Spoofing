"""
Test Face Capture Module

Unit tests for face capture functionality including face detection,
image processing, and quality validation.
"""

import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    from src.face_capture import FaceCapture
except ImportError:
    # Mock the imports if dependencies are not available
    FaceCapture = Mock


class TestFaceCapture:
    """Test class for FaceCapture functionality."""
    
    @pytest.fixture
    def face_capture(self):
        """Create FaceCapture instance for testing."""
        return FaceCapture(device='cpu', image_size=160)
    
    @pytest.fixture
    def sample_image(self):
        """Create sample image for testing."""
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    def test_initialization(self, face_capture):
        """Test FaceCapture initialization."""
        assert face_capture.device.type == 'cpu'
        assert face_capture.image_size == 160
        assert face_capture.margin == 32
    
    def test_detect_faces(self, face_capture, sample_image):
        """Test face detection functionality."""
        # Mock MTCNN detection
        with patch.object(face_capture.mtcnn, 'detect') as mock_detect:
            mock_detect.return_value = (
                np.array([[100, 100, 200, 200]]),  # boxes
                np.array([0.95]),  # probabilities
                np.array([[[120, 120], [140, 120], [130, 140], [120, 160], [140, 160]]])  # landmarks
            )
            
            faces = face_capture.detect_faces(sample_image)
            
            assert len(faces) == 1
            assert faces[0]['confidence'] == 0.95
            assert faces[0]['face_id'] == 0
            assert 'bbox' in faces[0]
            assert 'landmarks' in faces[0]
    
    def test_detect_faces_no_detection(self, face_capture, sample_image):
        """Test face detection with no faces found."""
        with patch.object(face_capture.mtcnn, 'detect') as mock_detect:
            mock_detect.return_value = (None, None, None)
            
            faces = face_capture.detect_faces(sample_image)
            
            assert len(faces) == 0
    
    def test_extract_face(self, face_capture, sample_image):
        """Test face extraction from image."""
        bbox = np.array([100, 100, 200, 200])
        
        extracted_face = face_capture.extract_face(sample_image, bbox)
        
        assert extracted_face is not None
        assert extracted_face.shape == (160, 160, 3)
    
    def test_extract_face_invalid_bbox(self, face_capture, sample_image):
        """Test face extraction with invalid bbox."""
        bbox = np.array([-100, -100, -50, -50])  # Invalid bbox
        
        extracted_face = face_capture.extract_face(sample_image, bbox)
        
        # Should still work due to bounds checking
        assert extracted_face is not None
    
    def test_preprocess_face(self, face_capture):
        """Test face preprocessing."""
        face = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
        
        preprocessed = face_capture.preprocess_face(face)
        
        assert preprocessed is not None
        assert preprocessed.shape == (3, 160, 160)
        assert preprocessed.dtype == np.float64
        assert -1.0 <= preprocessed.min() <= preprocessed.max() <= 1.0
    
    def test_validate_face_quality(self, face_capture):
        """Test face quality validation."""
        # Create a high-quality test face
        face = np.random.randint(50, 200, (160, 160, 3), dtype=np.uint8)
        
        quality_metrics = face_capture.validate_face_quality(face)
        
        assert 'blur_score' in quality_metrics
        assert 'brightness' in quality_metrics
        assert 'contrast' in quality_metrics
        assert 'size_score' in quality_metrics
        assert 'overall_quality' in quality_metrics
        
        assert 0.0 <= quality_metrics['overall_quality'] <= 1.0
    
    def test_validate_face_quality_low_quality(self, face_capture):
        """Test face quality validation with low quality image."""
        # Create a low-quality test face (very dark)
        face = np.ones((160, 160, 3), dtype=np.uint8) * 10
        
        quality_metrics = face_capture.validate_face_quality(face)
        
        # Should detect low quality
        assert quality_metrics['overall_quality'] < 0.5
    
    @patch('cv2.VideoCapture')
    def test_capture_from_webcam(self, mock_video_capture, face_capture):
        """Test webcam capture functionality."""
        # Mock video capture
        mock_cap = Mock()
        mock_video_capture.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        mock_cap.read.side_effect = [
            (True, np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
            for _ in range(10)
        ] + [(False, None)]  # End capture
        
        # Mock face detection
        with patch.object(face_capture, 'detect_faces') as mock_detect:
            mock_detect.return_value = [{
                'bbox': np.array([100, 100, 200, 200]),
                'confidence': 0.95,
                'face_id': 0
            }]
            
            with patch.object(face_capture, 'extract_face') as mock_extract:
                mock_extract.return_value = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
                
                with patch('cv2.imshow'), patch('cv2.waitKey', return_value=ord('q')):
                    faces = face_capture.capture_from_webcam(duration=1)
                    
                    assert len(faces) > 0
    
    @patch('cv2.VideoCapture')
    def test_capture_from_webcam_no_camera(self, mock_video_capture, face_capture):
        """Test webcam capture with no camera available."""
        mock_cap = Mock()
        mock_video_capture.return_value = mock_cap
        mock_cap.isOpened.return_value = False
        
        faces = face_capture.capture_from_webcam(duration=1)
        
        assert len(faces) == 0
    
    @patch('cv2.imread')
    def test_capture_from_image(self, mock_imread, face_capture):
        """Test face capture from static image."""
        # Mock image loading
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        mock_imread.return_value = test_image
        
        # Mock face detection
        with patch.object(face_capture, 'detect_faces') as mock_detect:
            mock_detect.return_value = [{
                'bbox': np.array([100, 100, 200, 200]),
                'confidence': 0.95,
                'face_id': 0
            }]
            
            with patch.object(face_capture, 'extract_face') as mock_extract:
                mock_extract.return_value = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
                
                faces = face_capture.capture_from_image("test_image.jpg")
                
                assert len(faces) == 1
    
    @patch('cv2.imread')
    def test_capture_from_image_invalid_path(self, mock_imread, face_capture):
        """Test face capture from invalid image path."""
        mock_imread.return_value = None
        
        faces = face_capture.capture_from_image("invalid_path.jpg")
        
        assert len(faces) == 0
    
    def test_calculate_overall_quality(self, face_capture):
        """Test overall quality calculation."""
        # Test with good quality parameters
        quality = face_capture._calculate_overall_quality(
            blur=150.0, brightness=127.5, contrast=40.0, size=160.0
        )
        
        assert 0.0 <= quality <= 1.0
        assert quality > 0.5  # Should be decent quality
        
        # Test with poor quality parameters
        poor_quality = face_capture._calculate_overall_quality(
            blur=10.0, brightness=10.0, contrast=5.0, size=50.0
        )
        
        assert 0.0 <= poor_quality <= 1.0
        assert poor_quality < quality


class TestFaceCaptureIntegration:
    """Integration tests for FaceCapture."""
    
    @pytest.fixture
    def face_capture(self):
        """Create FaceCapture instance for integration testing."""
        return FaceCapture(device='cpu', image_size=160)
    
    def test_full_pipeline_simulation(self, face_capture):
        """Test full face capture pipeline with mocked components."""
        # Create realistic test image
        test_image = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        
        # Mock successful face detection
        with patch.object(face_capture.mtcnn, 'detect') as mock_detect:
            mock_detect.return_value = (
                np.array([[150, 150, 350, 350]]),
                np.array([0.95]),
                np.array([[[200, 180], [300, 180], [250, 220], [200, 260], [300, 260]]])
            )
            
            # Test detection
            faces = face_capture.detect_faces(test_image)
            assert len(faces) == 1
            
            # Test extraction
            face_img = face_capture.extract_face(test_image, faces[0]['bbox'])
            assert face_img is not None
            
            # Test preprocessing
            preprocessed = face_capture.preprocess_face(face_img)
            assert preprocessed is not None
            
            # Test quality validation
            quality = face_capture.validate_face_quality(face_img)
            assert 'overall_quality' in quality


@pytest.mark.asyncio
class TestFaceCaptureAsync:
    """Test async functionality of FaceCapture."""
    
    @pytest.fixture
    def face_capture(self):
        """Create FaceCapture instance for async testing."""
        return FaceCapture(device='cpu', image_size=160)
    
    async def test_capture_from_webcam_async(self, face_capture):
        """Test async webcam capture."""
        with patch.object(face_capture, 'capture_from_webcam') as mock_capture:
            mock_capture.return_value = [
                np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
            ]
            
            faces = await face_capture.capture_from_webcam_async(duration=1)
            
            assert len(faces) == 1
            mock_capture.assert_called_once_with(1)


if __name__ == "__main__":
    # Simple test runner
    pytest.main([__file__, "-v"])