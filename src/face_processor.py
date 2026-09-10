"""
Face Processor Module

This module handles pure face image processing operations including
face detection, extraction, alignment, preprocessing, and quality validation.
It is source-agnostic and accepts numpy array images/frames.
"""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple, List, Dict
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaceProcessor:
    """
    Face processing class for detection, extraction, alignment, preprocessing,
    and quality validation. Accepts numpy array images/frames from any source.
    """

    def __init__(self,
                 device: str = 'cpu',
                 image_size: int = 160,
                 margin: int = 32,
                 min_face_size: int = 20,
                 detection_confidence: float = 0.5):
        """
        Initialize face processor.

        Args:
            device: Device to run detection on ('cpu' or 'cuda')
            image_size: Size of output face images
            margin: Margin around detected face
            min_face_size: Minimum face size for detection
            detection_confidence: Minimum confidence for face detection
        """
        self.device = torch.device(device)
        self.image_size = image_size
        self.margin = margin
        self.min_face_size = min_face_size
        self.detection_confidence = detection_confidence

        # Initialize OpenCV DNN face detector
        model_file = "res10_300x300_ssd_iter_140000.caffemodel"
        config_file = "deploy.prototxt"
        try:
            self.face_net = cv2.dnn.readNetFromCaffe(config_file, model_file)
        except:
            # Use Haar Cascade as fallback
            self.face_net = None
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            logger.warning("Using Haar Cascade fallback for face detection")

        logger.info(f"FaceProcessor initialized with device: {device}")

    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """
        Detect faces in an image using OpenCV DNN.

        Args:
            image: Input image as numpy array (BGR format)

        Returns:
            List of dictionaries containing face information
        """
        try:
            h, w = image.shape[:2]
            faces = []

            if self.face_net is not None:
                # Use DNN detector
                blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
                self.face_net.setInput(blob)
                detections = self.face_net.forward()

                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > self.detection_confidence:
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        faces.append({
                            'bbox': box.astype(int),
                            'confidence': float(confidence),
                            'landmarks': None,
                            'face_id': i
                        })
            else:
                # Use Haar Cascade fallback
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                detected = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                for i, (x, y, w_box, h_box) in enumerate(detected):
                    faces.append({
                        'bbox': np.array([x, y, x + w_box, y + h_box]),
                        'confidence': 0.9,
                        'landmarks': None,
                        'face_id': i
                    })

            return faces

        except Exception as e:
            logger.error(f"Face detection error: {str(e)}")
            return []

    def extract_face(self, image: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract and align face from image using bounding box.

        Args:
            image: Input image
            bbox: Bounding box [x1, y1, x2, y2]

        Returns:
            Aligned face image or None if extraction fails
        """
        try:
            x1, y1, x2, y2 = bbox.astype(int)

            # Add margin and ensure bounds
            h, w = image.shape[:2]
            x1 = max(0, x1 - self.margin)
            y1 = max(0, y1 - self.margin)
            x2 = min(w, x2 + self.margin)
            y2 = min(h, y2 + self.margin)

            # Extract face region
            face = image[y1:y2, x1:x2]

            # Resize to standard size
            face_resized = cv2.resize(face, (self.image_size, self.image_size))

            return face_resized

        except Exception as e:
            logger.error(f"Face extraction error: {str(e)}")
            return None

    def preprocess_face(self, face: np.ndarray) -> np.ndarray:
        """
        Preprocess face image for embedding extraction.

        Args:
            face: Face image as numpy array

        Returns:
            Preprocessed face image
        """
        try:
            # Convert to RGB if needed
            if len(face.shape) == 3 and face.shape[2] == 3:
                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            else:
                face_rgb = face

            # Normalize to [-1, 1] range
            face_normalized = (face_rgb.astype(np.float64) / 127.5) - 1.0

            # Convert to tensor format (C, H, W)
            face_tensor = np.transpose(face_normalized, (2, 0, 1))

            return face_tensor

        except Exception as e:
            logger.error(f"Face preprocessing error: {str(e)}")
            return None

    def validate_face_quality(self, face: np.ndarray) -> Dict[str, float]:
        """
        Validate face image quality based on various metrics.

        Args:
            face: Face image as numpy array

        Returns:
            Dictionary with quality metrics
        """
        try:
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            # Blur detection (Laplacian variance)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

            # Brightness analysis
            brightness = np.mean(gray)

            # Contrast analysis
            contrast = gray.std()

            # Face size check
            height, width = face.shape[:2]
            size_score = min(height, width)

            quality_metrics = {
                'blur_score': blur_score,
                'brightness': brightness,
                'contrast': contrast,
                'size_score': size_score,
                'overall_quality': self._calculate_overall_quality(blur_score, brightness, contrast, size_score)
            }

            return quality_metrics

        except Exception as e:
            logger.error(f"Quality validation error: {str(e)}")
            return {'overall_quality': 0.0}

    def _calculate_overall_quality(self, blur: float, brightness: float, contrast: float, size: float) -> float:
        """
        Calculate overall face quality score.

        Args:
            blur: Blur score (higher is better)
            brightness: Brightness score (50-200 is good)
            contrast: Contrast score (higher is better)
            size: Size score (larger is better)

        Returns:
            Overall quality score (0-1)
        """
        # Normalize scores
        blur_normalized = min(1.0, blur / 100.0)
        brightness_normalized = 1.0 - abs(brightness - 127.5) / 127.5
        contrast_normalized = min(1.0, contrast / 50.0)
        size_normalized = min(1.0, size / 160.0)

        # Weighted average
        overall = (blur_normalized * 0.3 +
                   brightness_normalized * 0.2 +
                   contrast_normalized * 0.2 +
                   size_normalized * 0.3)

        return overall

    def process_frame(self, frame: np.ndarray) -> List[Dict]:
        """
        Process a single frame: detect faces, extract, validate quality.

        Args:
            frame: Input frame as numpy array (BGR)

        Returns:
            List of processed face dictionaries with face image and metadata
        """
        results = []

        faces = self.detect_faces(frame)
        for face_info in faces:
            face_img = self.extract_face(frame, face_info['bbox'])
            if face_img is not None:
                quality = self.validate_face_quality(face_img)
                face_info['face_image'] = face_img
                face_info['quality'] = quality
                results.append(face_info)

        return results


# Example usage and testing
if __name__ == "__main__":
    processor = FaceProcessor()

    # Test with dummy image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Test detection
    faces = processor.detect_faces(test_image)
    print(f"Detected {len(faces)} faces")

    if faces:
        face_img = processor.extract_face(test_image, faces[0]['bbox'])
        if face_img is not None:
            print(f"Extracted face shape: {face_img.shape}")

            # Test preprocessing
            preprocessed = processor.preprocess_face(face_img)
            print(f"Preprocessed shape: {preprocessed.shape}")

            # Test quality
            quality = processor.validate_face_quality(face_img)
            print(f"Quality metrics: {quality}")

            # Test full frame processing
            results = processor.process_frame(test_image)
            print(f"Processed {len(results)} faces from frame")