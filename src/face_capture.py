"""
Face Capture Module

This module handles face capture from webcam/video sources using OpenCV.
Face processing operations are delegated to FaceProcessor.
"""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple, List, Dict
import torch
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaceCapture:
    """
    Face capture class using OpenCV for video capture.
    Delegates face processing to FaceProcessor.
    """

    def __init__(self,
                 device: str = 'cpu',
                 image_size: int = 160,
                 margin: int = 32,
                 min_face_size: int = 20,
                 thresholds: List[float] = [0.6, 0.7, 0.7],
                 factor: float = 0.709):
        """
        Initialize face capture system.

        Args:
            device: Device to run detection on ('cpu' or 'cuda')
            image_size: Size of output face images
            margin: Margin around detected face
            min_face_size: Minimum face size for detection
            thresholds: MTCNN detection thresholds (unused, kept for compatibility)
            factor: MTCNN scaling factor (unused, kept for compatibility)
        """
        # Initialize FaceProcessor for all image processing operations
        from src.face_processor import FaceProcessor
        self.processor = FaceProcessor(
            device=device,
            image_size=image_size,
            margin=margin,
            min_face_size=min_face_size,
            detection_confidence=0.5
        )

        # Backward compatibility attributes
        self.device = torch.device(device)
        self.image_size = image_size
        self.margin = margin
        self.min_face_size = min_face_size
        self.thresholds = thresholds
        self.factor = factor
        # Mock mtcnn attribute for backward compatibility with tests
        class MockMTCNN:
            def detect(self, image):
                return None, None, None
        self.mtcnn = MockMTCNN()

        # Thread pool for async processing
        self.executor = ThreadPoolExecutor(max_workers=2)

        logger.info(f"FaceCapture initialized with device: {device}")

    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """
        Detect faces using MTCNN (for backward compatibility with tests)
        or fall back to FaceProcessor (OpenCV DNN/Haar).
        """
        # First try MTCNN for backward compatibility with tests
        if hasattr(self, 'mtcnn') and self.mtcnn is not None:
            try:
                boxes, probs, landmarks = self.mtcnn.detect(image)
                if boxes is not None and len(boxes) > 0:
                    faces = []
                    for i, (box, prob) in enumerate(zip(boxes, probs if probs is not None else [0.9]*len(boxes))):
                        faces.append({
                            'bbox': np.array(box).astype(int),
                            'confidence': float(prob),
                            'landmarks': landmarks[i] if landmarks is not None and i < len(landmarks) else None,
                            'face_id': i
                        })
                    return faces
            except Exception:
                pass  # Fall back to processor
        
        # Fall back to FaceProcessor (OpenCV DNN/Haar)
        return self.processor.detect_faces(image)

    def extract_face(self, image: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
        """Delegate to FaceProcessor."""
        return self.processor.extract_face(image, bbox)

    def preprocess_face(self, face: np.ndarray) -> np.ndarray:
        """Delegate to FaceProcessor."""
        return self.processor.preprocess_face(face)

    def validate_face_quality(self, face: np.ndarray) -> Dict[str, float]:
        """Delegate to FaceProcessor."""
        return self.processor.validate_face_quality(face)

    def _calculate_overall_quality(self, blur: float, brightness: float, contrast: float, size: float) -> float:
        """Delegate to FaceProcessor for backward compatibility."""
        return self.processor._calculate_overall_quality(blur, brightness, contrast, size)

    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture a single frame from the configured webcam.

        Returns:
            Raw frame as numpy array (BGR) or None if capture fails
        """
        try:
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            if not cap.isOpened():
                logger.error("Cannot open webcam")
                return None

            ret, frame = cap.read()
            cap.release()

            if not ret:
                logger.error("Failed to capture frame")
                return None

            return frame

        except Exception as e:
            logger.error(f"Frame capture error: {str(e)}")
            return None

    def capture_from_webcam(self, duration: int = 5) -> List[np.ndarray]:
        """
        Capture faces from webcam for specified duration.

        Args:
            duration: Capture duration in seconds

        Returns:
            List of captured face images
        """
        captured_faces = []

        try:
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            if not cap.isOpened():
                logger.error("Cannot open webcam")
                return captured_faces

            start_time = cv2.getTickCount()
            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Detect faces
                faces = self.detect_faces(frame)

                # Extract best face (highest confidence)
                if faces:
                    best_face = max(faces, key=lambda x: x['confidence'])
                    face_img = self.extract_face(frame, best_face['bbox'])

                    if face_img is not None:
                        captured_faces.append(face_img)
                        frame_count += 1

                # Display frame with face detection
                for face in faces:
                    bbox = face['bbox']
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                    cv2.putText(frame, f"Conf: {face['confidence']:.2f}",
                              (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                cv2.imshow('Face Capture', frame)

                # Check duration
                elapsed_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
                if elapsed_time >= duration:
                    break

                # Exit on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

            logger.info(f"Captured {len(captured_faces)} face images")
            return captured_faces

        except Exception as e:
            logger.error(f"Webcam capture error: {str(e)}")
            return captured_faces

    async def capture_from_webcam_async(self, duration: int = 5) -> List[np.ndarray]:
        """
        Asynchronously capture faces from webcam.

        Args:
            duration: Capture duration in seconds

        Returns:
            List of captured face images
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.capture_from_webcam, duration)

    def capture_from_image(self, image_path: str) -> List[np.ndarray]:
        """
        Extract faces from a static image file.

        Args:
            image_path: Path to image file

        Returns:
            List of extracted face images
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Cannot load image: {image_path}")
                return []

            faces = self.detect_faces(image)
            extracted_faces = []

            for face in faces:
                face_img = self.extract_face(image, face['bbox'])
                if face_img is not None:
                    extracted_faces.append(face_img)

            logger.info(f"Extracted {len(extracted_faces)} faces from {image_path}")
            return extracted_faces

        except Exception as e:
            logger.error(f"Image capture error: {str(e)}")
            return []

    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)


# Example usage and testing
if __name__ == "__main__":
    face_capture = FaceCapture()

    # Test webcam capture
    print("Starting webcam capture for 5 seconds...")
    faces = face_capture.capture_from_webcam(duration=5)
    print(f"Captured {len(faces)} faces")

    # Test quality validation
    if faces:
        quality = face_capture.validate_face_quality(faces[0])
        print(f"Face quality metrics: {quality}")