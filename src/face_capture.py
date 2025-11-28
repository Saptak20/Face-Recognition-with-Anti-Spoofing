"""
Face Capture Module

This module handles face detection, capture, alignment, and normalization
from webcam/video sources using OpenCV and MTCNN for robust face detection.
"""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple, List, Dict
from PIL import Image
import torch
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaceCapture:
    """
    Face capture and preprocessing class using MTCNN for face detection
    and OpenCV for video capture and image processing.
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
            device: Device to run MTCNN on ('cpu' or 'cuda')
            image_size: Size of output face images
            margin: Margin around detected face
            min_face_size: Minimum face size for detection
            thresholds: MTCNN detection thresholds
            factor: MTCNN scaling factor
        """
        self.device = torch.device(device)
        self.image_size = image_size
        self.margin = margin
        self.min_face_size = min_face_size
        self.thresholds = thresholds
        
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
        
        # Thread pool for async processing
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        logger.info(f"FaceCapture initialized with device: {device}")
    
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
                    if confidence > 0.5:
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
                        'bbox': np.array([x, y, x+w_box, y+h_box]),
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
            face_normalized = (face_rgb.astype(np.float32) / 127.5) - 1.0
            
            # Convert to tensor format (C, H, W)
            face_tensor = np.transpose(face_normalized, (2, 0, 1))
            
            return face_tensor
            
        except Exception as e:
            logger.error(f"Face preprocessing error: {str(e)}")
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
