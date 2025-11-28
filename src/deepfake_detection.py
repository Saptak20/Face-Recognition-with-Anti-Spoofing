"""
Deepfake Detection Module

This module implements deepfake detection using Vision Transformer (ViT) models
from Hugging Face to identify synthetic/manipulated faces and videos.
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import logging
from typing import Optional, Dict, List, Union, Tuple
from PIL import Image
import torchvision.transforms as transforms
from transformers import AutoModel, AutoProcessor, ViTModel, ViTImageProcessor
import time
from pathlib import Path
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ViTDeepfakeDetector(nn.Module):
    """
    Vision Transformer-based deepfake detection model.
    """
    
    def __init__(self, 
                 model_name: str = 'google/vit-base-patch16-224',
                 num_classes: int = 2,
                 freeze_backbone: bool = False,
                 dropout_rate: float = 0.1):
        """
        Initialize ViT deepfake detector.
        
        Args:
            model_name: Hugging Face model name
            num_classes: Number of output classes (2 for binary: real/fake)
            freeze_backbone: Whether to freeze the ViT backbone
            dropout_rate: Dropout rate for regularization
        """
        super(ViTDeepfakeDetector, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Load pretrained ViT model
        try:
            self.vit = ViTModel.from_pretrained(model_name)
            self.hidden_size = self.vit.config.hidden_size
        except Exception as e:
            logger.error(f"Failed to load ViT model: {e}")
            # Fallback to a basic structure
            self.hidden_size = 768
            self.vit = None
        
        # Freeze backbone if requested
        if freeze_backbone and self.vit:
            for param in self.vit.parameters():
                param.requires_grad = False
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(256, num_classes)
        )
        
        # Initialize classifier weights
        self._initialize_classifier()
    
    def _initialize_classifier(self):
        """Initialize classifier weights."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            pixel_values: Input images tensor
            
        Returns:
            Classification logits
        """
        if self.vit is None:
            # Fallback processing
            batch_size = pixel_values.size(0)
            features = torch.randn(batch_size, self.hidden_size, device=pixel_values.device)
        else:
            # Extract features using ViT
            outputs = self.vit(pixel_values=pixel_values)
            features = outputs.last_hidden_state[:, 0]  # Use [CLS] token
        
        # Classification
        logits = self.classifier(features)
        
        return logits


class DeepfakeDetector:
    """
    Deepfake detection system using Vision Transformer models.
    """
    
    def __init__(self,
                 model_name: str = 'google/vit-base-patch16-224',
                 model_path: Optional[str] = None,
                 device: str = 'cpu',
                 threshold: float = 0.5,
                 image_size: int = 224):
        """
        Initialize deepfake detector.
        
        Args:
            model_name: Hugging Face ViT model name
            model_path: Path to fine-tuned model weights
            device: Device to run model on
            threshold: Classification threshold
            image_size: Input image size
        """
        self.device = torch.device(device)
        self.threshold = threshold
        self.image_size = image_size
        self.model_name = model_name
        
        # Initialize model
        self.model = self._load_model(model_name, model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize image processor
        try:
            self.processor = ViTImageProcessor.from_pretrained(model_name)
        except Exception as e:
            logger.warning(f"Failed to load image processor: {e}, using default")
            self.processor = None
        
        # Fallback transform if processor fails
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        logger.info(f"DeepfakeDetector initialized with {model_name}")
    
    def _load_model(self, model_name: str, model_path: Optional[str]) -> nn.Module:
        """
        Load the deepfake detection model.
        
        Args:
            model_name: Hugging Face model name
            model_path: Path to fine-tuned weights
            
        Returns:
            Loaded model
        """
        try:
            model = ViTDeepfakeDetector(model_name=model_name, num_classes=2)
            
            # Load fine-tuned weights if provided
            if model_path and Path(model_path).exists():
                checkpoint = torch.load(model_path, map_location=self.device)
                if 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                logger.info(f"Loaded fine-tuned weights from {model_path}")
            else:
                logger.warning("No fine-tuned weights loaded, using pretrained ViT only")
            
            return model
            
        except Exception as e:
            logger.error(f"Model loading error: {str(e)}")
            # Return a simple fallback model
            return ViTDeepfakeDetector(model_name='google/vit-base-patch16-224', num_classes=2)
    
    def preprocess_image(self, image: Union[np.ndarray, Image.Image]) -> torch.Tensor:
        """
        Preprocess image for deepfake detection.
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed tensor
        """
        try:
            # Convert numpy array to PIL Image if needed
            if isinstance(image, np.ndarray):
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_pil = Image.fromarray(image.astype(np.uint8))
            else:
                image_pil = image
            
            # Use processor if available, otherwise use transform
            if self.processor:
                try:
                    inputs = self.processor(images=image_pil, return_tensors="pt")
                    return inputs['pixel_values'].to(self.device)
                except Exception as e:
                    logger.warning(f"Processor failed: {e}, using fallback transform")
            
            # Fallback to manual transform
            tensor = self.transform(image_pil)
            if len(tensor.shape) == 3:
                tensor = tensor.unsqueeze(0)
            
            return tensor.to(self.device)
            
        except Exception as e:
            logger.error(f"Image preprocessing error: {str(e)}")
            return None
    
    def detect_deepfake(self, image: Union[np.ndarray, Image.Image]) -> Dict[str, float]:
        """
        Detect deepfake in a single image.
        
        Args:
            image: Input face image
            
        Returns:
            Dictionary with deepfake detection results
        """
        try:
            # Preprocess image
            tensor = self.preprocess_image(image)
            if tensor is None:
                return {'is_deepfake': 0.0, 'confidence': 0.0, 'real_score': 0.0, 'fake_score': 0.0}
            
            # Run inference
            with torch.no_grad():
                logits = self.model(tensor)
                probabilities = torch.softmax(logits, dim=1)
                scores = probabilities.cpu().numpy()[0]
            
            # Extract scores
            real_score = float(scores[0])
            fake_score = float(scores[1])
            
            # Determine prediction
            is_deepfake = fake_score > self.threshold
            confidence = max(real_score, fake_score)
            
            return {
                'is_deepfake': float(is_deepfake),
                'real_score': real_score,
                'fake_score': fake_score,
                'confidence': confidence,
                'raw_scores': scores.tolist()
            }
            
        except Exception as e:
            logger.error(f"Deepfake detection error: {str(e)}")
            return {'is_deepfake': 0.0, 'confidence': 0.0, 'real_score': 0.0, 'fake_score': 0.0}
    
    def detect_batch_deepfake(self, images: List[Union[np.ndarray, Image.Image]]) -> List[Dict[str, float]]:
        """
        Detect deepfakes in a batch of images.
        
        Args:
            images: List of input images
            
        Returns:
            List of deepfake detection results
        """
        results = []
        
        try:
            # Preprocess all images
            tensors = []
            valid_indices = []
            
            for i, image in enumerate(images):
                tensor = self.preprocess_image(image)
                if tensor is not None:
                    tensors.append(tensor.squeeze(0))
                    valid_indices.append(i)
            
            if not tensors:
                return [{'is_deepfake': 0.0, 'confidence': 0.0, 'real_score': 0.0, 'fake_score': 0.0}] * len(images)
            
            # Stack tensors into batch
            batch_tensor = torch.stack(tensors).to(self.device)
            
            # Run inference
            with torch.no_grad():
                logits = self.model(batch_tensor)
                probabilities = torch.softmax(logits, dim=1)
                scores = probabilities.cpu().numpy()
            
            # Process results
            result_list = [{'is_deepfake': 0.0, 'confidence': 0.0, 'real_score': 0.0, 'fake_score': 0.0}] * len(images)
            
            for i, valid_idx in enumerate(valid_indices):
                real_score = float(scores[i][0])
                fake_score = float(scores[i][1])
                is_deepfake = fake_score > self.threshold
                confidence = max(real_score, fake_score)
                
                result_list[valid_idx] = {
                    'is_deepfake': float(is_deepfake),
                    'real_score': real_score,
                    'fake_score': fake_score,
                    'confidence': confidence,
                    'raw_scores': scores[i].tolist()
                }
            
            results = result_list
            
        except Exception as e:
            logger.error(f"Batch deepfake detection error: {str(e)}")
            results = [{'is_deepfake': 0.0, 'confidence': 0.0, 'real_score': 0.0, 'fake_score': 0.0}] * len(images)
        
        return results
    
    def analyze_video_frames(self, video_path: str, 
                           frame_interval: int = 30,
                           max_frames: int = 100) -> Dict[str, Union[float, List]]:
        """
        Analyze video for deepfake content by sampling frames.
        
        Args:
            video_path: Path to video file
            frame_interval: Interval between frames to analyze
            max_frames: Maximum number of frames to analyze
            
        Returns:
            Video analysis results
        """
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                logger.error(f"Cannot open video: {video_path}")
                return {'is_deepfake_video': 0.0, 'confidence': 0.0, 'frame_results': []}
            
            frames = []
            frame_count = 0
            current_frame = 0
            
            while len(frames) < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if current_frame % frame_interval == 0:
                    frames.append(frame)
                    frame_count += 1
                
                current_frame += 1
            
            cap.release()
            
            if not frames:
                return {'is_deepfake_video': 0.0, 'confidence': 0.0, 'frame_results': []}
            
            # Analyze frames
            frame_results = self.detect_batch_deepfake(frames)
            
            # Aggregate results
            fake_scores = [result['fake_score'] for result in frame_results if result['confidence'] > 0.5]
            
            if not fake_scores:
                avg_fake_score = 0.0
                video_confidence = 0.0
            else:
                avg_fake_score = np.mean(fake_scores)
                video_confidence = np.std(fake_scores)  # Higher std might indicate inconsistency
            
            is_deepfake_video = avg_fake_score > self.threshold
            
            return {
                'is_deepfake_video': float(is_deepfake_video),
                'avg_fake_score': float(avg_fake_score),
                'confidence': float(video_confidence),
                'frames_analyzed': len(frames),
                'frame_results': frame_results
            }
            
        except Exception as e:
            logger.error(f"Video analysis error: {str(e)}")
            return {'is_deepfake_video': 0.0, 'confidence': 0.0, 'frame_results': []}
    
    def extract_spatial_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extract spatial features that might indicate deepfake artifacts.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary with spatial features
        """
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Calculate gradient-based features
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Edge consistency (deepfakes might have inconsistent edges)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Frequency domain analysis
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.log(np.abs(f_shift) + 1)
            
            # Calculate frequency features
            high_freq_energy = np.sum(magnitude_spectrum > np.percentile(magnitude_spectrum, 80))
            low_freq_energy = np.sum(magnitude_spectrum <= np.percentile(magnitude_spectrum, 20))
            freq_ratio = high_freq_energy / (low_freq_energy + 1e-8)
            
            # Texture uniformity (deepfakes might have different texture properties)
            texture_variance = np.var(gray)
            texture_entropy = -np.sum(np.histogram(gray, bins=256)[0] * np.log(np.histogram(gray, bins=256)[0] + 1e-8))
            
            return {
                'gradient_magnitude_mean': float(np.mean(gradient_magnitude)),
                'gradient_magnitude_std': float(np.std(gradient_magnitude)),
                'edge_density': float(edge_density),
                'freq_ratio': float(freq_ratio),
                'texture_variance': float(texture_variance),
                'texture_entropy': float(texture_entropy)
            }
            
        except Exception as e:
            logger.error(f"Spatial feature extraction error: {str(e)}")
            return {
                'gradient_magnitude_mean': 0.0,
                'gradient_magnitude_std': 0.0,
                'edge_density': 0.0,
                'freq_ratio': 0.0,
                'texture_variance': 0.0,
                'texture_entropy': 0.0
            }
    
    def comprehensive_deepfake_analysis(self, image: Union[np.ndarray, Image.Image]) -> Dict[str, Union[float, Dict]]:
        """
        Perform comprehensive deepfake analysis combining ViT prediction and spatial features.
        
        Args:
            image: Input face image
            
        Returns:
            Comprehensive analysis results
        """
        try:
            # Get ViT-based prediction
            vit_result = self.detect_deepfake(image)
            
            # Convert to numpy array if needed
            if isinstance(image, Image.Image):
                image_np = np.array(image)
            else:
                image_np = image
            
            # Extract spatial features
            spatial_features = self.extract_spatial_features(image_np)
            
            # Heuristic combination of features
            # Real images typically have more natural gradient distributions and edge consistency
            spatial_score = min(1.0, (
                spatial_features['gradient_magnitude_std'] / 50.0 +
                spatial_features['edge_density'] * 2.0 +
                min(1.0, spatial_features['freq_ratio'] / 2.0)
            ) / 3.0)
            
            # Weighted combination
            vit_weight = 0.8
            spatial_weight = 0.2
            
            combined_fake_score = (vit_result['fake_score'] * vit_weight + 
                                 (1.0 - spatial_score) * spatial_weight)
            
            is_deepfake_combined = combined_fake_score > self.threshold
            
            return {
                'is_deepfake': float(is_deepfake_combined),
                'combined_fake_score': float(combined_fake_score),
                'vit_fake_score': vit_result['fake_score'],
                'spatial_score': float(spatial_score),
                'confidence': max(combined_fake_score, 1.0 - combined_fake_score),
                'spatial_features': spatial_features,
                'vit_raw_scores': vit_result['raw_scores']
            }
            
        except Exception as e:
            logger.error(f"Comprehensive deepfake analysis error: {str(e)}")
            return {'is_deepfake': 0.0, 'combined_fake_score': 0.0, 'confidence': 0.0}
    
    def save_model(self, filepath: str) -> bool:
        """
        Save the current model to file.
        
        Args:
            filepath: Path to save the model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            checkpoint = {
                'state_dict': self.model.state_dict(),
                'model_name': self.model_name,
                'threshold': self.threshold,
                'image_size': self.image_size
            }
            torch.save(checkpoint, filepath)
            logger.info(f"Model saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Model save error: {str(e)}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """
        Load model from file.
        
        Args:
            filepath: Path to model file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.threshold = checkpoint.get('threshold', self.threshold)
            logger.info(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Model load error: {str(e)}")
            return False
    
    def benchmark_inference_time(self, num_samples: int = 100) -> Dict[str, float]:
        """
        Benchmark the inference time of the deepfake detection model.
        
        Args:
            num_samples: Number of samples to test
            
        Returns:
            Timing statistics
        """
        try:
            # Create dummy input
            dummy_input = torch.randn(1, 3, self.image_size, self.image_size).to(self.device)
            
            # Warm up
            with torch.no_grad():
                for _ in range(10):
                    _ = self.model(dummy_input)
            
            # Benchmark
            times = []
            with torch.no_grad():
                for _ in range(num_samples):
                    start_time = time.time()
                    _ = self.model(dummy_input)
                    end_time = time.time()
                    times.append(end_time - start_time)
            
            times = np.array(times) * 1000  # Convert to milliseconds
            
            return {
                'mean_time_ms': float(np.mean(times)),
                'std_time_ms': float(np.std(times)),
                'min_time_ms': float(np.min(times)),
                'max_time_ms': float(np.max(times)),
                'fps': float(1000.0 / np.mean(times))
            }
            
        except Exception as e:
            logger.error(f"Benchmark error: {str(e)}")
            return {}


# Example usage and testing
if __name__ == "__main__":
    # Initialize deepfake detector
    detector = DeepfakeDetector(model_name='google/vit-base-patch16-224', device='cpu')
    
    # Test with dummy image
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Test single image detection
    result = detector.detect_deepfake(dummy_image)
    print(f"Deepfake detection result: {result}")
    
    # Test comprehensive analysis
    comprehensive_result = detector.comprehensive_deepfake_analysis(dummy_image)
    print(f"Comprehensive deepfake analysis: {comprehensive_result}")
    
    # Benchmark inference time
    benchmark = detector.benchmark_inference_time(num_samples=10)
    print(f"Benchmark results: {benchmark}")