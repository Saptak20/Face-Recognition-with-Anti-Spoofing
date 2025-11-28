"""
Liveness Detection Module

This module implements CNN-based liveness detection to prevent spoofing attacks
using photos, videos, or other non-live presentations. Uses lightweight models
like MobileNet or custom CNN architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import logging
from typing import Optional, Dict, List, Tuple, Union
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import pickle
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MobileNetLiveness(nn.Module):
    """
    MobileNet-based lightweight liveness detection model.
    """
    
    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.2):
        """
        Initialize MobileNet liveness model.
        
        Args:
            num_classes: Number of output classes (2 for binary: real/fake)
            dropout_rate: Dropout rate for regularization
        """
        super(MobileNetLiveness, self).__init__()
        
        # Depthwise separable convolution blocks
        self.conv1 = self._conv_block(3, 32, 2)
        self.conv2 = self._depthwise_block(32, 64, 1)
        self.conv3 = self._depthwise_block(64, 128, 2)
        self.conv4 = self._depthwise_block(128, 128, 1)
        self.conv5 = self._depthwise_block(128, 256, 2)
        self.conv6 = self._depthwise_block(256, 256, 1)
        self.conv7 = self._depthwise_block(256, 512, 2)
        
        # Multiple depthwise blocks
        self.conv8_13 = nn.Sequential(*[
            self._depthwise_block(512, 512, 1) for _ in range(5)
        ])
        
        self.conv14 = self._depthwise_block(512, 1024, 2)
        self.conv15 = self._depthwise_block(1024, 1024, 1)
        
        # Global average pooling and classifier
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(1024, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _conv_block(self, in_channels: int, out_channels: int, stride: int) -> nn.Sequential:
        """Create standard convolution block."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )
    
    def _depthwise_block(self, in_channels: int, out_channels: int, stride: int) -> nn.Sequential:
        """Create depthwise separable convolution block."""
        return nn.Sequential(
            # Depthwise convolution
            nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),
            
            # Pointwise convolution
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8_13(x)
        x = self.conv14(x)
        x = self.conv15(x)
        
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.classifier(x)
        
        return x


class CustomCNNLiveness(nn.Module):
    """
    Custom lightweight CNN for liveness detection.
    """
    
    def __init__(self, num_classes: int = 2, input_size: int = 64):
        """
        Initialize custom CNN liveness model.
        
        Args:
            num_classes: Number of output classes
            input_size: Input image size (assumes square images)
        """
        super(CustomCNNLiveness, self).__init__()
        
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Second block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Third block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Fourth block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class LivenessDetector:
    """
    Liveness detection system using CNN models to detect spoofing attacks.
    """
    
    def __init__(self, 
                 model_type: str = 'mobilenet',
                 model_path: Optional[str] = None,
                 device: str = 'cpu',
                 input_size: int = 64,
                 threshold: float = 0.5):
        """
        Initialize liveness detector.
        
        Args:
            model_type: Type of model ('mobilenet', 'custom_cnn')
            model_path: Path to pretrained model weights
            device: Device to run model on
            input_size: Input image size
            threshold: Classification threshold
        """
        self.device = torch.device(device)
        self.input_size = input_size
        self.threshold = threshold
        self.model_type = model_type
        
        # Initialize model
        self.model = self._load_model(model_type, model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        logger.info(f"LivenessDetector initialized with {model_type} model")
    
    def _load_model(self, model_type: str, model_path: Optional[str]) -> nn.Module:
        """
        Load the liveness detection model.
        
        Args:
            model_type: Type of model to load
            model_path: Path to model weights
            
        Returns:
            Initialized model
        """
        try:
            if model_type == 'mobilenet':
                model = MobileNetLiveness(num_classes=2)
            elif model_type == 'custom_cnn':
                model = CustomCNNLiveness(num_classes=2, input_size=self.input_size)
            else:
                logger.warning(f"Unknown model type: {model_type}, using custom_cnn")
                model = CustomCNNLiveness(num_classes=2, input_size=self.input_size)
            
            # Load pretrained weights if provided
            if model_path and Path(model_path).exists():
                checkpoint = torch.load(model_path, map_location=self.device)
                if 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                logger.info(f"Loaded pretrained weights from {model_path}")
            else:
                logger.warning("No pretrained weights loaded, using random initialization")
            
            return model
            
        except Exception as e:
            logger.error(f"Model loading error: {str(e)}")
            return CustomCNNLiveness(num_classes=2, input_size=self.input_size)
    
    def preprocess_image(self, image: Union[np.ndarray, Image.Image]) -> torch.Tensor:
        """
        Preprocess image for liveness detection.
        
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
            
            # Apply transformations
            tensor = self.transform(image_pil)
            
            # Add batch dimension
            if len(tensor.shape) == 3:
                tensor = tensor.unsqueeze(0)
            
            return tensor.to(self.device)
            
        except Exception as e:
            logger.error(f"Image preprocessing error: {str(e)}")
            return None
    
    def detect_liveness(self, image: Union[np.ndarray, Image.Image]) -> Dict[str, float]:
        """
        Detect liveness in a single image.
        
        Args:
            image: Input face image
            
        Returns:
            Dictionary with liveness scores and prediction
        """
        try:
            # Preprocess image
            tensor = self.preprocess_image(image)
            if tensor is None:
                return {'is_live': 0.0, 'confidence': 0.0, 'raw_scores': [0.0, 0.0]}
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(tensor)
                probabilities = F.softmax(outputs, dim=1)
                scores = probabilities.cpu().numpy()[0]
            
            # Extract scores
            fake_score = float(scores[0])
            live_score = float(scores[1])
            
            # Determine prediction
            is_live = live_score > self.threshold
            confidence = max(fake_score, live_score)
            
            return {
                'is_live': float(is_live),
                'live_score': live_score,
                'fake_score': fake_score,
                'confidence': confidence,
                'raw_scores': scores.tolist()
            }
            
        except Exception as e:
            logger.error(f"Liveness detection error: {str(e)}")
            return {'is_live': 0.0, 'confidence': 0.0, 'raw_scores': [0.0, 0.0]}
    
    def detect_batch_liveness(self, images: List[Union[np.ndarray, Image.Image]]) -> List[Dict[str, float]]:
        """
        Detect liveness in a batch of images.
        
        Args:
            images: List of input images
            
        Returns:
            List of liveness detection results
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
                return [{'is_live': 0.0, 'confidence': 0.0, 'raw_scores': [0.0, 0.0]}] * len(images)
            
            # Stack tensors into batch
            batch_tensor = torch.stack(tensors).to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                probabilities = F.softmax(outputs, dim=1)
                scores = probabilities.cpu().numpy()
            
            # Process results
            result_list = [{'is_live': 0.0, 'confidence': 0.0, 'raw_scores': [0.0, 0.0]}] * len(images)
            
            for i, valid_idx in enumerate(valid_indices):
                fake_score = float(scores[i][0])
                live_score = float(scores[i][1])
                is_live = live_score > self.threshold
                confidence = max(fake_score, live_score)
                
                result_list[valid_idx] = {
                    'is_live': float(is_live),
                    'live_score': live_score,
                    'fake_score': fake_score,
                    'confidence': confidence,
                    'raw_scores': scores[i].tolist()
                }
            
            results = result_list
            
        except Exception as e:
            logger.error(f"Batch liveness detection error: {str(e)}")
            results = [{'is_live': 0.0, 'confidence': 0.0, 'raw_scores': [0.0, 0.0]}] * len(images)
        
        return results
    
    def analyze_temporal_consistency(self, video_frames: List[np.ndarray], 
                                   window_size: int = 5) -> Dict[str, float]:
        """
        Analyze temporal consistency across video frames for enhanced liveness detection.
        
        Args:
            video_frames: List of video frames
            window_size: Size of temporal window for analysis
            
        Returns:
            Temporal consistency analysis results
        """
        try:
            if len(video_frames) < window_size:
                logger.warning(f"Not enough frames for temporal analysis: {len(video_frames)} < {window_size}")
                return {'temporal_score': 0.0, 'consistency': 0.0}
            
            # Get liveness scores for all frames
            frame_results = self.detect_batch_liveness(video_frames)
            live_scores = [result['live_score'] for result in frame_results]
            
            # Calculate temporal features
            temporal_features = []
            
            for i in range(len(live_scores) - window_size + 1):
                window_scores = live_scores[i:i+window_size]
                
                # Calculate variance in window
                variance = np.var(window_scores)
                
                # Calculate trend (slope)
                x = np.arange(len(window_scores))
                trend = np.polyfit(x, window_scores, 1)[0]
                
                temporal_features.append({
                    'variance': variance,
                    'trend': abs(trend),
                    'mean_score': np.mean(window_scores)
                })
            
            # Aggregate temporal features
            avg_variance = np.mean([f['variance'] for f in temporal_features])
            avg_trend = np.mean([f['trend'] for f in temporal_features])
            overall_mean = np.mean(live_scores)
            
            # Calculate temporal consistency score
            # Real faces should have some natural variation but not too much
            variance_score = 1.0 - min(1.0, avg_variance / 0.1)  # Normalize by expected variance
            trend_score = 1.0 - min(1.0, avg_trend / 0.05)       # Normalize by expected trend
            
            temporal_score = (variance_score + trend_score + overall_mean) / 3.0
            
            return {
                'temporal_score': float(temporal_score),
                'consistency': float(variance_score),
                'avg_variance': float(avg_variance),
                'avg_trend': float(avg_trend),
                'overall_mean': float(overall_mean)
            }
            
        except Exception as e:
            logger.error(f"Temporal consistency analysis error: {str(e)}")
            return {'temporal_score': 0.0, 'consistency': 0.0}
    
    def extract_texture_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extract texture features that can help distinguish real faces from photos.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary with texture features
        """
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Calculate LBP (Local Binary Pattern)
            def calculate_lbp(img, radius=1, n_points=8):
                """Calculate Local Binary Pattern."""
                lbp = np.zeros_like(img)
                for i in range(radius, img.shape[0] - radius):
                    for j in range(radius, img.shape[1] - radius):
                        center = img[i, j]
                        binary_string = ''
                        for k in range(n_points):
                            angle = 2 * np.pi * k / n_points
                            x = int(i + radius * np.cos(angle))
                            y = int(j + radius * np.sin(angle))
                            if img[x, y] >= center:
                                binary_string += '1'
                            else:
                                binary_string += '0'
                        lbp[i, j] = int(binary_string, 2)
                return lbp
            
            lbp = calculate_lbp(gray)
            lbp_variance = np.var(lbp)
            
            # Calculate texture contrast
            contrast = gray.std()
            
            # Calculate edge density
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Calculate gradient magnitude
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            avg_gradient = np.mean(gradient_magnitude)
            
            return {
                'lbp_variance': float(lbp_variance),
                'contrast': float(contrast),
                'edge_density': float(edge_density),
                'avg_gradient': float(avg_gradient)
            }
            
        except Exception as e:
            logger.error(f"Texture feature extraction error: {str(e)}")
            return {'lbp_variance': 0.0, 'contrast': 0.0, 'edge_density': 0.0, 'avg_gradient': 0.0}
    
    def comprehensive_liveness_check(self, image: Union[np.ndarray, Image.Image]) -> Dict[str, float]:
        """
        Perform comprehensive liveness check combining CNN prediction and texture analysis.
        
        Args:
            image: Input face image
            
        Returns:
            Comprehensive liveness analysis results
        """
        try:
            # Get CNN-based liveness prediction
            cnn_result = self.detect_liveness(image)
            
            # Convert to numpy array if needed
            if isinstance(image, Image.Image):
                image_np = np.array(image)
            else:
                image_np = image
            
            # Extract texture features
            texture_features = self.extract_texture_features(image_np)
            
            # Combine results
            # Real faces typically have higher texture variance and contrast
            texture_score = min(1.0, (texture_features['lbp_variance'] / 1000.0 + 
                                    texture_features['contrast'] / 50.0 + 
                                    texture_features['edge_density'] * 2.0) / 3.0)
            
            # Weighted combination of CNN and texture scores
            cnn_weight = 0.7
            texture_weight = 0.3
            
            combined_score = (cnn_result['live_score'] * cnn_weight + 
                            texture_score * texture_weight)
            
            is_live_combined = combined_score > self.threshold
            
            return {
                'is_live': float(is_live_combined),
                'combined_score': float(combined_score),
                'cnn_score': cnn_result['live_score'],
                'texture_score': float(texture_score),
                'confidence': max(combined_score, 1.0 - combined_score),
                'texture_features': texture_features,
                'cnn_raw_scores': cnn_result['raw_scores']
            }
            
        except Exception as e:
            logger.error(f"Comprehensive liveness check error: {str(e)}")
            return {'is_live': 0.0, 'combined_score': 0.0, 'confidence': 0.0}
    
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
                'model_type': self.model_type,
                'input_size': self.input_size,
                'threshold': self.threshold
            }
            torch.save(checkpoint, filepath)
            logger.info(f"Model saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Model save error: {str(e)}")
            return False
    
    def benchmark_inference_time(self, num_samples: int = 100) -> Dict[str, float]:
        """
        Benchmark the inference time of the liveness detection model.
        
        Args:
            num_samples: Number of samples to test
            
        Returns:
            Timing statistics
        """
        try:
            # Create dummy input
            dummy_input = torch.randn(1, 3, self.input_size, self.input_size).to(self.device)
            
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
    # Initialize liveness detector
    detector = LivenessDetector(model_type='custom_cnn', input_size=64)
    
    # Test with dummy image
    dummy_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    
    # Test single image detection
    result = detector.detect_liveness(dummy_image)
    print(f"Liveness detection result: {result}")
    
    # Test comprehensive liveness check
    comprehensive_result = detector.comprehensive_liveness_check(dummy_image)
    print(f"Comprehensive liveness result: {comprehensive_result}")
    
    # Benchmark inference time
    benchmark = detector.benchmark_inference_time(num_samples=10)
    print(f"Benchmark results: {benchmark}")