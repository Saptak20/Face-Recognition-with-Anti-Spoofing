"""
Liveness Detection Module

This module implements CNN-based liveness detection to prevent spoofing attacks
using photos, videos, or other non-live presentations. Uses the official
Silent-Face-Anti-Spoofing multi-scale ensemble approach.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import logging
from typing import Optional, Dict, List, Tuple, Union
from PIL import Image
from pathlib import Path
import time

from src.models.mini_fas_net import MiniFASNetV2, MiniFASNetV1SE

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CropImage:
    """Port of official Silent-Face-Anti-Spoofing CropImage class."""
    
    @staticmethod
    def _get_new_box(src_w, src_h, bbox, scale):
        x = bbox[0]
        y = bbox[1]
        box_w = bbox[2]
        box_h = bbox[3]

        scale = min((src_h-1)/box_h, min((src_w-1)/box_w, scale))

        new_width = box_w * scale
        new_height = box_h * scale
        center_x, center_y = box_w/2+x, box_h/2+y

        left_top_x = center_x-new_width/2
        left_top_y = center_y-new_height/2
        right_bottom_x = center_x+new_width/2
        right_bottom_y = center_y+new_height/2

        if left_top_x < 0:
            right_bottom_x -= left_top_x
            left_top_x = 0

        if left_top_y < 0:
            right_bottom_y -= left_top_y
            left_top_y = 0

        if right_bottom_x > src_w-1:
            left_top_x -= right_bottom_x-src_w+1
            right_bottom_x = src_w-1

        if right_bottom_y > src_h-1:
            left_top_y -= right_bottom_y-src_h+1
            right_bottom_y = src_h-1

        return int(left_top_x), int(left_top_y),\
               int(right_bottom_x), int(right_bottom_y)

    def crop(self, org_img, bbox, scale, out_w, out_h, crop=True):
        if not crop:
            dst_img = cv2.resize(org_img, (out_w, out_h))
        else:
            src_h, src_w, _ = np.shape(org_img)
            left_top_x, left_top_y, \
                right_bottom_x, right_bottom_y = self._get_new_box(src_w, src_h, bbox, scale)

            img = org_img[left_top_y: right_bottom_y+1,
                          left_top_x: right_bottom_x+1]
            dst_img = cv2.resize(img, (out_w, out_h))
        return dst_img


class SilentFaceEnsembleDetector:
    """
    Liveness detection using the official Silent-Face-Anti-Spoofing multi-scale ensemble.
    
    Official implementation uses two models at different scales:
    - 2.7_80x80_MiniFASNetV2.pth (scale=2.7, MiniFASNetV2)
    - 4_0_0_80x80_MiniFASNetV1SE.pth (scale=4.0, MiniFASNetV1SE)
    
    Predictions are SUMMED across models, then argmax gives final class:
    - Class 0: Spoof (print)
    - Class 1: Live (real)
    - Class 2: Spoof (replay/unknown)
    """
    
    # Official model configurations
    ENSEMBLE_CONFIGS = [
        {
            'name': '2.7_80x80_MiniFASNetV2',
            'checkpoint_name': '2.7_80x80_MiniFASNetV2.pth',
            'model_fn': MiniFASNetV2,
            'model_args': {'embedding_size': 128, 'drop_p': 0.2, 'num_classes': 3, 'img_channel': 3},
            'scale': 2.7,
            'input_size': (80, 80),
        },
        {
            'name': '4_0_0_80x80_MiniFASNetV1SE',
            'checkpoint_name': '4_0_0_80x80_MiniFASNetV1SE.pth',
            'model_fn': MiniFASNetV1SE,
            'model_args': {'embedding_size': 128, 'drop_p': 0.75, 'num_classes': 3, 'img_channel': 3},
            'scale': 4.0,
            'input_size': (80, 80),
        },
    ]
    
    def __init__(self, 
                 model_dir: Optional[str] = None,
                 device: str = 'cpu',
                 threshold: float = 0.5,
                 face_detector=None):
        """
        Initialize ensemble liveness detector.
        
        Args:
            model_dir: Directory containing checkpoint files. If None, uses default paths.
            device: Device to run models on ('cpu' or 'cuda')
            threshold: Decision threshold for live_score (default 0.5)
            face_detector: Optional FaceCapture instance for face detection
        """
        self.device = torch.device(device)
        self.threshold = threshold
        self.face_detector = face_detector
        self.weights_loaded = False
        self.model_infos = []
        
        # Initialize face cropper
        self.cropper = CropImage()
        
        # Load ensemble models
        self.models = self._load_ensemble(model_dir)
        
        # Determine if all required models loaded
        self.weights_loaded = all(m['loaded'] for m in self.model_infos)
        
        if self.weights_loaded:
            logger.info(f"SilentFaceEnsembleDetector initialized with {len(self.models)} models")
        else:
            logger.warning(f"SilentFaceEnsembleDetector: only {len(self.models)}/{len(self.ENSEMBLE_CONFIGS)} models loaded")
        
        # Store variant info for compatibility
        self.model_variant = "SilentFaceEnsemble"
    
    def _load_ensemble(self, model_dir: Optional[str]) -> nn.ModuleList:
        """Load all ensemble models from checkpoint files."""
        models = nn.ModuleList()
        self.model_infos = []
        
        for config in self.ENSEMBLE_CONFIGS:
            info = {
                'name': config['name'],
                'scale': config['scale'],
                'input_size': config['input_size'],
                'loaded': False,
                'error': None,
            }
            
            # Determine checkpoint path
            if model_dir:
                checkpoint_path = Path(model_dir) / config['checkpoint_name']
            else:
                # Check common locations
                for base in ['/tmp', 'models', 'data/models', '.']:
                    p = Path(base) / config['checkpoint_name']
                    if p.exists():
                        checkpoint_path = p
                        break
                else:
                    # Try absolute paths for /tmp
                    p = Path('/tmp') / config['checkpoint_name']
                    if p.exists():
                        checkpoint_path = p
                    else:
                        checkpoint_path = None
            
            if not checkpoint_path or not checkpoint_path.exists():
                info['error'] = f"Checkpoint not found: {config['checkpoint_name']}"
                logger.error(info['error'])
                self.model_infos.append(info)
                continue
            
            try:
                # Create model
                model = config['model_fn'](**config['model_args'])
                
                # Load checkpoint
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                
                # Strip 'module.' prefix
                state_dict = {}
                for k, v in checkpoint.items():
                    if k.startswith('module.'):
                        state_dict[k[7:]] = v
                    else:
                        state_dict[k] = v
                
                # Load with strict=True
                model.load_state_dict(state_dict, strict=True)
                model.to(self.device)
                model.eval()
                
                models.append(model)
                info['loaded'] = True
                logger.info(f"Loaded {config['name']} (strict=True, {len(state_dict)} params)")
                
            except Exception as e:
                info['error'] = str(e)
                logger.error(f"Failed to load {config['name']}: {e}")
            
            self.model_infos.append(info)
        
        return models
    
    def get_model_info(self) -> Dict:
        """Get information about the ensemble."""
        return {
            'model_variant': self.model_variant,
            'ensemble_size': len(self.models),
            'weights_loaded': self.weights_loaded,
            'threshold': self.threshold,
            'models': self.model_infos,
        }
    
    def _detect_and_crop(self, image: np.ndarray, scale: float, input_size: Tuple[int, int]) -> np.ndarray:
        """
        Detect face and crop using official preprocessing.
        
        Args:
            image: RGB image (H, W, 3)
            scale: Face crop scale factor
            input_size: (width, height) for output
            
        Returns:
            Cropped face as RGB array (input_size[1], input_size[0], 3)
        """
        # Use provided face detector or fallback to Haar cascade
        if self.face_detector is not None:
            # FaceCapture expects BGR
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            detections = self.face_detector.detect_faces(image_bgr)
            
            if detections:
                best = max(detections, key=lambda d: d['confidence'])
                bbox = best['bbox']  # [x1, y1, x2, y2]
                x, y, x2, y2 = bbox
                box_w = x2 - x
                box_h = y2 - y
                bbox_wh = [x, y, box_w, box_h]
                return self.cropper.crop(image, bbox_wh, scale, input_size[0], input_size[1], crop=True)
        
        # Fallback: Haar cascade with more sensitive parameters
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        
        # Try multiple parameter combinations for better detection
        detected = []
        for scale in [1.05, 1.1, 1.2]:
            for neighbors in [3, 4]:
                detected = face_cascade.detectMultiScale(gray, scale, neighbors)
                if len(detected) > 0:
                    break
            if len(detected) > 0:
                break
        
        if len(detected) > 0:
            x, y, w, h = max(detected, key=lambda d: d[2] * d[3])
            return self.cropper.crop(image, [x, y, w, h], scale, input_size[0], input_size[1], crop=True)
        
        # Ultimate fallback: center crop
        logger.warning("No face detected, using center crop")
        h, w = image.shape[:2]
        size = min(h, w)
        y = (h - size) // 2
        x = (w - size) // 2
        crop = image[y:y+size, x:x+size]
        return cv2.resize(crop, input_size)
    
    def _preprocess_cropped(self, cropped: np.ndarray) -> torch.Tensor:
        """
        Preprocess cropped face for model input.
        
        Official preprocessing: ToTensor() only
        - Converts HWC to CHW
        - Converts to float32
        - Scales to [0, 1] range (divides by 255)
        - NO ImageNet normalization
        """
        # HWC to CHW
        img = np.transpose(cropped, (2, 0, 1))
        # Add batch dim, convert to float, scale to [0, 1]
        img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0
        return torch.from_numpy(img)
    
    def _predict_single_model(self, model: nn.Module, image: np.ndarray, 
                               scale: float, input_size: Tuple[int, int]) -> np.ndarray:
        """Run inference on a single model."""
        # Crop face for this model's scale
        cropped = self._detect_and_crop(image, scale, input_size)
        
        # Preprocess
        tensor = self._preprocess_cropped(cropped).to(self.device)
        
        # Inference
        model.eval()
        with torch.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        
        return probs
    
    def detect_liveness(self, image: Union[np.ndarray, Image.Image]) -> Dict[str, float]:
        """
        Detect liveness using ensemble of models.
        
        Returns:
            Dict with:
            - is_live: bool (live_score > threshold)
            - live_score: float (probability of class 1)
            - fake_score: float (probability of class 0 + class 2)
            - confidence: float
            - raw_scores: list of 3 probabilities
            - ensemble_details: per-model predictions
        """
        try:
            # Convert to RGB numpy array
            if isinstance(image, Image.Image):
                image_np = np.array(image.convert('RGB'))
            else:
                image_np = image
                if image_np.shape[2] == 3:
                    # Assume BGR from OpenCV, convert to RGB
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            
            # Run each model in ensemble
            ensemble_probs = np.zeros(3)
            model_details = []
            
            for i, model in enumerate(self.models):
                config = self.ENSEMBLE_CONFIGS[i]
                probs = self._predict_single_model(
                    model, image_np, config['scale'], config['input_size']
                )
                ensemble_probs += probs
                
                model_details.append({
                    'model': config['name'],
                    'scale': config['scale'],
                    'probs': probs.tolist(),
                    'predicted_class': int(np.argmax(probs)),
                })
            
            # Official decision: argmax of SUMMED predictions
            final_probs = ensemble_probs
            predicted_class = int(np.argmax(final_probs))
            
            # Class mapping: 0=spoof(print), 1=live, 2=spoof(replay)
            live_score = float(final_probs[1])
            spoof_score = float(final_probs[0] + final_probs[2])
            
            is_live = live_score > self.threshold
            confidence = max(live_score, spoof_score)
            
            return {
                'is_live': float(is_live),
                'live_score': live_score,
                'fake_score': spoof_score,
                'confidence': confidence,
                'raw_scores': final_probs.tolist(),
                'ensemble_details': model_details,
            }
            
        except Exception as e:
            logger.error(f"Ensemble liveness detection error: {str(e)}")
            return self._empty_result()
    
    def detect_batch_liveness(self, images: List[Union[np.ndarray, Image.Image]]) -> List[Dict[str, float]]:
        """Batch version of detect_liveness."""
        return [self.detect_liveness(img) for img in images]
    
    def _empty_result(self) -> Dict[str, float]:
        """Return empty result for failed detection."""
        return {
            'is_live': 0.0,
            'live_score': 0.0,
            'fake_score': 0.0,
            'confidence': 0.0,
            'raw_scores': [0.0, 0.0, 0.0],
            'ensemble_details': [],
        }


# Backward compatibility wrapper
class SilentFaceLivenessDetector:
    """
    Backward-compatible wrapper that uses the ensemble detector.
    
    Maintains the same interface as the original SilentFaceLivenessDetector.
    """
    
    def __init__(self, 
                 model_type: str = 'mobilenet',
                 model_path: Optional[str] = None,
                 device: str = 'cpu',
                 input_size: int = 64,
                 threshold: float = 0.5):
        """
        Initialize liveness detector.
        
        Note: model_type and model_path are now used to locate the ensemble directory.
        The ensemble always uses both official checkpoints.
        """
        self.device = torch.device(device)
        self.input_size = input_size
        self.threshold = threshold
        self.model_type = 'SilentFaceEnsemble'
        self.model_variant = 'SilentFaceEnsemble'
        self.weights_loaded = False
        
        # Determine model directory from model_path
        model_dir = None
        if model_path:
            model_dir = str(Path(model_path).parent)
        
        # Initialize ensemble detector
        self.ensemble = SilentFaceEnsembleDetector(
            model_dir=model_dir,
            device=device,
            threshold=threshold,
        )
        
        self.weights_loaded = self.ensemble.weights_loaded
        
        logger.info(f"SilentFaceLivenessDetector (ensemble) initialized: weights_loaded={self.weights_loaded}")
    
    def get_model_info(self) -> Dict:
        return self.ensemble.get_model_info()
    
    def detect_liveness(self, image: Union[np.ndarray, Image.Image]) -> Dict[str, float]:
        return self.ensemble.detect_liveness(image)
    
    def detect_batch_liveness(self, images: List[Union[np.ndarray, Image.Image]]) -> List[Dict[str, float]]:
        return self.ensemble.detect_batch_liveness(images)
    
    # Keep other methods for backward compatibility
    def analyze_temporal_consistency(self, video_frames: List[np.ndarray], 
                                    window_size: int = 5) -> Dict[str, float]:
        # Delegate to ensemble's single-frame detection
        frame_results = self.detect_batch_liveness(video_frames)
        live_scores = [r['live_score'] for r in frame_results]
        
        if len(live_scores) < window_size:
            return {'temporal_score': 0.0, 'consistency': 0.0}
        
        temporal_features = []
        for i in range(len(live_scores) - window_size + 1):
            window = live_scores[i:i+window_size]
            variance = np.var(window)
            trend = abs(np.polyfit(np.arange(len(window)), window, 1)[0])
            temporal_features.append({'variance': variance, 'trend': trend, 'mean_score': np.mean(window)})
        
        avg_variance = np.mean([f['variance'] for f in temporal_features])
        avg_trend = np.mean([f['trend'] for f in temporal_features])
        overall_mean = np.mean(live_scores)
        
        variance_score = 1.0 - min(1.0, avg_variance / 0.1)
        trend_score = 1.0 - min(1.0, avg_trend / 0.05)
        temporal_score = (variance_score + trend_score + overall_mean) / 3.0
        
        return {
            'temporal_score': float(temporal_score),
            'consistency': float(variance_score),
            'avg_variance': float(avg_variance),
            'avg_trend': float(avg_trend),
            'overall_mean': float(overall_mean)
        }
    
    def extract_texture_features(self, image: np.ndarray) -> Dict[str, float]:
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            lbp = np.zeros_like(gray)
            for i in range(1, gray.shape[0] - 1):
                for j in range(1, gray.shape[1] - 1):
                    center = gray[i, j]
                    binary = ''
                    for k in range(8):
                        angle = 2 * np.pi * k / 8
                        x = int(i + np.cos(angle))
                        y = int(j + np.sin(angle))
                        binary += '1' if gray[x, y] >= center else '0'
                    lbp[i, j] = int(binary, 2)
            
            edges = cv2.Canny(gray, 50, 150)
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            
            return {
                'lbp_variance': float(np.var(lbp)),
                'contrast': float(gray.std()),
                'edge_density': float(np.sum(edges > 0) / edges.size),
                'avg_gradient': float(np.mean(grad_mag))
            }
        except Exception as e:
            logger.error(f"Texture feature extraction error: {str(e)}")
            return {'lbp_variance': 0.0, 'contrast': 0.0, 'edge_density': 0.0, 'avg_gradient': 0.0}
    
    def comprehensive_liveness_check(self, image: Union[np.ndarray, Image.Image]) -> Dict[str, float]:
        cnn_result = self.detect_liveness(image)
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
        texture = self.extract_texture_features(image_np)
        
        texture_score = min(1.0, (texture['lbp_variance'] / 1000.0 + 
                                  texture['contrast'] / 50.0 + 
                                  texture['edge_density'] * 2.0) / 3.0)
        
        combined = cnn_result['live_score'] * 0.7 + texture_score * 0.3
        
        return {
            'is_live': float(combined > self.threshold),
            'combined_score': float(combined),
            'cnn_score': cnn_result['live_score'],
            'texture_score': float(texture_score),
            'confidence': max(combined, 1.0 - combined),
            'texture_features': texture,
            'cnn_raw_scores': cnn_result['raw_scores']
        }
    
    def save_model(self, filepath: str) -> bool:
        try:
            checkpoint = {
                'ensemble_state': {m['name']: m.state_dict() for m in self.ensemble.model_infos},
                'threshold': self.threshold
            }
            torch.save(checkpoint, filepath)
            return True
        except Exception as e:
            logger.error(f"Model save error: {str(e)}")
            return False
    
    def benchmark_inference_time(self, num_samples: int = 100) -> Dict[str, float]:
        try:
            dummy = torch.randn(1, 3, 80, 80).to(self.device)
            # Warmup
            with torch.no_grad():
                for m in self.ensemble.models:
                    for _ in range(10):
                        _ = m(dummy)
            
            times = []
            with torch.no_grad():
                for _ in range(num_samples):
                    start = time.time()
                    for m in self.ensemble.models:
                        _ = m(dummy)
                    times.append(time.time() - start)
            
            times = np.array(times) * 1000
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


# Keep legacy model classes for backward compatibility
class MobileNetLiveness(nn.Module):
    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU6(True),
            nn.Conv2d(32, 64, 3, 1, 1, groups=32, bias=False), nn.BatchNorm2d(64), nn.ReLU6(True),
            nn.Conv2d(64, 64, 1, 1, 0, bias=False), nn.BatchNorm2d(64), nn.ReLU6(True),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout_rate), nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class CustomCNNLiveness(nn.Module):
    def __init__(self, num_classes: int = 2, input_size: int = 64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True), nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(256, 128), nn.ReLU(True), nn.Dropout(0.3), nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)