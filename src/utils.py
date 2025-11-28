"""
Utilities Module

Helper functions and utilities for face recognition system including
image processing, data conversion, validation, and common operations.
"""

import cv2
import numpy as np
import logging
import time
import hashlib
import base64
import io
import json
import re
from typing import Union, List, Tuple, Optional, Dict, Any
from PIL import Image, ImageEnhance, ImageFilter
from pathlib import Path
import math
import random
from datetime import datetime, timedelta
import urllib.request
import zipfile
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageProcessor:
    """
    Image processing utilities for face recognition system.
    """
    
    @staticmethod
    def resize_image(image: np.ndarray, 
                    target_size: Tuple[int, int], 
                    maintain_aspect_ratio: bool = True) -> np.ndarray:
        """
        Resize image to target size.
        
        Args:
            image: Input image as numpy array
            target_size: Target size as (width, height)
            maintain_aspect_ratio: Whether to maintain aspect ratio
            
        Returns:
            Resized image
        """
        try:
            if maintain_aspect_ratio:
                height, width = image.shape[:2]
                target_width, target_height = target_size
                
                # Calculate scaling factor
                scale = min(target_width / width, target_height / height)
                
                # Calculate new dimensions
                new_width = int(width * scale)
                new_height = int(height * scale)
                
                # Resize image
                resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
                
                # Pad to target size if needed
                if new_width != target_width or new_height != target_height:
                    # Create blank image with target size
                    padded = np.zeros((target_height, target_width, image.shape[2]), dtype=image.dtype)
                    
                    # Calculate padding offsets
                    y_offset = (target_height - new_height) // 2
                    x_offset = (target_width - new_width) // 2
                    
                    # Place resized image in center
                    padded[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
                    
                    return padded
                else:
                    return resized
            else:
                return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
                
        except Exception as e:
            logger.error(f"Image resize error: {str(e)}")
            return image
    
    @staticmethod
    def normalize_image(image: np.ndarray, 
                       method: str = 'standard',
                       target_range: Tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
        """
        Normalize image pixel values.
        
        Args:
            image: Input image
            method: Normalization method ('standard', 'minmax', 'zscore')
            target_range: Target range for normalization
            
        Returns:
            Normalized image
        """
        try:
            image_float = image.astype(np.float32)
            
            if method == 'standard':
                # Standard normalization to [0, 1]
                normalized = image_float / 255.0
            elif method == 'minmax':
                # Min-max normalization
                min_val = np.min(image_float)
                max_val = np.max(image_float)
                if max_val > min_val:
                    normalized = (image_float - min_val) / (max_val - min_val)
                else:
                    normalized = image_float
            elif method == 'zscore':
                # Z-score normalization
                mean = np.mean(image_float)
                std = np.std(image_float)
                if std > 0:
                    normalized = (image_float - mean) / std
                else:
                    normalized = image_float - mean
            else:
                logger.warning(f"Unknown normalization method: {method}, using standard")
                normalized = image_float / 255.0
            
            # Scale to target range
            if target_range != (0.0, 1.0):
                min_target, max_target = target_range
                normalized = normalized * (max_target - min_target) + min_target
            
            return normalized
            
        except Exception as e:
            logger.error(f"Image normalization error: {str(e)}")
            return image.astype(np.float32)
    
    @staticmethod
    def enhance_image_quality(image: np.ndarray, 
                            brightness_factor: float = 1.0,
                            contrast_factor: float = 1.0,
                            sharpness_factor: float = 1.0,
                            denoise: bool = False) -> np.ndarray:
        """
        Enhance image quality using various filters.
        
        Args:
            image: Input image
            brightness_factor: Brightness enhancement factor (1.0 = no change)
            contrast_factor: Contrast enhancement factor (1.0 = no change)
            sharpness_factor: Sharpness enhancement factor (1.0 = no change)
            denoise: Whether to apply denoising
            
        Returns:
            Enhanced image
        """
        try:
            # Convert numpy array to PIL Image
            if isinstance(image, np.ndarray):
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)
                pil_image = Image.fromarray(image)
            else:
                pil_image = image
            
            # Apply brightness enhancement
            if brightness_factor != 1.0:
                enhancer = ImageEnhance.Brightness(pil_image)
                pil_image = enhancer.enhance(brightness_factor)
            
            # Apply contrast enhancement
            if contrast_factor != 1.0:
                enhancer = ImageEnhance.Contrast(pil_image)
                pil_image = enhancer.enhance(contrast_factor)
            
            # Apply sharpness enhancement
            if sharpness_factor != 1.0:
                enhancer = ImageEnhance.Sharpness(pil_image)
                pil_image = enhancer.enhance(sharpness_factor)
            
            # Apply denoising
            if denoise:
                pil_image = pil_image.filter(ImageFilter.MedianFilter(size=3))
            
            # Convert back to numpy array
            enhanced = np.array(pil_image)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Image enhancement error: {str(e)}")
            return image
    
    @staticmethod
    def detect_blur(image: np.ndarray, threshold: float = 100.0) -> Dict[str, float]:
        """
        Detect blur in image using Laplacian variance.
        
        Args:
            image: Input image
            threshold: Blur threshold (lower = more blurry)
            
        Returns:
            Dictionary with blur metrics
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Calculate Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Determine if image is blurry
            is_blurry = laplacian_var < threshold
            
            return {
                'laplacian_variance': float(laplacian_var),
                'is_blurry': bool(is_blurry),
                'blur_score': float(min(1.0, laplacian_var / threshold))
            }
            
        except Exception as e:
            logger.error(f"Blur detection error: {str(e)}")
            return {'laplacian_variance': 0.0, 'is_blurry': True, 'blur_score': 0.0}
    
    @staticmethod
    def adjust_lighting(image: np.ndarray, 
                       target_brightness: float = 127.5) -> np.ndarray:
        """
        Adjust image lighting to target brightness.
        
        Args:
            image: Input image
            target_brightness: Target mean brightness (0-255)
            
        Returns:
            Brightness-adjusted image
        """
        try:
            # Convert to grayscale to calculate brightness
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Calculate current brightness
            current_brightness = np.mean(gray)
            
            # Calculate adjustment factor
            brightness_diff = target_brightness - current_brightness
            
            # Apply brightness adjustment
            adjusted = image.astype(np.float32)
            adjusted += brightness_diff
            
            # Clip values to valid range
            adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
            
            return adjusted
            
        except Exception as e:
            logger.error(f"Lighting adjustment error: {str(e)}")
            return image


class DataConverter:
    """
    Data conversion utilities for various formats.
    """
    
    @staticmethod
    def numpy_to_base64(array: np.ndarray) -> str:
        """
        Convert numpy array to base64 string.
        
        Args:
            array: Numpy array
            
        Returns:
            Base64 encoded string
        """
        try:
            # Convert to bytes
            array_bytes = array.tobytes()
            
            # Encode to base64
            base64_str = base64.b64encode(array_bytes).decode('utf-8')
            
            return base64_str
            
        except Exception as e:
            logger.error(f"Numpy to base64 conversion error: {str(e)}")
            return ""
    
    @staticmethod
    def base64_to_numpy(base64_str: str, 
                       shape: Tuple[int, ...], 
                       dtype: np.dtype = np.float32) -> Optional[np.ndarray]:
        """
        Convert base64 string to numpy array.
        
        Args:
            base64_str: Base64 encoded string
            shape: Array shape
            dtype: Array data type
            
        Returns:
            Numpy array or None if conversion fails
        """
        try:
            # Decode from base64
            array_bytes = base64.b64decode(base64_str.encode('utf-8'))
            
            # Convert to numpy array
            array = np.frombuffer(array_bytes, dtype=dtype).reshape(shape)
            
            return array
            
        except Exception as e:
            logger.error(f"Base64 to numpy conversion error: {str(e)}")
            return None
    
    @staticmethod
    def image_to_base64(image: Union[np.ndarray, Image.Image], 
                       format: str = 'PNG') -> str:
        """
        Convert image to base64 string.
        
        Args:
            image: Image as numpy array or PIL Image
            format: Image format ('PNG', 'JPEG')
            
        Returns:
            Base64 encoded image string
        """
        try:
            # Convert numpy array to PIL Image if needed
            if isinstance(image, np.ndarray):
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)
                pil_image = Image.fromarray(image)
            else:
                pil_image = image
            
            # Convert to base64
            buffer = io.BytesIO()
            pil_image.save(buffer, format=format)
            image_bytes = buffer.getvalue()
            
            base64_str = base64.b64encode(image_bytes).decode('utf-8')
            
            return f"data:image/{format.lower()};base64,{base64_str}"
            
        except Exception as e:
            logger.error(f"Image to base64 conversion error: {str(e)}")
            return ""
    
    @staticmethod
    def base64_to_image(base64_str: str) -> Optional[np.ndarray]:
        """
        Convert base64 string to image.
        
        Args:
            base64_str: Base64 encoded image string
            
        Returns:
            Image as numpy array or None if conversion fails
        """
        try:
            # Remove data URL prefix if present
            if base64_str.startswith('data:image'):
                base64_str = base64_str.split(',')[1]
            
            # Decode from base64
            image_bytes = base64.b64decode(base64_str)
            
            # Convert to PIL Image
            pil_image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to numpy array
            image_array = np.array(pil_image)
            
            return image_array
            
        except Exception as e:
            logger.error(f"Base64 to image conversion error: {str(e)}")
            return None


class ValidationUtils:
    """
    Validation utilities for input data and parameters.
    """
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """
        Validate email address format.
        
        Args:
            email: Email address string
            
        Returns:
            True if valid email format, False otherwise
        """
        try:
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            return bool(re.match(email_pattern, email))
        except Exception:
            return False
    
    @staticmethod
    def validate_phone(phone: str) -> bool:
        """
        Validate phone number format.
        
        Args:
            phone: Phone number string
            
        Returns:
            True if valid phone format, False otherwise
        """
        try:
            # Remove common separators
            cleaned_phone = re.sub(r'[\s\-\(\)\+]', '', phone)
            
            # Check if all remaining characters are digits
            if not cleaned_phone.isdigit():
                return False
            
            # Check length (typically 10-15 digits)
            return 10 <= len(cleaned_phone) <= 15
            
        except Exception:
            return False
    
    @staticmethod
    def validate_user_id(user_id: str) -> bool:
        """
        Validate user ID format.
        
        Args:
            user_id: User ID string
            
        Returns:
            True if valid user ID format, False otherwise
        """
        try:
            # Check length
            if not (3 <= len(user_id) <= 50):
                return False
            
            # Check format (alphanumeric, underscore, hyphen)
            pattern = r'^[a-zA-Z0-9_-]+$'
            return bool(re.match(pattern, user_id))
            
        except Exception:
            return False
    
    @staticmethod
    def validate_image_array(image: np.ndarray) -> Dict[str, Any]:
        """
        Validate image numpy array.
        
        Args:
            image: Image array
            
        Returns:
            Validation results dictionary
        """
        try:
            validation_result = {
                'valid': True,
                'errors': [],
                'warnings': [],
                'info': {}
            }
            
            # Check if it's a numpy array
            if not isinstance(image, np.ndarray):
                validation_result['valid'] = False
                validation_result['errors'].append("Input is not a numpy array")
                return validation_result
            
            # Check dimensions
            if len(image.shape) not in [2, 3]:
                validation_result['valid'] = False
                validation_result['errors'].append(f"Invalid image dimensions: {image.shape}")
                return validation_result
            
            # Check channels for color images
            if len(image.shape) == 3:
                if image.shape[2] not in [1, 3, 4]:
                    validation_result['warnings'].append(f"Unusual number of channels: {image.shape[2]}")
            
            # Check data type
            if image.dtype not in [np.uint8, np.float32, np.float64]:
                validation_result['warnings'].append(f"Unusual data type: {image.dtype}")
            
            # Check value range
            min_val, max_val = np.min(image), np.max(image)
            if image.dtype == np.uint8:
                if min_val < 0 or max_val > 255:
                    validation_result['warnings'].append(f"Values outside expected range [0, 255]: [{min_val}, {max_val}]")
            elif image.dtype in [np.float32, np.float64]:
                if min_val < 0.0 or max_val > 1.0:
                    validation_result['warnings'].append(f"Float values outside [0, 1]: [{min_val}, {max_val}]")
            
            # Store info
            validation_result['info'] = {
                'shape': image.shape,
                'dtype': str(image.dtype),
                'min_value': float(min_val),
                'max_value': float(max_val),
                'mean_value': float(np.mean(image))
            }
            
            return validation_result
            
        except Exception as e:
            return {
                'valid': False,
                'errors': [f"Validation error: {str(e)}"],
                'warnings': [],
                'info': {}
            }


class SecurityUtils:
    """
    Security utilities for authentication and data protection.
    """
    
    @staticmethod
    def generate_api_key(length: int = 32) -> str:
        """
        Generate secure API key.
        
        Args:
            length: Key length
            
        Returns:
            Generated API key
        """
        try:
            # Generate random bytes
            random_bytes = np.random.bytes(length)
            
            # Encode to base64 and remove padding
            api_key = base64.urlsafe_b64encode(random_bytes).decode('utf-8').rstrip('=')
            
            return api_key
            
        except Exception as e:
            logger.error(f"API key generation error: {str(e)}")
            return ""
    
    @staticmethod
    def hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
        """
        Hash password with salt.
        
        Args:
            password: Plain text password
            salt: Optional salt (generated if not provided)
            
        Returns:
            Tuple of (hashed_password, salt)
        """
        try:
            if salt is None:
                salt = base64.urlsafe_b64encode(np.random.bytes(32)).decode('utf-8')
            
            # Hash password with salt
            password_hash = hashlib.pbkdf2_hmac('sha256', 
                                               password.encode('utf-8'), 
                                               salt.encode('utf-8'), 
                                               100000)
            
            hashed_password = base64.urlsafe_b64encode(password_hash).decode('utf-8')
            
            return hashed_password, salt
            
        except Exception as e:
            logger.error(f"Password hashing error: {str(e)}")
            return "", ""
    
    @staticmethod
    def verify_password(password: str, hashed_password: str, salt: str) -> bool:
        """
        Verify password against hash.
        
        Args:
            password: Plain text password
            hashed_password: Stored password hash
            salt: Password salt
            
        Returns:
            True if password matches, False otherwise
        """
        try:
            # Hash provided password with stored salt
            test_hash, _ = SecurityUtils.hash_password(password, salt)
            
            # Compare hashes
            return test_hash == hashed_password
            
        except Exception as e:
            logger.error(f"Password verification error: {str(e)}")
            return False


class PerformanceUtils:
    """
    Performance monitoring and optimization utilities.
    """
    
    def __init__(self):
        self.timers = {}
    
    def start_timer(self, name: str) -> None:
        """
        Start a named timer.
        
        Args:
            name: Timer name
        """
        self.timers[name] = time.time()
    
    def end_timer(self, name: str) -> float:
        """
        End a named timer and return elapsed time.
        
        Args:
            name: Timer name
            
        Returns:
            Elapsed time in seconds
        """
        if name in self.timers:
            elapsed = time.time() - self.timers[name]
            del self.timers[name]
            return elapsed
        else:
            logger.warning(f"Timer '{name}' not found")
            return 0.0
    
    @staticmethod
    def measure_memory_usage() -> Dict[str, float]:
        """
        Measure current memory usage.
        
        Returns:
            Memory usage statistics
        """
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                'rss_mb': memory_info.rss / 1024 / 1024,  # Physical memory
                'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual memory
                'percent': process.memory_percent()
            }
            
        except ImportError:
            logger.warning("psutil not available for memory monitoring")
            return {'rss_mb': 0.0, 'vms_mb': 0.0, 'percent': 0.0}
        except Exception as e:
            logger.error(f"Memory measurement error: {str(e)}")
            return {'rss_mb': 0.0, 'vms_mb': 0.0, 'percent': 0.0}


class FileUtils:
    """
    File system utilities for data management.
    """
    
    @staticmethod
    def ensure_directory(path: Union[str, Path]) -> bool:
        """
        Ensure directory exists, create if necessary.
        
        Args:
            path: Directory path
            
        Returns:
            True if directory exists or was created successfully
        """
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"Directory creation error: {str(e)}")
            return False
    
    @staticmethod
    def get_file_size(file_path: Union[str, Path]) -> int:
        """
        Get file size in bytes.
        
        Args:
            file_path: Path to file
            
        Returns:
            File size in bytes
        """
        try:
            return Path(file_path).stat().st_size
        except Exception as e:
            logger.error(f"File size error: {str(e)}")
            return 0
    
    @staticmethod
    def clean_old_files(directory: Union[str, Path], 
                       max_age_days: int = 7,
                       pattern: str = "*") -> int:
        """
        Clean old files from directory.
        
        Args:
            directory: Directory to clean
            max_age_days: Maximum file age in days
            pattern: File pattern to match
            
        Returns:
            Number of files deleted
        """
        try:
            directory_path = Path(directory)
            if not directory_path.exists():
                return 0
            
            cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)
            deleted_count = 0
            
            for file_path in directory_path.glob(pattern):
                if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()
                    deleted_count += 1
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"File cleanup error: {str(e)}")
            return 0


# Example usage and testing
if __name__ == "__main__":
    # Test image processing
    processor = ImageProcessor()
    
    # Create test image
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    # Test resize
    resized = processor.resize_image(test_image, (64, 64))
    print(f"Original shape: {test_image.shape}, Resized shape: {resized.shape}")
    
    # Test blur detection
    blur_result = processor.detect_blur(test_image)
    print(f"Blur detection: {blur_result}")
    
    # Test data converter
    converter = DataConverter()
    
    # Test numpy to base64
    test_array = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    base64_str = converter.numpy_to_base64(test_array)
    recovered_array = converter.base64_to_numpy(base64_str, test_array.shape, np.float32)
    print(f"Array conversion successful: {np.array_equal(test_array, recovered_array)}")
    
    # Test validation
    validator = ValidationUtils()
    
    print(f"Email validation: {validator.validate_email('test@example.com')}")
    print(f"Phone validation: {validator.validate_phone('+1-555-123-4567')}")
    print(f"User ID validation: {validator.validate_user_id('user_123')}")
    
    # Test performance utils
    perf = PerformanceUtils()
    
    perf.start_timer('test')
    time.sleep(0.1)
    elapsed = perf.end_timer('test')
    print(f"Timer test: {elapsed:.3f} seconds")
    
    print("Utils module test completed")
