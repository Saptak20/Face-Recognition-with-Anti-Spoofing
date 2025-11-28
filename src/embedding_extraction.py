"""
Embedding Extraction Module

This module handles face embedding extraction using pretrained models
like FaceNet/InceptionResnetV1 for generating 128D or 512D face embeddings.
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Optional, List, Union, Tuple
from PIL import Image
import cv2
from facenet_pytorch import InceptionResnetV1
from transformers import AutoModel, AutoProcessor
import torchvision.transforms as transforms
from pathlib import Path
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingExtractor:
    """
    Face embedding extraction using pretrained deep learning models.
    Supports FaceNet (InceptionResnetV1) and other transformer-based models.
    """
    
    def __init__(self, 
                 model_name: str = 'vggface2',
                 device: str = 'cpu',
                 embedding_size: int = 512,
                 pretrained: bool = True):
        """
        Initialize embedding extractor.
        
        Args:
            model_name: Model type ('vggface2', 'casia-webface', or custom)
            device: Device to run model on ('cpu' or 'cuda')
            embedding_size: Size of output embeddings
            pretrained: Whether to use pretrained weights
        """
        self.device = torch.device(device)
        self.embedding_size = embedding_size
        self.model_name = model_name
        
        # Initialize model
        self.model = self._load_model(model_name, pretrained)
        self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.ToPILImage() if isinstance(transforms.ToPILImage(), type) else transforms.Lambda(lambda x: x),
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        logger.info(f"EmbeddingExtractor initialized with model: {model_name}, device: {device}")
    
    def _load_model(self, model_name: str, pretrained: bool) -> nn.Module:
        """
        Load the embedding extraction model.
        
        Args:
            model_name: Name of the model to load
            pretrained: Whether to use pretrained weights
            
        Returns:
            Loaded PyTorch model
        """
        try:
            if model_name in ['vggface2', 'casia-webface']:
                model = InceptionResnetV1(pretrained=model_name, classify=False)
                logger.info(f"Loaded InceptionResnetV1 with {model_name} weights")
                return model
            else:
                # Custom model loading logic can be added here
                logger.warning(f"Unknown model name: {model_name}, using default InceptionResnetV1")
                return InceptionResnetV1(pretrained='vggface2', classify=False)
                
        except Exception as e:
            logger.error(f"Model loading error: {str(e)}")
            # Fallback to basic model
            return InceptionResnetV1(pretrained=None, classify=False)
    
    def preprocess_image(self, image: Union[np.ndarray, Image.Image]) -> torch.Tensor:
        """
        Preprocess image for embedding extraction.
        
        Args:
            image: Input image as numpy array or PIL Image
            
        Returns:
            Preprocessed tensor ready for model input
        """
        try:
            # Convert numpy array to PIL Image if needed
            if isinstance(image, np.ndarray):
                if len(image.shape) == 3:
                    # Convert BGR to RGB if needed
                    if image.shape[2] == 3:
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
    
    def extract_embedding(self, image: Union[np.ndarray, Image.Image]) -> Optional[np.ndarray]:
        """
        Extract face embedding from a single image.
        
        Args:
            image: Input face image
            
        Returns:
            Face embedding as numpy array or None if extraction fails
        """
        try:
            # Preprocess image
            tensor = self.preprocess_image(image)
            if tensor is None:
                return None
            
            # Extract embedding
            with torch.no_grad():
                embedding = self.model(tensor)
                
            # Convert to numpy and normalize
            embedding_np = embedding.cpu().numpy().flatten()
            embedding_normalized = self.normalize_embedding(embedding_np)
            
            return embedding_normalized
            
        except Exception as e:
            logger.error(f"Embedding extraction error: {str(e)}")
            return None
    
    def extract_batch_embeddings(self, images: List[Union[np.ndarray, Image.Image]]) -> List[Optional[np.ndarray]]:
        """
        Extract embeddings from a batch of images for efficiency.
        
        Args:
            images: List of input images
            
        Returns:
            List of embeddings (same order as input)
        """
        embeddings = []
        
        try:
            # Preprocess all images
            tensors = []
            valid_indices = []
            
            for i, image in enumerate(images):
                tensor = self.preprocess_image(image)
                if tensor is not None:
                    tensors.append(tensor.squeeze(0))  # Remove batch dim for stacking
                    valid_indices.append(i)
            
            if not tensors:
                return [None] * len(images)
            
            # Stack tensors into batch
            batch_tensor = torch.stack(tensors).to(self.device)
            
            # Extract embeddings for batch
            with torch.no_grad():
                batch_embeddings = self.model(batch_tensor)
            
            # Convert to numpy and normalize
            batch_embeddings_np = batch_embeddings.cpu().numpy()
            
            # Create result list with proper ordering
            result_embeddings = [None] * len(images)
            for i, valid_idx in enumerate(valid_indices):
                embedding = batch_embeddings_np[i]
                result_embeddings[valid_idx] = self.normalize_embedding(embedding)
            
            embeddings = result_embeddings
            
        except Exception as e:
            logger.error(f"Batch embedding extraction error: {str(e)}")
            embeddings = [None] * len(images)
        
        return embeddings
    
    def normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """
        Normalize embedding vector to unit length.
        
        Args:
            embedding: Input embedding vector
            
        Returns:
            Normalized embedding vector
        """
        try:
            norm = np.linalg.norm(embedding)
            if norm == 0:
                return embedding
            return embedding / norm
        except Exception as e:
            logger.error(f"Embedding normalization error: {str(e)}")
            return embedding
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (-1 to 1)
        """
        try:
            # Ensure embeddings are normalized
            emb1_norm = self.normalize_embedding(embedding1)
            emb2_norm = self.normalize_embedding(embedding2)
            
            # Compute cosine similarity
            similarity = np.dot(emb1_norm, emb2_norm)
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Similarity computation error: {str(e)}")
            return 0.0
    
    def compute_distance(self, embedding1: np.ndarray, embedding2: np.ndarray, metric: str = 'euclidean') -> float:
        """
        Compute distance between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            metric: Distance metric ('euclidean', 'cosine', 'manhattan')
            
        Returns:
            Distance value (lower means more similar)
        """
        try:
            if metric == 'euclidean':
                return float(np.linalg.norm(embedding1 - embedding2))
            elif metric == 'cosine':
                return float(1.0 - self.compute_similarity(embedding1, embedding2))
            elif metric == 'manhattan':
                return float(np.sum(np.abs(embedding1 - embedding2)))
            else:
                logger.warning(f"Unknown metric: {metric}, using euclidean")
                return float(np.linalg.norm(embedding1 - embedding2))
                
        except Exception as e:
            logger.error(f"Distance computation error: {str(e)}")
            return float('inf')
    
    def save_embeddings(self, embeddings: List[np.ndarray], labels: List[str], filepath: str) -> bool:
        """
        Save embeddings and labels to file.
        
        Args:
            embeddings: List of embedding vectors
            labels: List of corresponding labels
            filepath: Path to save file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            data = {
                'embeddings': embeddings,
                'labels': labels,
                'model_name': self.model_name,
                'embedding_size': self.embedding_size
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            
            logger.info(f"Saved {len(embeddings)} embeddings to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Embedding save error: {str(e)}")
            return False
    
    def load_embeddings(self, filepath: str) -> Tuple[List[np.ndarray], List[str]]:
        """
        Load embeddings and labels from file.
        
        Args:
            filepath: Path to embedding file
            
        Returns:
            Tuple of (embeddings, labels)
        """
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            embeddings = data.get('embeddings', [])
            labels = data.get('labels', [])
            
            logger.info(f"Loaded {len(embeddings)} embeddings from {filepath}")
            return embeddings, labels
            
        except Exception as e:
            logger.error(f"Embedding load error: {str(e)}")
            return [], []
    
    def get_model_info(self) -> dict:
        """
        Get information about the current model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_name': self.model_name,
            'embedding_size': self.embedding_size,
            'device': str(self.device),
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'model_trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
    
    def benchmark_inference_time(self, num_samples: int = 100) -> dict:
        """
        Benchmark the inference time of the model.
        
        Args:
            num_samples: Number of samples to test
            
        Returns:
            Dictionary with timing statistics
        """
        import time
        
        try:
            # Create dummy input
            dummy_input = torch.randn(1, 3, 160, 160).to(self.device)
            
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
                'median_time_ms': float(np.median(times)),
                'fps': float(1000.0 / np.mean(times))
            }
            
        except Exception as e:
            logger.error(f"Benchmark error: {str(e)}")
            return {}


class MultiModelEmbedding:
    """
    Ensemble of multiple embedding models for robust feature extraction.
    """
    
    def __init__(self, model_configs: List[dict], device: str = 'cpu'):
        """
        Initialize multi-model embedding system.
        
        Args:
            model_configs: List of model configuration dictionaries
            device: Device to run models on
        """
        self.device = device
        self.models = []
        
        for config in model_configs:
            try:
                model = EmbeddingExtractor(**config, device=device)
                self.models.append(model)
                logger.info(f"Loaded model: {config.get('model_name', 'unknown')}")
            except Exception as e:
                logger.error(f"Failed to load model {config}: {str(e)}")
    
    def extract_ensemble_embedding(self, image: Union[np.ndarray, Image.Image]) -> Optional[np.ndarray]:
        """
        Extract ensemble embedding by concatenating embeddings from all models.
        
        Args:
            image: Input face image
            
        Returns:
            Concatenated embedding vector or None if all models fail
        """
        try:
            embeddings = []
            
            for model in self.models:
                embedding = model.extract_embedding(image)
                if embedding is not None:
                    embeddings.append(embedding)
            
            if not embeddings:
                return None
            
            # Concatenate embeddings
            ensemble_embedding = np.concatenate(embeddings)
            
            # Normalize the final embedding
            return self.models[0].normalize_embedding(ensemble_embedding)
            
        except Exception as e:
            logger.error(f"Ensemble embedding extraction error: {str(e)}")
            return None


# Example usage and testing
if __name__ == "__main__":
    # Initialize embedding extractor
    extractor = EmbeddingExtractor(model_name='vggface2', device='cpu')
    
    # Print model info
    model_info = extractor.get_model_info()
    print(f"Model info: {model_info}")
    
    # Benchmark inference time
    benchmark = extractor.benchmark_inference_time(num_samples=10)
    print(f"Benchmark results: {benchmark}")
    
    # Test with dummy image
    dummy_image = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
    embedding = extractor.extract_embedding(dummy_image)
    
    if embedding is not None:
        print(f"Extracted embedding shape: {embedding.shape}")
        print(f"Embedding norm: {np.linalg.norm(embedding)}")
    else:
        print("Failed to extract embedding")