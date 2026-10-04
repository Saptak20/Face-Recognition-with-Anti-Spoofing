"""
Embedding Extraction Module

This module handles face embedding extraction using genuinely pretrained
facial recognition models (FaceNet Inception-ResNet-v1 trained on VGGFace2).
Produces 512-dimensional L2-normalized identity embeddings.
"""

import os
import logging
from typing import Optional, List, Union, Tuple, Dict, Any
from pathlib import Path
import pickle

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import cv2

from src.models.inception_resnet_v1 import InceptionResnetV1

logger = logging.getLogger(__name__)


class EmbeddingExtractor:
    """
    Face embedding extraction using genuinely pretrained deep learning models.
    Default backbone: InceptionResnetV1 pretrained on VGGFace2 (FaceNet).
    Outputs 512-dimensional L2-normalized identity embeddings.
    """

    def __init__(self,
                 model_name: str = 'vggface2',
                 device: str = 'cpu',
                 embedding_size: int = 512,
                 pretrained: bool = True,
                 checkpoint_path: Optional[str] = None):
        """
        Initialize embedding extractor.

        Args:
            model_name: Model backbone ('vggface2' or 'casia-webface')
            device: Device to run model on ('cpu' or 'cuda')
            embedding_size: Size of output embeddings (must be 512 for InceptionResnetV1)
            pretrained: Whether to use pretrained weights
            checkpoint_path: Optional explicit local path to pretrained weights file
        """
        self.device = torch.device(device if (device != 'cuda' or torch.cuda.is_available()) else 'cpu')
        self.embedding_size = embedding_size
        self.model_name = model_name
        self.pretrained = pretrained
        self.checkpoint_path = checkpoint_path

        if embedding_size != 512:
            logger.warning(
                f"InceptionResnetV1 natively outputs 512-dimensional embeddings; requested size {embedding_size}."
            )

        # Initialize model
        self.model = self._load_model(model_name, pretrained, checkpoint_path=checkpoint_path)
        self.model.to(self.device)
        self.model.eval()

        # Image preprocessing pipeline (160x160 RGB, standardized to [-1, 1])
        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        logger.info(
            f"EmbeddingExtractor initialized with model: {model_name}, "
            f"device: {self.device}, pretrained: {pretrained}"
        )

    def _load_model(self,
                    model_name: str,
                    pretrained: bool,
                    checkpoint_path: Optional[str] = None) -> nn.Module:
        """
        Load the embedding extraction model with genuine pretrained weights.

        Args:
            model_name: Name of the model dataset to load ('vggface2' or 'casia-webface')
            pretrained: Whether to load pretrained weights
            checkpoint_path: Optional explicit local path to weights file

        Returns:
            InceptionResnetV1 model in feature extraction mode (classify=False).

        Raises:
            ValueError: If model_name is unsupported.
            RuntimeError: If pretrained weights cannot be loaded. Never silently falls back.
        """
        valid_models = ('vggface2', 'casia-webface')
        if model_name not in valid_models:
            raise ValueError(
                f"Unsupported embedding model '{model_name}'. Supported models: {valid_models}"
            )

        try:
            logger.info(
                f"Loading InceptionResnetV1 (pretrained={model_name if pretrained else None})..."
            )
            model = InceptionResnetV1(
                pretrained=model_name if pretrained else None,
                classify=False,
                checkpoint_path=checkpoint_path
            )
            return model
        except Exception as e:
            logger.error(
                f"Failed to load pretrained face recognition model '{model_name}': {e}"
            )
            # CRITICAL: Fail explicitly. Do NOT silently substitute random weights.
            raise RuntimeError(
                f"Failed to load pretrained face recognition model '{model_name}': {e}. "
                "Explicit error raised: unverified or random weights will not be substituted."
            ) from e

    def preprocess_image(self, image: Union[np.ndarray, Image.Image]) -> Optional[torch.Tensor]:
        """
        Preprocess input image for embedding extraction.
        Validates input geometry, channels, and numerical stability.

        Args:
            image: Input face crop as numpy array (H, W, C) or PIL Image

        Returns:
            Preprocessed tensor (1, 3, 160, 160) on self.device, or None if invalid
        """
        if image is None:
            logger.error("Image preprocessing error: Input image is None")
            return None

        try:
            # Handle numpy array
            if isinstance(image, np.ndarray):
                if image.size == 0:
                    logger.error("Image preprocessing error: Empty numpy array")
                    return None

                # Check for NaN / Inf
                if not np.all(np.isfinite(image)):
                    logger.error("Image preprocessing error: Non-finite values detected in image")
                    return None

                # Validate dimensions
                if image.ndim == 2:
                    # Grayscale (H, W) -> RGB
                    image_rgb = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
                elif image.ndim == 3:
                    h, w, c = image.shape
                    if h < 10 or w < 10:
                        logger.error(f"Image preprocessing error: Image dimensions too small ({w}x{h})")
                        return None

                    if c == 1:
                        image_rgb = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
                    elif c == 3:
                        # Assuming BGR from OpenCV, convert to RGB
                        image_rgb = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)
                    elif c == 4:
                        # BGRA -> RGB
                        image_rgb = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGRA2RGB)
                    else:
                        logger.error(f"Image preprocessing error: Unsupported channel count ({c})")
                        return None
                else:
                    logger.error(f"Image preprocessing error: Unsupported array ndim ({image.ndim})")
                    return None

                image_pil = Image.fromarray(image_rgb)

            elif isinstance(image, Image.Image):
                if image.width < 10 or image.height < 10:
                    logger.error(f"Image preprocessing error: PIL Image too small ({image.width}x{image.height})")
                    return None
                image_pil = image.convert('RGB')
            else:
                logger.error(f"Image preprocessing error: Unsupported type {type(image)}")
                return None

            # Apply transformations (Resize -> ToTensor [0, 1] -> Normalize [-1, 1])
            tensor = self.transform(image_pil)

            # Add batch dimension: (3, 160, 160) -> (1, 3, 160, 160)
            tensor = tensor.unsqueeze(0)

            return tensor.to(self.device)

        except Exception as e:
            logger.error(f"Image preprocessing error: {str(e)}")
            return None

    def extract_embedding(self, image: Union[np.ndarray, Image.Image]) -> Optional[np.ndarray]:
        """
        Extract 512-dimensional face embedding from a single image crop.

        Args:
            image: Input face image (NumPy array or PIL Image)

        Returns:
            L2-normalized 512D float32 numpy array or None if extraction fails
        """
        try:
            tensor = self.preprocess_image(image)
            if tensor is None:
                return None

            with torch.no_grad():
                embedding_tensor = self.model(tensor)

            # Convert to numpy
            embedding_np = embedding_tensor.squeeze(0).cpu().numpy().astype(np.float32)

            # Numerical stability check
            if not np.all(np.isfinite(embedding_np)):
                logger.error("Extracted embedding contains non-finite values")
                return None

            # Ensure unit length
            return self.normalize_embedding(embedding_np)

        except Exception as e:
            logger.error(f"Embedding extraction error: {str(e)}")
            return None

    def extract_batch_embeddings(self,
                                 images: List[Union[np.ndarray, Image.Image]]) -> List[Optional[np.ndarray]]:
        """
        Extract embeddings from a batch of images for efficiency.

        Args:
            images: List of input face images

        Returns:
            List of 512D embeddings (same order and length as input)
        """
        if not images:
            return []

        results: List[Optional[np.ndarray]] = [None] * len(images)

        try:
            tensors = []
            valid_indices = []

            for i, img in enumerate(images):
                tensor = self.preprocess_image(img)
                if tensor is not None:
                    tensors.append(tensor)
                    valid_indices.append(i)

            if not tensors:
                return results

            # Stack into single batch tensor: (B, 3, 160, 160)
            batch_tensor = torch.cat(tensors, dim=0).to(self.device)

            with torch.no_grad():
                batch_embeddings = self.model(batch_tensor)

            batch_np = batch_embeddings.cpu().numpy().astype(np.float32)

            for idx, valid_idx in enumerate(valid_indices):
                emb = batch_np[idx]
                if np.all(np.isfinite(emb)):
                    results[valid_idx] = self.normalize_embedding(emb)
                else:
                    results[valid_idx] = None

            return results

        except Exception as e:
            logger.error(f"Batch embedding extraction error: {str(e)}")
            return results

    def normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """
        Normalize embedding vector to unit length (L2 norm = 1.0).

        Args:
            embedding: Input embedding vector

        Returns:
            L2-normalized embedding vector
        """
        try:
            norm = np.linalg.norm(embedding)
            if norm == 0 or not np.isfinite(norm):
                return embedding
            return (embedding / norm).astype(np.float32)
        except Exception as e:
            logger.error(f"Embedding normalization error: {str(e)}")
            return embedding

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        Since embeddings are L2-normalized, cosine similarity equals their dot product.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score (-1.0 to 1.0)
        """
        try:
            emb1_norm = self.normalize_embedding(embedding1)
            emb2_norm = self.normalize_embedding(embedding2)
            similarity = np.dot(emb1_norm, emb2_norm)
            return float(np.clip(similarity, -1.0, 1.0))
        except Exception as e:
            logger.error(f"Similarity computation error: {str(e)}")
            return 0.0

    def compute_distance(self,
                         embedding1: np.ndarray,
                         embedding2: np.ndarray,
                         metric: str = 'cosine') -> float:
        """
        Compute distance between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            metric: Distance metric ('cosine', 'euclidean', 'manhattan')

        Returns:
            Distance value (lower means more similar)
        """
        try:
            if metric == 'cosine':
                return float(1.0 - self.compute_similarity(embedding1, embedding2))
            elif metric == 'euclidean':
                return float(np.linalg.norm(embedding1 - embedding2))
            elif metric == 'manhattan':
                return float(np.sum(np.abs(embedding1 - embedding2)))
            else:
                logger.warning(f"Unknown metric: {metric}, defaulting to cosine")
                return float(1.0 - self.compute_similarity(embedding1, embedding2))
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

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get metadata about the current embedding model.

        Returns:
            Dictionary with model metadata and parameter statistics
        """
        return {
            'model_name': self.model_name,
            'architecture': 'InceptionResnetV1',
            'embedding_size': self.embedding_size,
            'pretrained': self.pretrained,
            'device': str(self.device),
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'model_trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }

    def benchmark_inference_time(self, num_samples: int = 20) -> Dict[str, float]:
        """
        Benchmark inference time of the model.

        Args:
            num_samples: Number of benchmark forward passes

        Returns:
            Dictionary with timing statistics in milliseconds and FPS
        """
        import time
        try:
            dummy_input = torch.randn(1, 3, 160, 160).to(self.device)

            # Warm-up passes
            with torch.no_grad():
                for _ in range(3):
                    _ = self.model(dummy_input)

            times = []
            with torch.no_grad():
                for _ in range(num_samples):
                    t0 = time.perf_counter()
                    _ = self.model(dummy_input)
                    times.append((time.perf_counter() - t0) * 1000.0)

            times_arr = np.array(times)
            mean_ms = float(np.mean(times_arr))
            return {
                'mean_time_ms': mean_ms,
                'std_time_ms': float(np.std(times_arr)),
                'min_time_ms': float(np.min(times_arr)),
                'max_time_ms': float(np.max(times_arr)),
                'median_time_ms': float(np.median(times_arr)),
                'fps': float(1000.0 / mean_ms) if mean_ms > 0 else 0.0
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
        Initialize multi-model embedding ensemble.

        Args:
            model_configs: List of configuration dictionaries for EmbeddingExtractor
            device: Target device
        """
        self.device = device
        self.models: List[EmbeddingExtractor] = []

        for config in model_configs:
            try:
                model = EmbeddingExtractor(**config, device=device)
                self.models.append(model)
                logger.info(f"Loaded ensemble sub-model: {config.get('model_name', 'unknown')}")
            except Exception as e:
                logger.error(f"Failed to load ensemble sub-model {config}: {str(e)}")

    def extract_ensemble_embedding(self, image: Union[np.ndarray, Image.Image]) -> Optional[np.ndarray]:
        """
        Extract ensemble embedding by concatenating embeddings from all sub-models.

        Args:
            image: Input face image

        Returns:
            Normalized concatenated embedding vector or None if all models fail
        """
        try:
            embeddings = []
            for model in self.models:
                emb = model.extract_embedding(image)
                if emb is not None:
                    embeddings.append(emb)

            if not embeddings:
                return None

            ensemble_emb = np.concatenate(embeddings)
            norm = np.linalg.norm(ensemble_emb)
            if norm == 0:
                return ensemble_emb
            return (ensemble_emb / norm).astype(np.float32)

        except Exception as e:
            logger.error(f"Ensemble embedding extraction error: {str(e)}")
            return None


if __name__ == "__main__":
    extractor = EmbeddingExtractor(model_name='vggface2', device='cpu')
    model_info = extractor.get_model_info()
    print(f"Model info: {model_info}")

    dummy_image = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
    embedding = extractor.extract_embedding(dummy_image)

    if embedding is not None:
        print(f"Extracted embedding shape: {embedding.shape}")
        print(f"Embedding norm: {np.linalg.norm(embedding)}")
    else:
        print("Failed to extract embedding")
