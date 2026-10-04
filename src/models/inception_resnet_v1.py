"""
InceptionResnetV1 Face Recognition Model Architecture.

Port of the Inception-ResNet-v1 architecture trained for facial recognition
(FaceNet), compatible with PyTorch 2.x and Python 3.10+.
Provides pretrained weights for VGGFace2 and CASIA-Webface datasets.
Outputs 512-dimensional L2-normalized identity embeddings.
"""

import os
import logging
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Pretrained model weight locations
PRETRAINED_URLS = {
    'vggface2': 'https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt',
    'casia-webface': 'https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180408-102900-casia-webface.pt'
}

NUM_CLASSES = {
    'vggface2': 8631,
    'casia-webface': 10575
}


def get_torch_home() -> str:
    """Return the PyTorch cache directory path."""
    return os.path.expanduser(
        os.getenv(
            'TORCH_HOME',
            os.path.join(os.getenv('XDG_CACHE_HOME', '~/.cache'), 'torch')
        )
    )


def get_checkpoint_dirs() -> list:
    """Return candidate directories for cached checkpoints."""
    dirs = []
    try:
        import torch.hub
        dirs.append(os.path.join(torch.hub.get_dir(), 'checkpoints'))
    except Exception:
        pass
    dirs.append(os.path.join(get_torch_home(), 'checkpoints'))
    dirs.append(os.path.join(get_torch_home(), 'hub', 'checkpoints'))
    return dirs


class BasicConv2d(nn.Module):
    """Basic convolutional block with BatchNorm and ReLU."""

    def __init__(self, in_planes: int, out_planes: int, kernel_size, stride, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_planes, out_planes,
            kernel_size=kernel_size, stride=stride,
            padding=padding, bias=False
        )
        self.bn = nn.BatchNorm2d(
            out_planes,
            eps=0.001,
            momentum=0.1,
            affine=True
        )
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Block35(nn.Module):
    """Inception-ResNet Block-35."""

    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = scale
        self.branch0 = BasicConv2d(256, 32, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(
            BasicConv2d(256, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(256, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        )
        self.conv2d = nn.Conv2d(96, 256, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Block17(nn.Module):
    """Inception-ResNet Block-17."""

    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = scale
        self.branch0 = BasicConv2d(896, 128, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(
            BasicConv2d(896, 128, kernel_size=1, stride=1),
            BasicConv2d(128, 128, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            BasicConv2d(128, 128, kernel_size=(7, 1), stride=1, padding=(3, 0))
        )
        self.conv2d = nn.Conv2d(256, 896, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Block8(nn.Module):
    """Inception-ResNet Block-8."""

    def __init__(self, scale: float = 1.0, noReLU: bool = False):
        super().__init__()
        self.scale = scale
        self.noReLU = noReLU
        self.branch0 = BasicConv2d(1792, 192, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(
            BasicConv2d(1792, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            BasicConv2d(192, 192, kernel_size=(3, 1), stride=1, padding=(1, 0))
        )
        self.conv2d = nn.Conv2d(384, 1792, kernel_size=1, stride=1)
        if not self.noReLU:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        if not self.noReLU:
            out = self.relu(out)
        return out


class Mixed_6a(nn.Module):
    """Reduction-A block reducing feature map from 35x35 to 17x17."""

    def __init__(self):
        super().__init__()
        self.branch0 = BasicConv2d(256, 384, kernel_size=3, stride=2)
        self.branch1 = nn.Sequential(
            BasicConv2d(256, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=3, stride=1, padding=1),
            BasicConv2d(192, 256, kernel_size=3, stride=2)
        )
        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        return torch.cat((x0, x1, x2), 1)


class Mixed_7a(nn.Module):
    """Reduction-B block reducing feature map from 17x17 to 8x8."""

    def __init__(self):
        super().__init__()
        self.branch0 = nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2)
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=3, stride=2)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),
            BasicConv2d(256, 256, kernel_size=3, stride=2)
        )
        self.branch3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        return torch.cat((x0, x1, x2, x3), 1)


class InceptionResnetV1(nn.Module):
    """
    Inception Resnet V1 facial recognition model.

    Generates 512-dimensional L2-normalized embeddings for face images.
    Can be initialized with pretrained weights trained on 'vggface2' or 'casia-webface'.
    """

    def __init__(self,
                 pretrained: Optional[str] = None,
                 classify: bool = False,
                 num_classes: Optional[int] = None,
                 dropout_prob: float = 0.6,
                 device: Optional[torch.device] = None,
                 checkpoint_path: Optional[str] = None):
        """
        Initialize InceptionResnetV1.

        Args:
            pretrained: Dataset name ('vggface2' or 'casia-webface') or None.
            classify: If True, outputs logits; if False (default), outputs 512D L2-normalized embeddings.
            num_classes: Number of classification classes if classify=True.
            dropout_prob: Dropout probability before the linear projection layer.
            device: Target device (cpu or cuda).
            checkpoint_path: Optional local path to model weights file.
        """
        super().__init__()

        self.pretrained = pretrained
        self.classify = classify
        self.num_classes = num_classes

        if pretrained is None and self.classify and self.num_classes is None:
            raise ValueError(
                'If "pretrained" is not specified and "classify" is True, "num_classes" must be specified'
            )

        # Setup feature extraction backbone
        self.conv2d_1a = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1)
        self.conv2d_2b = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool_3a = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1, stride=1)
        self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3, stride=1)
        self.conv2d_4b = BasicConv2d(192, 256, kernel_size=3, stride=2)

        self.repeat_1 = nn.Sequential(
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
        )
        self.mixed_6a = Mixed_6a()
        self.repeat_2 = nn.Sequential(
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
        )
        self.mixed_7a = Mixed_7a()
        self.repeat_3 = nn.Sequential(
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
        )
        self.block8 = Block8(noReLU=True)
        self.avgpool_1a = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_prob)
        self.last_linear = nn.Linear(1792, 512, bias=False)
        self.last_bn = nn.BatchNorm1d(512, eps=0.001, momentum=0.1, affine=True)

        if pretrained is not None:
            tmp_classes = NUM_CLASSES.get(pretrained)
            if tmp_classes is None:
                raise ValueError(
                    f"Unknown pretrained dataset '{pretrained}'. "
                    f"Supported options: {list(NUM_CLASSES.keys())}"
                )
            self.logits = nn.Linear(512, tmp_classes)
            load_weights(self, pretrained, checkpoint_path=checkpoint_path)

        if self.classify and self.num_classes is not None:
            self.logits = nn.Linear(512, self.num_classes)

        if device is not None:
            self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Batch of normalized face tensors (B, 3, 160, 160).

        Returns:
            Normalized 512D embeddings (B, 512) or logits (B, num_classes).
        """
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.conv2d_4b(x)
        x = self.repeat_1(x)
        x = self.mixed_6a(x)
        x = self.repeat_2(x)
        x = self.mixed_7a(x)
        x = self.repeat_3(x)
        x = self.block8(x)
        x = self.avgpool_1a(x)
        x = self.dropout(x)
        x = self.last_linear(x.view(x.shape[0], -1))
        x = self.last_bn(x)
        if self.classify:
            x = self.logits(x)
        else:
            x = F.normalize(x, p=2, dim=1)
        return x


def load_weights(model: nn.Module,
                 name: str,
                 checkpoint_path: Optional[str] = None) -> None:
    """
    Load verified pretrained face recognition weights into InceptionResnetV1.

    Args:
        model: InceptionResnetV1 instance to populate.
        name: Pretrained dataset name ('vggface2' or 'casia-webface').
        checkpoint_path: Explicit local path to weights file, if provided.

    Raises:
        ValueError: If name is not recognized.
        RuntimeError: If weights file cannot be found, downloaded, or loaded.
                      Never silently falls back to random weights.
    """
    if name not in PRETRAINED_URLS:
        raise ValueError(
            f"Pretrained weights only available for {list(PRETRAINED_URLS.keys())}, got '{name}'"
        )

    url = PRETRAINED_URLS[name]
    filename = os.path.basename(url)

    target_file = None

    # 1. Check explicit path
    if checkpoint_path is not None:
        if os.path.exists(checkpoint_path):
            target_file = checkpoint_path
        else:
            raise RuntimeError(
                f"Specified checkpoint_path does not exist: {checkpoint_path}"
            )

    # 2. Check local repository models/ directory
    if target_file is None:
        repo_local = os.path.join(os.getcwd(), 'models', filename)
        if os.path.exists(repo_local):
            target_file = repo_local

    # 3. Check candidate cache directories
    if target_file is None:
        for cdir in get_checkpoint_dirs():
            candidate = os.path.join(cdir, filename)
            if os.path.exists(candidate) and os.path.getsize(candidate) > 0:
                target_file = candidate
                break

    # 4. If not found locally, download from official repository
    if target_file is None:
        cache_dir = os.path.join(get_torch_home(), 'checkpoints')
        os.makedirs(cache_dir, exist_ok=True)
        cached_file = os.path.join(cache_dir, filename)
        logger.info(f"Downloading pretrained '{name}' weights from {url} to {cached_file}...")
        try:
            torch.hub.download_url_to_file(url, cached_file, progress=True)
            target_file = cached_file
        except Exception as e:
            # Clean up partial download if it exists
            if os.path.exists(cached_file) and os.path.getsize(cached_file) == 0:
                os.remove(cached_file)
            raise RuntimeError(
                f"Failed to download pretrained face-recognition weights for '{name}' from {url}: {e}. "
                "Explicit error raised: random weights will not be substituted."
            ) from e

    # 5. Load state dict into model
    try:
        logger.info(f"Loading pretrained '{name}' weights from {target_file}")
        state_dict = torch.load(target_file, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict)
    except Exception as e:
        # Retry with weights_only=False if PyTorch serialization format requires it
        try:
            state_dict = torch.load(target_file, map_location='cpu', weights_only=False)
            model.load_state_dict(state_dict)
        except Exception as inner_e:
            raise RuntimeError(
                f"Failed to load state_dict from {target_file}: {inner_e}. "
                "Model weights are corrupted or incompatible. Random weights will not be substituted."
            ) from inner_e
