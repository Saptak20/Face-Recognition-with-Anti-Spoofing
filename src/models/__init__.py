"""
Models package for Face Recognition with Anti-Spoofing
"""

from src.models.mini_fas_net import (
    MiniFASNetV1,
    MiniFASNetV2,
    MiniFASNetV1SE,
    MiniFASNetV2SE,
    MODEL_VARIANTS,
    PRETRAINED_URLS,
    MODEL_FILENAME_TO_VARIANT,
)

__all__ = [
    'MiniFASNetV1',
    'MiniFASNetV2',
    'MiniFASNetV1SE',
    'MiniFASNetV2SE',
    'MODEL_VARIANTS',
    'PRETRAINED_URLS',
    'MODEL_FILENAME_TO_VARIANT',
]