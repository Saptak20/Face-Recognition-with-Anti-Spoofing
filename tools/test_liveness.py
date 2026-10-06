#!/usr/bin/env python3
"""
Liveness Validation Utility for Silent-Face-Anti-Spoofing Ensemble

Development-only script to test the MiniFASNet ensemble liveness model
against real images. Not for production use.

Official Silent-Face-Anti-Spoofing ensemble (from test.py):
- 2.7_80x80_MiniFASNetV2.pth (scale=2.7, MiniFASNetV2)
- 4_0_0_80x80_MiniFASNetV1SE.pth (scale=4.0, MiniFASNetV1SE)

Class mapping (from official test.py):
- Class 0: Spoof (print attack)
- Class 1: Live / Real face
- Class 2: Spoof (replay/screen attack) or Unknown

Decision rule (from official test.py):
- Sum predictions from all models in ensemble
- label = argmax(summed_predictions)
- is_live = (label == 1)
- Score = summed_prediction[0][label] / num_models
"""

import sys
import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.liveness_detection import SilentFaceLivenessDetector, CropImage


def load_image(image_path: str) -> np.ndarray:
    """Load image and convert to RGB."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def main():
    parser = argparse.ArgumentParser(description='Test Silent-Face ensemble liveness model on images')
    parser.add_argument('images', nargs='+', help='Path(s) to test image(s)')
    parser.add_argument('--model-dir', default='/tmp', 
                        help='Directory containing checkpoint files')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'],
                        help='Device to run inference on')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Decision threshold for live score')
    parser.add_argument('--save-crops', action='store_true',
                        help='Save face crops for each model scale')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Silent-Face-Anti-Spoofing Ensemble Liveness Test")
    print("=" * 60)
    
    # Initialize detector
    # Use first image's directory as model dir, or default
    model_dir = args.model_dir
    detector = SilentFaceLivenessDetector(
        model_type='SilentFaceEnsemble',
        model_path=str(Path(args.images[0]).parent / "2.7_80x80_MiniFASNetV2.pth"),
        device=args.device,
        input_size=80,
        threshold=args.threshold
    )
    
    info = detector.get_model_info()
    print(f"\nModel: {info['model_variant']}")
    print(f"Ensemble size: {info['ensemble_size']}")
    print(f"Weights loaded: {info['weights_loaded']}")
    print(f"Threshold: {info['threshold']}")
    
    for m in info['models']:
        status = "✓" if m['loaded'] else "✗"
        print(f"  {status} {m['name']} (scale={m['scale']})")
        if m['error']:
            print(f"      Error: {m['error']}")
    
    if not info['weights_loaded']:
        print("\n⚠ WARNING: Not all ensemble models loaded!")
        print("Results may not be meaningful.")
    
    print("\n" + "=" * 60)
    
    for img_path in args.images:
        print(f"\nTesting: {img_path}")
        print("-" * 40)
        
        try:
            image = load_image(img_path)
            result = detector.detect_liveness(image)
            
            # Print results
            label = "LIVE" if result['is_live'] else "SPOOF"
            print(f"Prediction:     {label}")
            print(f"Live score:     {result['live_score']:.4f}")
            print(f"Spoof score:    {result['fake_score']:.4f}")
            print(f"Confidence:     {result['confidence']:.4f}")
            print(f"is_live:        {result['is_live']}")
            
            if result.get('ensemble_details'):
                print(f"\nPer-model details:")
                for detail in result['ensemble_details']:
                    class_names = {0: "SPOOF(print)", 1: "LIVE", 2: "SPOOF(replay)"}
                    pred_class = class_names.get(detail['predicted_class'], "UNKNOWN")
                    probs_str = ', '.join([f"{p:.4f}" for p in detail['probs']])
                    print(f"  {detail['model']} (scale={detail['scale']}):")
                    print(f"    probs: [{probs_str}]")
                    print(f"    predicted: {pred_class}")
            
            # Save crops if requested
            if args.save_crops and result.get('ensemble_details'):
                # Re-run with crop saving (simplified)
                pass
                
        except Exception as e:
            print(f"Error: {e}")


if __name__ == '__main__':
    main()