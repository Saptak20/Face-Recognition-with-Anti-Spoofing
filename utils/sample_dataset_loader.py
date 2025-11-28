"""
Sample Dataset Loader

Utilities for loading and managing sample face datasets for testing
and demonstration of the face recognition system.
"""

import os
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
import logging
from dataclasses import dataclass
from PIL import Image
import requests
from urllib.parse import urlparse
import zipfile
import shutil


@dataclass
class PersonSample:
    """Data class for a person's sample data."""
    person_id: str
    name: str
    images: List[str]  # List of image file paths
    metadata: Dict


class SampleDatasetLoader:
    """
    Dataset loader for sample face recognition datasets.
    
    Supports loading from:
    - Local directories
    - Sample datasets (LFW subset, custom samples)
    - Generated synthetic data for testing
    """
    
    def __init__(self, dataset_dir: str = "data/samples"):
        """
        Initialize the dataset loader.
        
        Args:
            dataset_dir: Directory to store/load datasets
        """
        self.dataset_dir = Path(dataset_dir)
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        # Sample dataset URLs (for demonstration)
        self.sample_datasets = {
            'demo_faces': {
                'url': 'https://github.com/example/demo-faces/archive/main.zip',
                'description': 'Small demo dataset with 10 people, 5 images each',
                'size': '~2MB'
            }
        }
    
    def create_sample_dataset(self, num_people: int = 5, images_per_person: int = 3) -> Dict:
        """
        Create a sample dataset with generated faces or stock images.
        
        Args:
            num_people: Number of people to include
            images_per_person: Number of images per person
            
        Returns:
            Dictionary with dataset information
        """
        self.logger.info(f"Creating sample dataset with {num_people} people, {images_per_person} images each")
        
        sample_dir = self.dataset_dir / "sample_generated"
        sample_dir.mkdir(exist_ok=True)
        
        dataset_info = {
            'name': 'Generated Sample Dataset',
            'description': f'Sample dataset with {num_people} people',
            'people': [],
            'total_images': 0,
            'created_at': str(Path().cwd())
        }
        
        # Generate sample people data
        for person_idx in range(num_people):
            person_id = f"person_{person_idx:03d}"
            person_name = f"Sample Person {person_idx + 1}"
            
            person_dir = sample_dir / person_id
            person_dir.mkdir(exist_ok=True)
            
            person_images = []
            
            # Generate sample images (colored rectangles with text)
            for img_idx in range(images_per_person):
                img_filename = f"{person_id}_img_{img_idx:02d}.png"
                img_path = person_dir / img_filename
                
                # Create a simple colored image with person ID
                self._create_sample_face_image(str(img_path), person_name, img_idx)
                person_images.append(str(img_path))
            
            person_data = {
                'person_id': person_id,
                'name': person_name,
                'images': person_images,
                'image_count': len(person_images)
            }
            
            dataset_info['people'].append(person_data)
            dataset_info['total_images'] += len(person_images)
        
        # Save dataset metadata
        metadata_path = sample_dir / "dataset_info.json"
        with open(metadata_path, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        self.logger.info(f"Sample dataset created at {sample_dir}")
        return dataset_info
    
    def _create_sample_face_image(self, image_path: str, person_name: str, variant: int):
        """
        Create a sample face image (colored rectangle with text).
        
        Args:
            image_path: Path to save the image
            person_name: Name to display on image
            variant: Variant number for different colors
        """
        # Create a colored image
        colors = [
            (255, 100, 100),  # Red
            (100, 255, 100),  # Green
            (100, 100, 255),  # Blue
            (255, 255, 100),  # Yellow
            (255, 100, 255),  # Magenta
        ]
        
        color = colors[variant % len(colors)]
        
        # Create image
        img = np.full((160, 160, 3), color, dtype=np.uint8)
        
        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_lines = [
            person_name.split()[-1],  # Last name
            f"Var {variant + 1}"
        ]
        
        y_offset = 60
        for line in text_lines:
            text_size = cv2.getTextSize(line, font, 0.6, 2)[0]
            x = (160 - text_size[0]) // 2
            cv2.putText(img, line, (x, y_offset), font, 0.6, (0, 0, 0), 2)
            y_offset += 30
        
        # Add simple "face-like" features
        # Eyes
        cv2.circle(img, (50, 50), 8, (0, 0, 0), -1)
        cv2.circle(img, (110, 50), 8, (0, 0, 0), -1)
        
        # Nose
        cv2.circle(img, (80, 75), 3, (0, 0, 0), -1)
        
        # Mouth
        cv2.ellipse(img, (80, 100), (15, 8), 0, 0, 180, (0, 0, 0), 2)
        
        # Save image
        cv2.imwrite(image_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    def load_from_directory(self, directory: str) -> List[PersonSample]:
        """
        Load dataset from a directory structure.
        
        Expected structure:
        directory/
        ├── person_001/
        │   ├── image1.jpg
        │   ├── image2.jpg
        │   └── ...
        ├── person_002/
        │   └── ...
        
        Args:
            directory: Root directory path
            
        Returns:
            List of PersonSample objects
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        self.logger.info(f"Loading dataset from {directory}")
        
        people = []
        
        # Iterate through person directories
        for person_dir in directory.iterdir():
            if not person_dir.is_dir():
                continue
            
            person_id = person_dir.name
            
            # Find all image files
            image_files = []
            for file_path in person_dir.iterdir():
                if file_path.suffix.lower() in self.supported_formats:
                    image_files.append(str(file_path))
            
            if not image_files:
                self.logger.warning(f"No images found for person {person_id}")
                continue
            
            # Create person sample
            person_sample = PersonSample(
                person_id=person_id,
                name=person_id.replace('_', ' ').title(),
                images=sorted(image_files),
                metadata={'image_count': len(image_files)}
            )
            
            people.append(person_sample)
        
        self.logger.info(f"Loaded {len(people)} people with {sum(len(p.images) for p in people)} total images")
        return people
    
    def download_sample_dataset(self, dataset_name: str) -> Optional[str]:
        """
        Download a sample dataset from the internet.
        
        Args:
            dataset_name: Name of the dataset to download
            
        Returns:
            Path to downloaded dataset directory, or None if failed
        """
        if dataset_name not in self.sample_datasets:
            self.logger.error(f"Unknown dataset: {dataset_name}")
            return None
        
        dataset_info = self.sample_datasets[dataset_name]
        url = dataset_info['url']
        
        self.logger.info(f"Downloading dataset '{dataset_name}' from {url}")
        
        # Create download directory
        download_dir = self.dataset_dir / dataset_name
        download_dir.mkdir(exist_ok=True)
        
        try:
            # Download file
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Save to temporary zip file
            zip_path = download_dir / "dataset.zip"
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Extract zip file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(download_dir)
            
            # Remove zip file
            zip_path.unlink()
            
            self.logger.info(f"Dataset downloaded and extracted to {download_dir}")
            return str(download_dir)
            
        except Exception as e:
            self.logger.error(f"Failed to download dataset: {e}")
            return None
    
    def validate_images(self, person_samples: List[PersonSample]) -> Dict:
        """
        Validate all images in the dataset.
        
        Args:
            person_samples: List of person samples to validate
            
        Returns:
            Validation report
        """
        report = {
            'total_people': len(person_samples),
            'total_images': 0,
            'valid_images': 0,
            'invalid_images': [],
            'resolution_stats': {},
            'format_stats': {}
        }
        
        for person in person_samples:
            for image_path in person.images:
                report['total_images'] += 1
                
                try:
                    # Try to load image
                    img = cv2.imread(image_path)
                    if img is None:
                        report['invalid_images'].append({
                            'path': image_path,
                            'person': person.person_id,
                            'error': 'Cannot load image'
                        })
                        continue
                    
                    # Check resolution
                    height, width = img.shape[:2]
                    resolution = f"{width}x{height}"
                    report['resolution_stats'][resolution] = report['resolution_stats'].get(resolution, 0) + 1
                    
                    # Check format
                    file_ext = Path(image_path).suffix.lower()
                    report['format_stats'][file_ext] = report['format_stats'].get(file_ext, 0) + 1
                    
                    report['valid_images'] += 1
                    
                except Exception as e:
                    report['invalid_images'].append({
                        'path': image_path,
                        'person': person.person_id,
                        'error': str(e)
                    })
        
        return report
    
    def create_train_test_split(self, person_samples: List[PersonSample], 
                               test_ratio: float = 0.3) -> Tuple[List[PersonSample], List[PersonSample]]:
        """
        Split dataset into training and testing sets.
        
        Args:
            person_samples: List of person samples
            test_ratio: Ratio of images to use for testing
            
        Returns:
            Tuple of (train_samples, test_samples)
        """
        train_samples = []
        test_samples = []
        
        for person in person_samples:
            images = person.images.copy()
            
            # Shuffle images
            np.random.shuffle(images)
            
            # Calculate split point
            num_test = max(1, int(len(images) * test_ratio))
            num_train = len(images) - num_test
            
            if num_train > 0:
                train_person = PersonSample(
                    person_id=person.person_id,
                    name=person.name,
                    images=images[:num_train],
                    metadata={**person.metadata, 'split': 'train'}
                )
                train_samples.append(train_person)
            
            if num_test > 0:
                test_person = PersonSample(
                    person_id=person.person_id,
                    name=person.name,
                    images=images[num_train:],
                    metadata={**person.metadata, 'split': 'test'}
                )
                test_samples.append(test_person)
        
        return train_samples, test_samples
    
    def export_dataset_info(self, person_samples: List[PersonSample], 
                           output_path: str) -> None:
        """
        Export dataset information to JSON file.
        
        Args:
            person_samples: List of person samples
            output_path: Path to save the JSON file
        """
        dataset_info = {
            'total_people': len(person_samples),
            'total_images': sum(len(p.images) for p in person_samples),
            'people': []
        }
        
        for person in person_samples:
            person_info = {
                'person_id': person.person_id,
                'name': person.name,
                'image_count': len(person.images),
                'images': person.images,
                'metadata': person.metadata
            }
            dataset_info['people'].append(person_info)
        
        with open(output_path, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        self.logger.info(f"Dataset info exported to {output_path}")
    
    def get_sample_images(self, person_samples: List[PersonSample], 
                         max_per_person: int = 1) -> List[Tuple[str, str]]:
        """
        Get sample images for demonstration.
        
        Args:
            person_samples: List of person samples
            max_per_person: Maximum images per person
            
        Returns:
            List of (image_path, person_name) tuples
        """
        sample_images = []
        
        for person in person_samples:
            selected_images = person.images[:max_per_person]
            for image_path in selected_images:
                sample_images.append((image_path, person.name))
        
        return sample_images
    
    def create_demo_registration_data(self, person_samples: List[PersonSample]) -> List[Dict]:
        """
        Create demo registration data for API testing.
        
        Args:
            person_samples: List of person samples
            
        Returns:
            List of registration data dictionaries
        """
        demo_data = []
        
        for person in person_samples:
            if person.images:
                demo_entry = {
                    'user_id': person.person_id,
                    'name': person.name,
                    'email': f"{person.person_id}@example.com",
                    'department': 'Demo',
                    'image_path': person.images[0],  # Use first image for registration
                    'additional_images': person.images[1:] if len(person.images) > 1 else []
                }
                demo_data.append(demo_entry)
        
        return demo_data


def main():
    """Demo script for dataset loader."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create dataset loader
    loader = SampleDatasetLoader()
    
    # Create sample dataset
    print("Creating sample dataset...")
    dataset_info = loader.create_sample_dataset(num_people=5, images_per_person=3)
    print(f"Created dataset with {dataset_info['total_images']} images")
    
    # Load the created dataset
    print("\nLoading dataset...")
    sample_dir = loader.dataset_dir / "sample_generated"
    person_samples = loader.load_from_directory(sample_dir)
    
    # Validate images
    print("\nValidating images...")
    validation_report = loader.validate_images(person_samples)
    print(f"Validation: {validation_report['valid_images']}/{validation_report['total_images']} images valid")
    
    # Create train/test split
    print("\nCreating train/test split...")
    train_samples, test_samples = loader.create_train_test_split(person_samples, test_ratio=0.3)
    print(f"Train: {len(train_samples)} people, Test: {len(test_samples)} people")
    
    # Export dataset info
    info_path = sample_dir / "dataset_complete_info.json"
    loader.export_dataset_info(person_samples, str(info_path))
    print(f"Dataset info exported to {info_path}")
    
    # Create demo registration data
    demo_data = loader.create_demo_registration_data(person_samples)
    demo_path = sample_dir / "demo_registration_data.json"
    with open(demo_path, 'w') as f:
        json.dump(demo_data, f, indent=2)
    print(f"Demo registration data saved to {demo_path}")
    
    print("\nDataset preparation complete!")


if __name__ == "__main__":
    main()