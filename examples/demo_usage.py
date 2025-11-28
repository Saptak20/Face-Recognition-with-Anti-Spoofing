"""
Demo Usage Examples

This script demonstrates how to use the face recognition system
with sample data and common use cases.
"""

import sys
import os
import asyncio
import logging
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))
sys.path.append(str(Path(__file__).parent))

# Import the face recognition system components
try:
    from src.main import FaceRecognitionSystem
    from src.config import ConfigManager
    from utils.sample_dataset_loader import SampleDatasetLoader
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all dependencies are installed and paths are correct")
    sys.exit(1)


class FaceRecognitionDemo:
    """
    Demonstration class for the face recognition system.
    
    Shows how to:
    - Initialize the system
    - Register users
    - Perform authentication
    - Use sample datasets
    - Handle different scenarios
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the demo.
        
        Args:
            config_path: Path to configuration file
        """
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        try:
            config_manager = ConfigManager()
            self.config = config_manager.load_config(config_path)
        except Exception as e:
            self.logger.warning(f"Could not load config from {config_path}: {e}")
            self.config = self._get_default_config()
        
        # Initialize system
        self.system = None
        self.dataset_loader = SampleDatasetLoader()
        
        # Demo data
        self.demo_users = []
    
    def _get_default_config(self) -> Dict:
        """Get default configuration for demo."""
        return {
            'database': {
                'database_path': 'data/demo_face_recognition.db',
                'index_path': 'data/demo_face_index.faiss',
                'embedding_dimension': 512,
                'similarity_threshold': 0.7,
                'use_faiss': False  # Use fallback for demo
            },
            'face_capture': {
                'min_face_size': 40,
                'scale_factor': 1.1,
                'min_neighbors': 5,
                'webcam_index': 0,
                'capture_timeout': 10
            },
            'liveness_detection': {
                'model_type': 'mobilenet',
                'threshold': 0.5,
                'enable_eye_blink': True,
                'enable_head_movement': True
            },
            'deepfake_detection': {
                'model_name': 'facebook/deit-tiny-patch16-224',
                'threshold': 0.5,
                'enable_spatial_analysis': True
            },
            'authentication': {
                'face_similarity_threshold': 0.7,
                'liveness_threshold': 0.5,
                'deepfake_threshold': 0.5,
                'overall_confidence_threshold': 0.6,
                'enable_mfa': False,
                'max_attempts_per_hour': 10
            },
            'api': {
                'host': '0.0.0.0',
                'port': 8000,
                'enable_cors': True
            }
        }
    
    async def initialize_system(self):
        """Initialize the face recognition system."""
        try:
            self.logger.info("Initializing face recognition system...")
            
            # Create system instance
            self.system = FaceRecognitionSystem(self.config)
            
            # Initialize system components
            await self.system.initialize()
            
            self.logger.info("System initialized successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize system: {e}")
            return False
    
    def prepare_sample_data(self):
        """Prepare sample dataset for demo."""
        self.logger.info("Preparing sample dataset...")
        
        # Create sample dataset
        dataset_info = self.dataset_loader.create_sample_dataset(
            num_people=5, 
            images_per_person=3
        )
        
        # Load the created dataset
        sample_dir = self.dataset_loader.dataset_dir / "sample_generated"
        person_samples = self.dataset_loader.load_from_directory(sample_dir)
        
        # Create demo registration data
        self.demo_users = self.dataset_loader.create_demo_registration_data(person_samples)
        
        self.logger.info(f"Prepared {len(self.demo_users)} demo users")
        return self.demo_users
    
    async def demo_user_registration(self):
        """Demonstrate user registration process."""
        self.logger.info("=== DEMO: User Registration ===")
        
        if not self.demo_users:
            self.prepare_sample_data()
        
        registered_users = []
        
        for user_data in self.demo_users[:3]:  # Register first 3 users
            try:
                self.logger.info(f"Registering user: {user_data['name']}")
                
                # Load image from file
                image_path = user_data['image_path']
                if not os.path.exists(image_path):
                    self.logger.error(f"Image not found: {image_path}")
                    continue
                
                # Register user using image file
                result = await self._register_user_from_image(
                    user_id=user_data['user_id'],
                    name=user_data['name'],
                    email=user_data['email'],
                    image_path=image_path
                )
                
                if result['success']:
                    self.logger.info(f"✅ Successfully registered {user_data['name']}")
                    registered_users.append(user_data)
                    
                    # Add additional embeddings if available
                    for additional_image in user_data['additional_images'][:2]:
                        if os.path.exists(additional_image):
                            await self._add_user_embedding(user_data['user_id'], additional_image)
                
                else:
                    self.logger.error(f"❌ Failed to register {user_data['name']}: {result.get('message', 'Unknown error')}")
            
            except Exception as e:
                self.logger.error(f"Error registering {user_data['name']}: {e}")
        
        self.logger.info(f"Registration complete. {len(registered_users)} users registered.")
        return registered_users
    
    async def _register_user_from_image(self, user_id: str, name: str, 
                                       email: str, image_path: str) -> Dict:
        """Register user from image file."""
        try:
            # Load and preprocess image
            image = cv2.imread(image_path)
            if image is None:
                return {'success': False, 'message': 'Could not load image'}
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Use the authentication engine to register user
            # Note: This simulates the registration process with file input
            result = self.system.auth_engine.register_user(
                user_id=user_id,
                name=name,
                email=email,
                use_webcam=False,  # Don't use webcam
                image_data=image   # Provide image directly
            )
            
            return result
            
        except Exception as e:
            return {'success': False, 'message': str(e)}
    
    async def _add_user_embedding(self, user_id: str, image_path: str):
        """Add additional embedding for user."""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return False
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Extract face
            faces = self.system.face_capture.detect_faces(image)
            if not faces:
                return False
            
            # Extract embedding
            face_image = self.system.face_capture.extract_face(image, faces[0])
            embedding = self.system.embedding_extractor.extract_embedding(face_image)
            
            # Add to database
            embedding_id = self.system.database_manager.add_embedding(user_id, embedding)
            
            if embedding_id:
                self.logger.info(f"Added additional embedding for {user_id}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error adding embedding for {user_id}: {e}")
            return False
    
    async def demo_authentication(self):
        """Demonstrate authentication process."""
        self.logger.info("=== DEMO: User Authentication ===")
        
        if not self.demo_users:
            self.logger.error("No demo users available. Run registration demo first.")
            return
        
        # Test authentication with known users
        for user_data in self.demo_users[:3]:
            try:
                self.logger.info(f"Testing authentication for: {user_data['name']}")
                
                # Get a test image for this user
                test_image_path = user_data['additional_images'][0] if user_data['additional_images'] else user_data['image_path']
                
                if not os.path.exists(test_image_path):
                    self.logger.error(f"Test image not found: {test_image_path}")
                    continue
                
                # Perform authentication
                result = await self._authenticate_from_image(test_image_path)
                
                if result['success']:
                    authenticated_user = result['user_id']
                    confidence = result['confidence']
                    self.logger.info(f"✅ Authentication successful!")
                    self.logger.info(f"   Identified as: {authenticated_user}")
                    self.logger.info(f"   Confidence: {confidence:.3f}")
                    
                    # Check if identification is correct
                    if authenticated_user == user_data['user_id']:
                        self.logger.info(f"   ✅ Correct identification!")
                    else:
                        self.logger.warning(f"   ⚠️ Incorrect identification (expected {user_data['user_id']})")
                
                else:
                    self.logger.error(f"❌ Authentication failed: {result.get('message', 'Unknown error')}")
            
            except Exception as e:
                self.logger.error(f"Error during authentication for {user_data['name']}: {e}")
        
        # Test with unknown user (should fail)
        self.logger.info("Testing with unknown user...")
        
        # Create a random test image
        unknown_image = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
        result = await self._authenticate_from_array(unknown_image)
        
        if result['success']:
            self.logger.warning(f"⚠️ Unknown user was authenticated as {result['user_id']} (unexpected)")
        else:
            self.logger.info(f"✅ Unknown user correctly rejected: {result.get('message', 'No match found')}")
    
    async def _authenticate_from_image(self, image_path: str) -> Dict:
        """Authenticate user from image file."""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return {'success': False, 'message': 'Could not load image'}
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            return await self._authenticate_from_array(image)
            
        except Exception as e:
            return {'success': False, 'message': str(e)}
    
    async def _authenticate_from_array(self, image: np.ndarray) -> Dict:
        """Authenticate user from image array."""
        try:
            # Use authentication engine with image data
            result = self.system.auth_engine.authenticate_user(
                use_webcam=False,
                image_data=image
            )
            
            return result
            
        except Exception as e:
            return {'success': False, 'message': str(e)}
    
    async def demo_security_features(self):
        """Demonstrate security features."""
        self.logger.info("=== DEMO: Security Features ===")
        
        # Test liveness detection
        self.logger.info("Testing liveness detection...")
        
        # Create a test image (static)
        test_image = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
        
        try:
            liveness_result = self.system.liveness_detector.comprehensive_liveness_check(test_image)
            liveness_score = liveness_result.get('combined_score', 0)
            
            self.logger.info(f"Liveness score: {liveness_score:.3f}")
            
            if liveness_score > self.system.auth_engine.liveness_threshold:
                self.logger.info("✅ Liveness check passed")
            else:
                self.logger.info("❌ Liveness check failed (expected for static image)")
        
        except Exception as e:
            self.logger.error(f"Liveness detection error: {e}")
        
        # Test deepfake detection
        self.logger.info("Testing deepfake detection...")
        
        try:
            deepfake_result = self.system.deepfake_detector.comprehensive_deepfake_analysis(test_image)
            deepfake_score = deepfake_result.get('combined_fake_score', 0)
            
            self.logger.info(f"Deepfake score: {deepfake_score:.3f}")
            
            if deepfake_score < self.system.auth_engine.deepfake_threshold:
                self.logger.info("✅ Deepfake check passed")
            else:
                self.logger.info("❌ Deepfake detected")
        
        except Exception as e:
            self.logger.error(f"Deepfake detection error: {e}")
    
    async def demo_database_operations(self):
        """Demonstrate database operations."""
        self.logger.info("=== DEMO: Database Operations ===")
        
        try:
            # Get user statistics
            if self.demo_users:
                user_id = self.demo_users[0]['user_id']
                
                user_info = self.system.database_manager.get_user(user_id)
                if user_info:
                    self.logger.info(f"User info for {user_id}:")
                    self.logger.info(f"  Name: {user_info['name']}")
                    self.logger.info(f"  Email: {user_info.get('email', 'N/A')}")
                    self.logger.info(f"  Active: {user_info['is_active']}")
                
                # Get embeddings
                embeddings = self.system.database_manager.get_user_embeddings(user_id)
                self.logger.info(f"  Embeddings: {len(embeddings)}")
                
                # Get user stats
                stats = self.system.database_manager.get_user_stats(user_id)
                if stats:
                    self.logger.info(f"  Total authentications: {stats.get('total_authentications', 0)}")
                    self.logger.info(f"  Success rate: {stats.get('successful_authentications', 0)}/{stats.get('total_authentications', 0)}")
        
        except Exception as e:
            self.logger.error(f"Database operations error: {e}")
    
    async def demo_api_simulation(self):
        """Simulate API operations."""
        self.logger.info("=== DEMO: API Simulation ===")
        
        # This would demonstrate API calls if the API server was running
        # For now, we'll just show the structure
        
        api_examples = {
            'registration': {
                'endpoint': 'POST /register',
                'data': {
                    'user_id': 'demo_user_001',
                    'name': 'Demo User',
                    'email': 'demo@example.com'
                },
                'files': {'image': 'user_photo.jpg'}
            },
            'authentication': {
                'endpoint': 'POST /authenticate',
                'files': {'image': 'auth_photo.jpg'}
            },
            'user_info': {
                'endpoint': 'GET /users/{user_id}'
            },
            'user_stats': {
                'endpoint': 'GET /users/{user_id}/stats'
            }
        }
        
        for operation, details in api_examples.items():
            self.logger.info(f"{operation.title()} API:")
            self.logger.info(f"  Endpoint: {details['endpoint']}")
            if 'data' in details:
                self.logger.info(f"  Data: {details['data']}")
            if 'files' in details:
                self.logger.info(f"  Files: {details['files']}")
    
    async def run_full_demo(self):
        """Run the complete demonstration."""
        self.logger.info("🚀 Starting Face Recognition System Demo")
        self.logger.info("=" * 50)
        
        try:
            # Initialize system
            if not await self.initialize_system():
                self.logger.error("Failed to initialize system. Demo cannot continue.")
                return
            
            # Prepare sample data
            self.prepare_sample_data()
            
            # Run demo components
            await self.demo_user_registration()
            await self.demo_authentication()
            await self.demo_security_features()
            await self.demo_database_operations()
            await self.demo_api_simulation()
            
            self.logger.info("=" * 50)
            self.logger.info("🎉 Demo completed successfully!")
            
        except Exception as e:
            self.logger.error(f"Demo failed: {e}")
        
        finally:
            # Cleanup
            if self.system:
                await self.system.shutdown()


async def main():
    """Main demo function."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and run demo
    demo = FaceRecognitionDemo()
    await demo.run_full_demo()


def quick_test():
    """Quick test without full demo."""
    print("Face Recognition System - Quick Test")
    print("=" * 40)
    
    # Test imports
    try:
        from src.face_capture import FaceCapture
        print("✅ Face capture module imported")
    except ImportError as e:
        print(f"❌ Face capture import failed: {e}")
    
    try:
        from src.embedding_extraction import EmbeddingExtractor
        print("✅ Embedding extraction module imported")
    except ImportError as e:
        print(f"❌ Embedding extraction import failed: {e}")
    
    try:
        from src.database_manager import DatabaseManager
        print("✅ Database manager module imported")
    except ImportError as e:
        print(f"❌ Database manager import failed: {e}")
    
    try:
        from src.authentication import AuthenticationEngine
        print("✅ Authentication engine module imported")
    except ImportError as e:
        print(f"❌ Authentication engine import failed: {e}")
    
    # Test sample dataset loader
    try:
        from utils.sample_dataset_loader import SampleDatasetLoader
        loader = SampleDatasetLoader()
        print("✅ Sample dataset loader created")
    except ImportError as e:
        print(f"❌ Sample dataset loader failed: {e}")
    
    print("\nQuick test completed!")
    print("Run 'python examples/demo_usage.py' for full demo")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Face Recognition System Demo")
    parser.add_argument('--quick', action='store_true', help='Run quick test only')
    args = parser.parse_args()
    
    if args.quick:
        quick_test()
    else:
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            print("\nDemo interrupted by user")
        except Exception as e:
            print(f"Demo error: {e}")