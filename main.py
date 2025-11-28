"""
Main Entry Point

Main application entry point for the Face Recognition System.
Initializes all components and starts the FastAPI server.
"""

import sys
import logging
import asyncio
import signal
from pathlib import Path
from typing import Optional
import argparse

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from src.config import ConfigManager
    from src.face_capture import FaceCapture
    from src.embedding_extraction import EmbeddingExtractor
    from src.liveness_detection import LivenessDetector
    from src.deepfake_detection import DeepfakeDetector
    from src.database_manager import DatabaseManager
    from src.authentication import AuthenticationEngine
    from src.api import FaceRecognitionAPI
    from src.utils import PerformanceUtils, FileUtils
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaceRecognitionSystem:
    """
    Main face recognition system orchestrator.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize face recognition system.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config_manager = None
        self.config = None
        
        # System components
        self.face_capture = None
        self.embedding_extractor = None
        self.liveness_detector = None
        self.deepfake_detector = None
        self.database_manager = None
        self.auth_engine = None
        self.api = None
        
        # Performance monitoring
        self.perf_utils = PerformanceUtils()
        
        # System state
        self.is_running = False
        self.initialization_complete = False
        
        logger.info("FaceRecognitionSystem initialized")
    
    def initialize(self) -> bool:
        """
        Initialize all system components.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            logger.info("Starting system initialization...")
            self.perf_utils.start_timer('initialization')
            
            # Load configuration
            if not self._load_configuration():
                logger.error("Failed to load configuration")
                return False
            
            # Setup logging
            self._setup_logging()
            
            # Create necessary directories
            self._create_directories()
            
            # Initialize components in order
            if not self._initialize_database():
                logger.error("Failed to initialize database")
                return False
            
            if not self._initialize_models():
                logger.error("Failed to initialize models")
                return False
            
            if not self._initialize_authentication():
                logger.error("Failed to initialize authentication")
                return False
            
            if not self._initialize_api():
                logger.error("Failed to initialize API")
                return False
            
            # Mark initialization as complete
            self.initialization_complete = True
            init_time = self.perf_utils.end_timer('initialization')
            
            logger.info(f"System initialization completed in {init_time:.2f} seconds")
            return True
            
        except Exception as e:
            logger.error(f"System initialization error: {str(e)}")
            return False
    
    def _load_configuration(self) -> bool:
        """
        Load system configuration.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.config_manager = ConfigManager(self.config_path)
            self.config = self.config_manager.get_config()
            
            # Validate configuration
            validation = self.config_manager.validate_config()
            if not validation['valid']:
                logger.error(f"Invalid configuration: {validation['errors']}")
                return False
            
            if validation['warnings']:
                for warning in validation['warnings']:
                    logger.warning(f"Configuration warning: {warning}")
            
            logger.info(f"Configuration loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Configuration loading error: {str(e)}")
            return False
    
    def _setup_logging(self) -> None:
        """Setup logging based on configuration."""
        try:
            log_config = self.config.logging
            
            # Set logging level
            log_level = getattr(logging, log_config.level.upper(), logging.INFO)
            logging.getLogger().setLevel(log_level)
            
            # Create logs directory
            if log_config.file_enabled:
                FileUtils.ensure_directory(Path(log_config.file_path).parent)
                
                # Add file handler
                file_handler = logging.FileHandler(log_config.file_path)
                file_handler.setLevel(log_level)
                
                # Set format
                formatter = logging.Formatter(log_config.format)
                file_handler.setFormatter(formatter)
                
                logging.getLogger().addHandler(file_handler)
                
                logger.info(f"File logging enabled: {log_config.file_path}")
            
        except Exception as e:
            logger.error(f"Logging setup error: {str(e)}")
    
    def _create_directories(self) -> None:
        """Create necessary directories."""
        try:
            directories = [
                self.config.data_dir,
                self.config.models_dir,
                self.config.logs_dir,
                Path(self.config.database.sqlite_db_path).parent,
                Path(self.config.database.faiss_index_path).parent,
                self.config.database.backup_path
            ]
            
            for directory in directories:
                FileUtils.ensure_directory(directory)
            
            logger.info("Directories created successfully")
            
        except Exception as e:
            logger.error(f"Directory creation error: {str(e)}")
    
    def _initialize_database(self) -> bool:
        """
        Initialize database manager.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Initializing database manager...")
            
            self.database_manager = DatabaseManager(
                db_path=self.config.database.sqlite_db_path,
                faiss_index_path=self.config.database.faiss_index_path,
                embedding_dim=self.config.models.embedding_dim,
                index_type=self.config.database.faiss_index_type
            )
            
            # Get initial statistics
            stats = self.database_manager.get_statistics()
            logger.info(f"Database initialized - Users: {stats.get('total_users', 0)}, "
                       f"Embeddings: {stats.get('total_embeddings', 0)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Database initialization error: {str(e)}")
            return False
    
    def _initialize_models(self) -> bool:
        """
        Initialize all ML models.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Initializing ML models...")
            
            # Initialize face capture
            logger.info("Loading face capture system...")
            self.face_capture = FaceCapture(
                device=self.config.models.device,
                image_size=self.config.models.face_image_size,
                margin=self.config.models.face_margin,
                min_face_size=self.config.models.min_face_size
            )
            
            # Initialize embedding extractor
            logger.info("Loading embedding extraction model...")
            self.embedding_extractor = EmbeddingExtractor(
                model_name=self.config.models.embedding_model,
                device=self.config.models.device,
                embedding_size=self.config.models.embedding_dim
            )
            
            # Initialize liveness detector
            logger.info("Loading liveness detection model...")
            self.liveness_detector = LivenessDetector(
                model_type=self.config.models.liveness_model_type,
                device=self.config.models.device,
                input_size=self.config.models.liveness_input_size,
                threshold=self.config.models.liveness_threshold
            )
            
            # Initialize deepfake detector
            logger.info("Loading deepfake detection model...")
            self.deepfake_detector = DeepfakeDetector(
                model_name=self.config.models.deepfake_model,
                device=self.config.models.device,
                threshold=self.config.models.deepfake_threshold,
                image_size=self.config.models.deepfake_image_size
            )
            
            logger.info("All ML models initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Model initialization error: {str(e)}")
            return False
    
    def _initialize_authentication(self) -> bool:
        """
        Initialize authentication engine.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Initializing authentication engine...")
            
            # Create authentication configuration
            auth_config = {
                'face_similarity_threshold': self.config.authentication.face_similarity_threshold,
                'liveness_threshold': self.config.models.liveness_threshold,
                'deepfake_threshold': self.config.models.deepfake_threshold,
                'overall_confidence_threshold': self.config.authentication.overall_confidence_threshold,
                'enable_mfa': self.config.authentication.enable_mfa,
                'max_attempts_per_hour': self.config.authentication.max_attempts_per_hour,
                'confidence_weights': {
                    'face_similarity': self.config.authentication.face_similarity_weight,
                    'liveness': self.config.authentication.liveness_weight,
                    'deepfake': self.config.authentication.deepfake_weight
                }
            }
            
            self.auth_engine = AuthenticationEngine(
                face_capture=self.face_capture,
                embedding_extractor=self.embedding_extractor,
                liveness_detector=self.liveness_detector,
                deepfake_detector=self.deepfake_detector,
                database_manager=self.database_manager,
                config=auth_config
            )
            
            logger.info("Authentication engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Authentication initialization error: {str(e)}")
            return False
    
    def _initialize_api(self) -> bool:
        """
        Initialize FastAPI application.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Initializing API server...")
            
            # Create API configuration
            api_config = {
                'allowed_origins': self.config.api.allowed_origins,
                'api_key_required': self.config.api.api_key_required
            }
            
            self.api = FaceRecognitionAPI(api_config)
            
            # Set system components
            self.api.set_components(
                face_capture=self.face_capture,
                embedding_extractor=self.embedding_extractor,
                liveness_detector=self.liveness_detector,
                deepfake_detector=self.deepfake_detector,
                database_manager=self.database_manager,
                auth_engine=self.auth_engine
            )
            
            logger.info("API server initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"API initialization error: {str(e)}")
            return False
    
    def run(self) -> None:
        """Run the face recognition system."""
        try:
            if not self.initialization_complete:
                logger.error("System not properly initialized")
                return
            
            logger.info("Starting Face Recognition System...")
            self.is_running = True
            
            # Setup signal handlers for graceful shutdown
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            # Start API server
            self.api.run(
                host=self.config.api.host,
                port=self.config.api.port,
                debug=self.config.api.debug
            )
            
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        except Exception as e:
            logger.error(f"System runtime error: {str(e)}")
        finally:
            self.shutdown()
    
    def _signal_handler(self, signum, frame):
        """Handle system signals for graceful shutdown."""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self.shutdown()
    
    def shutdown(self) -> None:
        """Shutdown the system gracefully."""
        try:
            logger.info("Shutting down Face Recognition System...")
            self.is_running = False
            
            # Close database connections
            if self.database_manager:
                self.database_manager.close()
                logger.info("Database connections closed")
            
            # Clean up temporary files
            temp_dirs = [
                Path("temp"),
                Path("cache")
            ]
            
            for temp_dir in temp_dirs:
                if temp_dir.exists():
                    FileUtils.clean_old_files(temp_dir, max_age_days=0)
            
            logger.info("System shutdown completed")
            
        except Exception as e:
            logger.error(f"Shutdown error: {str(e)}")
    
    def get_system_status(self) -> dict:
        """
        Get current system status.
        
        Returns:
            System status dictionary
        """
        try:
            status = {
                'initialized': self.initialization_complete,
                'running': self.is_running,
                'components': {
                    'face_capture': self.face_capture is not None,
                    'embedding_extractor': self.embedding_extractor is not None,
                    'liveness_detector': self.liveness_detector is not None,
                    'deepfake_detector': self.deepfake_detector is not None,
                    'database_manager': self.database_manager is not None,
                    'auth_engine': self.auth_engine is not None,
                    'api': self.api is not None
                }
            }
            
            # Add database statistics if available
            if self.database_manager:
                status['database_stats'] = self.database_manager.get_statistics()
            
            # Add performance metrics
            status['performance'] = PerformanceUtils.measure_memory_usage()
            
            return status
            
        except Exception as e:
            logger.error(f"Status check error: {str(e)}")
            return {'error': str(e)}


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Face Recognition System')
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='API server host address'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='API server port'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    
    parser.add_argument(
        '--create-sample-config',
        action='store_true',
        help='Create sample configuration file and exit'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    try:
        args = parse_arguments()
        
        # Create sample configuration if requested
        if args.create_sample_config:
            from src.config import create_sample_config
            if create_sample_config("config.yaml"):
                print("Sample configuration created: config.yaml")
                print("Please edit the configuration file and run the system again.")
            else:
                print("Failed to create sample configuration")
            return
        
        # Initialize system
        system = FaceRecognitionSystem(config_path=args.config)
        
        if not system.initialize():
            logger.error("System initialization failed")
            sys.exit(1)
        
        # Override config with command line arguments
        if args.host:
            system.config.api.host = args.host
        if args.port:
            system.config.api.port = args.port
        if args.debug:
            system.config.api.debug = True
        
        # Print system status
        status = system.get_system_status()
        logger.info(f"System status: {status}")
        
        # Run system
        system.run()
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Main execution error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()