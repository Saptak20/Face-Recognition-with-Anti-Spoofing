"""
Configuration Module

Centralized configuration management for the face recognition system
using JSON and YAML files with environment variable support.
"""

import json
import yaml
import os
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, asdict
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for ML models."""
    # Face detection and capture
    face_detection_confidence: float = 0.9
    face_image_size: int = 160
    face_margin: int = 32
    min_face_size: int = 20
    
    # Embedding extraction
    embedding_model: str = 'vggface2'
    embedding_dim: int = 512
    
    # Liveness detection
    liveness_model_type: str = 'custom_cnn'
    liveness_input_size: int = 64
    liveness_threshold: float = 0.5
    
    # Deepfake detection
    deepfake_model: str = 'google/vit-base-patch16-224'
    deepfake_image_size: int = 224
    deepfake_threshold: float = 0.5
    
    # Device configuration
    device: str = 'cpu'  # 'cpu' or 'cuda'


@dataclass
class DatabaseConfig:
    """Configuration for database systems."""
    # SQLite database
    sqlite_db_path: str = "data/face_recognition.db"
    
    # FAISS index
    faiss_index_path: str = "data/embeddings/face_index.faiss"
    faiss_index_type: str = 'IndexFlatIP'
    
    # Backup configuration
    backup_enabled: bool = True
    backup_interval_hours: int = 24
    backup_path: str = "data/backups"
    max_backups: int = 7


@dataclass
class AuthenticationConfig:
    """Configuration for authentication system."""
    # Similarity thresholds
    face_similarity_threshold: float = 0.7
    overall_confidence_threshold: float = 0.6
    
    # Confidence score weights
    face_similarity_weight: float = 0.5
    liveness_weight: float = 0.3
    deepfake_weight: float = 0.2
    
    # Rate limiting
    max_attempts_per_hour: int = 5
    
    # Multi-factor authentication
    enable_mfa: bool = False
    otp_length: int = 6
    otp_expiry_minutes: int = 5
    
    # Email configuration for MFA
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    sender_email: str = ""
    sender_password: str = ""


@dataclass
class APIConfig:
    """Configuration for FastAPI server."""
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    workers: int = 1
    
    # CORS settings
    allowed_origins: list = None
    allow_credentials: bool = True
    
    # Security
    api_key_required: bool = False
    api_key: str = ""
    jwt_secret_key: str = "your-secret-key-change-this"
    
    # Request limits
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    request_timeout: int = 30  # seconds
    
    def __post_init__(self):
        if self.allowed_origins is None:
            self.allowed_origins = ["*"]


@dataclass
class LoggingConfig:
    """Configuration for logging system."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # File logging
    file_enabled: bool = True
    file_path: str = "logs/face_recognition.log"
    file_max_size_mb: int = 10
    file_backup_count: int = 5
    
    # Console logging
    console_enabled: bool = True
    
    # Log rotation
    rotation_enabled: bool = True


@dataclass
class SystemConfig:
    """Main system configuration container."""
    models: ModelConfig
    database: DatabaseConfig
    authentication: AuthenticationConfig
    api: APIConfig
    logging: LoggingConfig
    
    # System paths
    project_root: str = "."
    data_dir: str = "data"
    models_dir: str = "models"
    logs_dir: str = "logs"
    config_dir: str = "config"
    
    # Environment
    environment: str = "development"  # development, staging, production
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)


class ConfigManager:
    """
    Configuration manager for loading and managing system configuration.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config: Optional[SystemConfig] = None
        self._env_prefix = "FACE_RECOGNITION_"
        
        # Load configuration
        self.load_config()
        
        logger.info(f"ConfigManager initialized with config from: {config_path}")
    
    def load_config(self) -> SystemConfig:
        """
        Load configuration from file and environment variables.
        
        Returns:
            System configuration object
        """
        try:
            # Start with default configuration
            config_dict = self._get_default_config()
            
            # Load from file if provided
            if self.config_path and Path(self.config_path).exists():
                file_config = self._load_config_file(self.config_path)
                config_dict = self._merge_configs(config_dict, file_config)
                logger.info(f"Loaded configuration from file: {self.config_path}")
            
            # Override with environment variables
            env_config = self._load_env_config()
            config_dict = self._merge_configs(config_dict, env_config)
            
            # Create configuration objects
            self.config = SystemConfig(
                models=ModelConfig(**config_dict.get('models', {})),
                database=DatabaseConfig(**config_dict.get('database', {})),
                authentication=AuthenticationConfig(**config_dict.get('authentication', {})),
                api=APIConfig(**config_dict.get('api', {})),
                logging=LoggingConfig(**config_dict.get('logging', {})),
                project_root=config_dict.get('project_root', '.'),
                data_dir=config_dict.get('data_dir', 'data'),
                models_dir=config_dict.get('models_dir', 'models'),
                logs_dir=config_dict.get('logs_dir', 'logs'),
                config_dir=config_dict.get('config_dir', 'config'),
                environment=config_dict.get('environment', 'development')
            )
            
            # Create directories
            self._create_directories()
            
            return self.config
            
        except Exception as e:
            logger.error(f"Configuration loading error: {str(e)}")
            # Return default configuration on error
            return self._get_default_system_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration dictionary."""
        return {
            'models': {},
            'database': {},
            'authentication': {},
            'api': {},
            'logging': {},
            'project_root': '.',
            'data_dir': 'data',
            'models_dir': 'models',
            'logs_dir': 'logs',
            'config_dir': 'config',
            'environment': 'development'
        }
    
    def _get_default_system_config(self) -> SystemConfig:
        """Get default system configuration object."""
        return SystemConfig(
            models=ModelConfig(),
            database=DatabaseConfig(),
            authentication=AuthenticationConfig(),
            api=APIConfig(),
            logging=LoggingConfig()
        )
    
    def _load_config_file(self, file_path: str) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            file_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        try:
            file_path = Path(file_path)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix.lower() in ['.yaml', '.yml']:
                    return yaml.safe_load(f) or {}
                elif file_path.suffix.lower() == '.json':
                    return json.load(f) or {}
                else:
                    logger.warning(f"Unsupported config file format: {file_path.suffix}")
                    return {}
                    
        except Exception as e:
            logger.error(f"Failed to load config file {file_path}: {str(e)}")
            return {}
    
    def _load_env_config(self) -> Dict[str, Any]:
        """
        Load configuration from environment variables.
        
        Returns:
            Configuration dictionary from environment variables
        """
        try:
            env_config = {}
            
            # Define environment variable mappings
            env_mappings = {
                # Models
                f"{self._env_prefix}DEVICE": ["models", "device"],
                f"{self._env_prefix}EMBEDDING_MODEL": ["models", "embedding_model"],
                f"{self._env_prefix}EMBEDDING_DIM": ["models", "embedding_dim"],
                f"{self._env_prefix}LIVENESS_THRESHOLD": ["models", "liveness_threshold"],
                f"{self._env_prefix}DEEPFAKE_THRESHOLD": ["models", "deepfake_threshold"],
                
                # Database
                f"{self._env_prefix}DB_PATH": ["database", "sqlite_db_path"],
                f"{self._env_prefix}FAISS_INDEX_PATH": ["database", "faiss_index_path"],
                
                # Authentication
                f"{self._env_prefix}FACE_SIMILARITY_THRESHOLD": ["authentication", "face_similarity_threshold"],
                f"{self._env_prefix}ENABLE_MFA": ["authentication", "enable_mfa"],
                f"{self._env_prefix}SENDER_EMAIL": ["authentication", "sender_email"],
                f"{self._env_prefix}SENDER_PASSWORD": ["authentication", "sender_password"],
                
                # API
                f"{self._env_prefix}HOST": ["api", "host"],
                f"{self._env_prefix}PORT": ["api", "port"],
                f"{self._env_prefix}DEBUG": ["api", "debug"],
                f"{self._env_prefix}API_KEY": ["api", "api_key"],
                f"{self._env_prefix}JWT_SECRET": ["api", "jwt_secret_key"],
                
                # Logging
                f"{self._env_prefix}LOG_LEVEL": ["logging", "level"],
                f"{self._env_prefix}LOG_FILE": ["logging", "file_path"],
                
                # System
                f"{self._env_prefix}ENVIRONMENT": ["environment"],
                f"{self._env_prefix}DATA_DIR": ["data_dir"],
            }
            
            for env_var, path in env_mappings.items():
                value = os.getenv(env_var)
                if value is not None:
                    # Type conversion
                    if env_var.endswith(('_THRESHOLD', '_DIM', '_SIZE', '_PORT', '_LENGTH', '_MINUTES', '_HOURS', '_COUNT')):
                        try:
                            value = float(value) if '.' in value else int(value)
                        except ValueError:
                            continue
                    elif env_var.endswith(('_DEBUG', '_ENABLED', '_REQUIRED', '_MFA')):
                        value = value.lower() in ('true', '1', 'yes', 'on')
                    
                    # Set nested value
                    self._set_nested_value(env_config, path, value)
            
            return env_config
            
        except Exception as e:
            logger.error(f"Failed to load environment config: {str(e)}")
            return {}
    
    def _set_nested_value(self, config: Dict[str, Any], path: list, value: Any) -> None:
        """
        Set nested dictionary value using path list.
        
        Args:
            config: Configuration dictionary
            path: List of keys for nested access
            value: Value to set
        """
        try:
            current = config
            for key in path[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            current[path[-1]] = value
        except Exception as e:
            logger.error(f"Failed to set nested value {path}: {str(e)}")
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge two configuration dictionaries.
        
        Args:
            base: Base configuration
            override: Override configuration
            
        Returns:
            Merged configuration
        """
        try:
            result = base.copy()
            
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = self._merge_configs(result[key], value)
                else:
                    result[key] = value
            
            return result
            
        except Exception as e:
            logger.error(f"Config merge error: {str(e)}")
            return base
    
    def _create_directories(self) -> None:
        """Create necessary directories based on configuration."""
        try:
            if self.config:
                directories = [
                    self.config.data_dir,
                    self.config.models_dir,
                    self.config.logs_dir,
                    self.config.config_dir,
                    Path(self.config.database.sqlite_db_path).parent,
                    Path(self.config.database.faiss_index_path).parent,
                    self.config.database.backup_path
                ]
                
                for directory in directories:
                    Path(directory).mkdir(parents=True, exist_ok=True)
                
                logger.info("Created necessary directories")
                
        except Exception as e:
            logger.error(f"Directory creation error: {str(e)}")
    
    def save_config(self, file_path: str, format: str = 'yaml') -> bool:
        """
        Save current configuration to file.
        
        Args:
            file_path: Path to save configuration
            format: File format ('yaml' or 'json')
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.config:
                logger.error("No configuration loaded to save")
                return False
            
            config_dict = self.config.to_dict()
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                if format.lower() == 'yaml':
                    yaml.safe_dump(config_dict, f, default_flow_style=False, indent=2)
                elif format.lower() == 'json':
                    json.dump(config_dict, f, indent=2, default=str)
                else:
                    logger.error(f"Unsupported format: {format}")
                    return False
            
            logger.info(f"Configuration saved to: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Configuration save error: {str(e)}")
            return False
    
    def get_config(self) -> SystemConfig:
        """
        Get current system configuration.
        
        Returns:
            System configuration object
        """
        if self.config is None:
            self.config = self.load_config()
        
        return self.config
    
    def reload_config(self) -> SystemConfig:
        """
        Reload configuration from sources.
        
        Returns:
            Reloaded system configuration
        """
        logger.info("Reloading configuration")
        return self.load_config()
    
    def validate_config(self) -> Dict[str, Any]:
        """
        Validate current configuration.
        
        Returns:
            Validation results
        """
        try:
            if not self.config:
                return {"valid": False, "errors": ["No configuration loaded"]}
            
            errors = []
            warnings = []
            
            # Validate model configuration
            if self.config.models.embedding_dim not in [128, 512]:
                warnings.append("Embedding dimension should typically be 128 or 512")
            
            # Validate thresholds (should be between 0 and 1)
            thresholds = [
                self.config.models.liveness_threshold,
                self.config.models.deepfake_threshold,
                self.config.authentication.face_similarity_threshold,
                self.config.authentication.overall_confidence_threshold
            ]
            
            for threshold in thresholds:
                if not (0.0 <= threshold <= 1.0):
                    errors.append(f"Threshold {threshold} should be between 0.0 and 1.0")
            
            # Validate weights sum to 1
            weight_sum = (self.config.authentication.face_similarity_weight +
                         self.config.authentication.liveness_weight +
                         self.config.authentication.deepfake_weight)
            
            if abs(weight_sum - 1.0) > 0.01:
                warnings.append(f"Confidence weights sum to {weight_sum:.3f}, should sum to 1.0")
            
            # Validate paths exist
            if not Path(self.config.data_dir).exists():
                warnings.append(f"Data directory does not exist: {self.config.data_dir}")
            
            # Validate API configuration
            if self.config.api.port < 1 or self.config.api.port > 65535:
                errors.append(f"Invalid port number: {self.config.api.port}")
            
            return {
                "valid": len(errors) == 0,
                "errors": errors,
                "warnings": warnings
            }
            
        except Exception as e:
            logger.error(f"Configuration validation error: {str(e)}")
            return {"valid": False, "errors": [str(e)]}


def create_sample_config(file_path: str = "config/config.yaml") -> bool:
    """
    Create a sample configuration file.
    
    Args:
        file_path: Path to create sample configuration
        
    Returns:
        True if successful, False otherwise
    """
    try:
        config_manager = ConfigManager()
        default_config = config_manager._get_default_system_config()
        config_manager.config = default_config
        
        return config_manager.save_config(file_path, format='yaml')
        
    except Exception as e:
        logger.error(f"Sample config creation error: {str(e)}")
        return False


# Example usage and testing
if __name__ == "__main__":
    # Test configuration manager
    config_manager = ConfigManager()
    
    # Get configuration
    config = config_manager.get_config()
    print(f"Configuration environment: {config.environment}")
    print(f"API host:port = {config.api.host}:{config.api.port}")
    print(f"Database path: {config.database.sqlite_db_path}")
    print(f"Device: {config.models.device}")
    
    # Validate configuration
    validation = config_manager.validate_config()
    print(f"Configuration valid: {validation['valid']}")
    if validation['errors']:
        print(f"Errors: {validation['errors']}")
    if validation['warnings']:
        print(f"Warnings: {validation['warnings']}")
    
    # Create sample configuration file
    sample_created = create_sample_config("sample_config.yaml")
    print(f"Sample config created: {sample_created}")
    
    print("Configuration module test completed")
