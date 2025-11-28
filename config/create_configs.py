"""
Sample Configuration Files

This module provides sample configuration files in different formats
for easy setup and customization of the face recognition system.
"""

import yaml
import json
from pathlib import Path


# Basic configuration template
BASIC_CONFIG = {
    'system': {
        'name': 'Face Recognition System',
        'version': '1.0.0',
        'description': 'Production-grade face recognition with anti-spoofing',
        'debug': False,
        'log_level': 'INFO'
    },
    
    'database': {
        'database_path': 'data/face_recognition.db',
        'index_path': 'data/face_index.faiss',
        'embedding_dimension': 512,
        'similarity_threshold': 0.7,
        'use_faiss': True,
        'auto_backup': True,
        'backup_interval_hours': 24
    },
    
    'face_capture': {
        'min_face_size': 40,
        'scale_factor': 1.1,
        'min_neighbors': 5,
        'confidence_threshold': 0.9,
        'webcam_index': 0,
        'capture_timeout': 10,
        'image_size': [160, 160],
        'normalize': True
    },
    
    'embedding_extraction': {
        'model_name': 'vggface2',
        'batch_size': 32,
        'device': 'auto',  # auto, cpu, cuda
        'normalize_embeddings': True,
        'embedding_dimension': 512
    },
    
    'liveness_detection': {
        'model_type': 'mobilenet',  # mobilenet, custom_cnn
        'threshold': 0.5,
        'enable_eye_blink': True,
        'enable_head_movement': True,
        'enable_texture_analysis': True,
        'temporal_frames': 5
    },
    
    'deepfake_detection': {
        'model_name': 'facebook/deit-tiny-patch16-224',
        'threshold': 0.5,
        'enable_spatial_analysis': True,
        'enable_temporal_analysis': False,
        'confidence_threshold': 0.8
    },
    
    'authentication': {
        'face_similarity_threshold': 0.7,
        'liveness_threshold': 0.5,
        'deepfake_threshold': 0.5,
        'overall_confidence_threshold': 0.6,
        'enable_mfa': False,
        'max_attempts_per_hour': 10,
        'confidence_weights': {
            'face_similarity': 0.5,
            'liveness': 0.3,
            'deepfake': 0.2
        }
    },
    
    'mfa': {
        'otp_length': 6,
        'otp_expiry_minutes': 5,
        'max_attempts': 3,
        'email_enabled': True,
        'sms_enabled': False
    },
    
    'api': {
        'host': '0.0.0.0',
        'port': 8000,
        'workers': 1,
        'enable_cors': True,
        'cors_origins': ['*'],
        'enable_rate_limiting': True,
        'rate_limit_calls': 100,
        'rate_limit_period': 3600,
        'enable_authentication': True,
        'jwt_secret_key': 'your-secret-key-here',
        'jwt_expiry_hours': 24,
        'max_file_size': 10485760,  # 10MB
        'allowed_file_types': ['image/jpeg', 'image/png', 'image/jpg']
    },
    
    'email': {
        'smtp_server': 'smtp.gmail.com',
        'smtp_port': 587,
        'username': '',
        'password': '',
        'sender_name': 'Face Recognition System',
        'sender_email': ''
    },
    
    'logging': {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file_path': 'logs/face_recognition.log',
        'max_size_mb': 10,
        'backup_count': 5,
        'console_output': True
    },
    
    'security': {
        'admin_api_key': 'your-admin-key-here',
        'enable_ip_whitelist': False,
        'allowed_ips': ['127.0.0.1', '192.168.1.0/24'],
        'enable_request_signing': False,
        'max_concurrent_requests': 100
    },
    
    'performance': {
        'use_gpu': False,
        'gpu_device_id': 0,
        'enable_caching': True,
        'cache_size': 1000,
        'enable_compression': True,
        'max_workers': 4
    },
    
    'monitoring': {
        'enable_metrics': True,
        'metrics_interval': 60,
        'enable_health_checks': True,
        'health_check_interval': 30
    }
}


# Development configuration
DEV_CONFIG = {
    **BASIC_CONFIG,
    'system': {
        **BASIC_CONFIG['system'],
        'debug': True,
        'log_level': 'DEBUG'
    },
    'database': {
        **BASIC_CONFIG['database'],
        'database_path': 'data/dev_face_recognition.db',
        'index_path': 'data/dev_face_index.faiss',
        'use_faiss': False  # Use fallback for development
    },
    'api': {
        **BASIC_CONFIG['api'],
        'enable_authentication': False,  # Disable auth for development
        'workers': 1,
        'cors_origins': ['http://localhost:3000', 'http://localhost:8080']
    },
    'logging': {
        **BASIC_CONFIG['logging'],
        'level': 'DEBUG',
        'file_path': 'logs/dev_face_recognition.log'
    },
    'security': {
        **BASIC_CONFIG['security'],
        'enable_ip_whitelist': False
    }
}


# Production configuration
PROD_CONFIG = {
    **BASIC_CONFIG,
    'system': {
        **BASIC_CONFIG['system'],
        'debug': False,
        'log_level': 'WARNING'
    },
    'database': {
        **BASIC_CONFIG['database'],
        'use_faiss': True,
        'auto_backup': True,
        'backup_interval_hours': 6  # More frequent backups
    },
    'authentication': {
        **BASIC_CONFIG['authentication'],
        'enable_mfa': True,
        'max_attempts_per_hour': 5,  # Stricter rate limiting
        'overall_confidence_threshold': 0.7  # Higher threshold
    },
    'api': {
        **BASIC_CONFIG['api'],
        'workers': 4,  # More workers for production
        'enable_authentication': True,
        'cors_origins': ['https://your-frontend-domain.com'],  # Specific origins only
        'rate_limit_calls': 50,  # Stricter rate limiting
        'max_file_size': 5242880  # 5MB limit for production
    },
    'security': {
        **BASIC_CONFIG['security'],
        'enable_ip_whitelist': True,
        'enable_request_signing': True,
        'max_concurrent_requests': 50
    },
    'performance': {
        **BASIC_CONFIG['performance'],
        'use_gpu': True,
        'enable_caching': True,
        'max_workers': 8
    },
    'monitoring': {
        **BASIC_CONFIG['monitoring'],
        'enable_metrics': True,
        'metrics_interval': 30,
        'enable_health_checks': True
    }
}


# Testing configuration
TEST_CONFIG = {
    **BASIC_CONFIG,
    'system': {
        **BASIC_CONFIG['system'],
        'debug': True,
        'log_level': 'DEBUG'
    },
    'database': {
        **BASIC_CONFIG['database'],
        'database_path': 'data/test_face_recognition.db',
        'index_path': 'data/test_face_index.faiss',
        'use_faiss': False,  # Use fallback for testing
        'auto_backup': False
    },
    'authentication': {
        **BASIC_CONFIG['authentication'],
        'enable_mfa': False,
        'max_attempts_per_hour': 1000,  # No rate limiting for tests
        'overall_confidence_threshold': 0.5  # Lower threshold for testing
    },
    'api': {
        **BASIC_CONFIG['api'],
        'port': 8001,  # Different port for testing
        'enable_authentication': False,
        'enable_rate_limiting': False
    },
    'logging': {
        **BASIC_CONFIG['logging'],
        'level': 'DEBUG',
        'file_path': 'logs/test_face_recognition.log',
        'console_output': False  # Reduce test output
    },
    'performance': {
        **BASIC_CONFIG['performance'],
        'use_gpu': False,
        'enable_caching': False,
        'max_workers': 1
    }
}


def save_config_files():
    """Save all configuration files."""
    config_dir = Path('config')
    config_dir.mkdir(exist_ok=True)
    
    configs = {
        'config.yaml': BASIC_CONFIG,
        'dev_config.yaml': DEV_CONFIG,
        'prod_config.yaml': PROD_CONFIG,
        'test_config.yaml': TEST_CONFIG
    }
    
    for filename, config in configs.items():
        # Save YAML format
        yaml_path = config_dir / filename
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False, indent=2)
        
        # Save JSON format
        json_filename = filename.replace('.yaml', '.json')
        json_path = config_dir / json_filename
        with open(json_path, 'w') as f:
            json.dump(config, f, indent=2, sort_keys=False)
        
        print(f"Created {yaml_path} and {json_path}")


def create_docker_config():
    """Create Docker-specific configuration."""
    docker_config = {
        **BASIC_CONFIG,
        'database': {
            **BASIC_CONFIG['database'],
            'database_path': '/app/data/face_recognition.db',
            'index_path': '/app/data/face_index.faiss'
        },
        'api': {
            **BASIC_CONFIG['api'],
            'host': '0.0.0.0',
            'port': 8000,
            'workers': 2
        },
        'logging': {
            **BASIC_CONFIG['logging'],
            'file_path': '/app/logs/face_recognition.log',
            'console_output': True  # Important for Docker logs
        }
    }
    
    config_dir = Path('config')
    config_dir.mkdir(exist_ok=True)
    
    docker_yaml_path = config_dir / 'docker_config.yaml'
    with open(docker_yaml_path, 'w') as f:
        yaml.dump(docker_config, f, default_flow_style=False, sort_keys=False, indent=2)
    
    print(f"Created {docker_yaml_path}")


def create_minimal_config():
    """Create minimal configuration for quick setup."""
    minimal_config = {
        'database': {
            'database_path': 'data/face_recognition.db',
            'use_faiss': False
        },
        'authentication': {
            'face_similarity_threshold': 0.7,
            'enable_mfa': False
        },
        'api': {
            'port': 8000,
            'enable_authentication': False
        },
        'logging': {
            'level': 'INFO',
            'console_output': True
        }
    }
    
    config_dir = Path('config')
    config_dir.mkdir(exist_ok=True)
    
    minimal_yaml_path = config_dir / 'minimal_config.yaml'
    with open(minimal_yaml_path, 'w') as f:
        yaml.dump(minimal_config, f, default_flow_style=False, sort_keys=False, indent=2)
    
    print(f"Created {minimal_yaml_path}")


def main():
    """Generate all configuration files."""
    print("Generating configuration files...")
    
    save_config_files()
    create_docker_config()
    create_minimal_config()
    
    print("\nConfiguration files created successfully!")
    print("\nAvailable configurations:")
    print("- config.yaml / config.json - Basic configuration")
    print("- dev_config.yaml / dev_config.json - Development settings")
    print("- prod_config.yaml / prod_config.json - Production settings")
    print("- test_config.yaml / test_config.json - Testing settings")
    print("- docker_config.yaml - Docker deployment settings")
    print("- minimal_config.yaml - Minimal setup")
    print("\nCopy .env.template to .env and customize as needed.")


if __name__ == "__main__":
    main()