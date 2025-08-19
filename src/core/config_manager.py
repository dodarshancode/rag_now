"""
Configuration Manager for OpenSCENARIO 2.0 RAG System
Handles YAML configuration loading with environment variable support.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ConfigManager:
    """Manages configuration for the OpenSCENARIO RAG system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file. If None, uses default.
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "openscenario_config.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file with environment variable substitution."""
        try:
            with open(self.config_path, 'r') as f:
                config_content = f.read()
            
            # Replace environment variables
            config_content = self._substitute_env_vars(config_content)
            
            config = yaml.safe_load(config_content)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
            
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
            raise
    
    def _substitute_env_vars(self, content: str) -> str:
        """Substitute environment variables in configuration content."""
        import re
        
        def replace_env_var(match):
            var_name = match.group(1)
            default_value = match.group(2) if match.group(2) else ""
            return os.getenv(var_name, default_value)
        
        # Pattern: ${VAR_NAME:default_value} or ${VAR_NAME}
        pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'
        return re.sub(pattern, replace_env_var, content)
    
    def _validate_config(self):
        """Validate critical configuration parameters."""
        required_sections = [
            'system', 'hardware', 'models', 'embedding', 
            'data', 'vector_store', 'retrieval', 'generation'
        ]
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate GPU configuration
        gpu_count = self.config['hardware']['gpu_count']
        if gpu_count <= 0:
            raise ValueError("GPU count must be positive")
        
        # Validate model paths
        models_dir = Path(self.config['data']['models_dir'])
        if not models_dir.exists():
            logger.warning(f"Models directory does not exist: {models_dir}")
        
        logger.info("Configuration validation passed")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to configuration value (e.g., 'models.codellama.name')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any):
        """Set configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to configuration value
            value: Value to set
        """
        keys = key_path.split('.')
        config_ref = self.config
        
        for key in keys[:-1]:
            if key not in config_ref:
                config_ref[key] = {}
            config_ref = config_ref[key]
        
        config_ref[keys[-1]] = value
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get configuration for a specific model.
        
        Args:
            model_name: Name of the model ('codellama' or 'open_model')
            
        Returns:
            Model configuration dictionary
        """
        model_config = self.get(f'models.{model_name}')
        if not model_config:
            raise ValueError(f"Model configuration not found: {model_name}")
        
        return model_config
    
    def get_data_paths(self) -> Dict[str, Path]:
        """Get all data paths as Path objects.
        
        Returns:
            Dictionary of data paths
        """
        data_config = self.config['data']
        paths = {}
        
        for key, path_str in data_config.items():
            paths[key] = Path(path_str)
        
        return paths
    
    def ensure_directories(self):
        """Ensure all required directories exist."""
        paths = self.get_data_paths()
        
        directories_to_create = [
            'processed_chunks_dir',
            'embeddings_cache_dir',
            'models_dir'
        ]
        
        for dir_key in directories_to_create:
            if dir_key in paths:
                paths[dir_key].mkdir(parents=True, exist_ok=True)
                logger.info(f"Ensured directory exists: {paths[dir_key]}")
        
        # Create cache directory
        cache_dir = Path(self.get('cache.disk_cache_dir', 'data/cache'))
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logs directory
        log_file = Path(self.get('monitoring.log_file', 'logs/openscenario_rag.log'))
        log_file.parent.mkdir(parents=True, exist_ok=True)
    
    def get_gpu_devices(self) -> list:
        """Get list of GPU device IDs to use.
        
        Returns:
            List of GPU device IDs
        """
        gpu_count = self.get('hardware.gpu_count', 1)
        return list(range(gpu_count))
    
    def get_generation_params(self) -> Dict[str, Any]:
        """Get generation parameters for LLM inference.
        
        Returns:
            Dictionary of generation parameters
        """
        return {
            'max_new_tokens': self.get('generation.max_tokens', 1024),
            'min_new_tokens': self.get('generation.min_tokens', 50),
            'temperature': self.get('generation.temperature', 0.2),
            'top_p': self.get('generation.top_p', 0.9),
            'repetition_penalty': self.get('generation.repetition_penalty', 1.1),
            'do_sample': True,
            'pad_token_id': None,  # Will be set by model
        }
    
    def get_retrieval_params(self) -> Dict[str, Any]:
        """Get retrieval parameters.
        
        Returns:
            Dictionary of retrieval parameters
        """
        return {
            'top_k_dense': self.get('retrieval.top_k_dense', 20),
            'top_k_sparse': self.get('retrieval.top_k_sparse', 15),
            'top_k_final': self.get('retrieval.top_k_final', 10),
            'dense_weight': self.get('retrieval.dense_weight', 0.7),
            'sparse_weight': self.get('retrieval.sparse_weight', 0.3),
            'tag_boost_factor': self.get('retrieval.tag_boost_factor', 1.5),
            'rerank': self.get('retrieval.rerank', True),
        }
    
    def save_config(self, output_path: Optional[str] = None):
        """Save current configuration to file.
        
        Args:
            output_path: Path to save configuration. If None, uses original path.
        """
        if output_path is None:
            output_path = self.config_path
        
        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration saved to {output_path}")
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access to configuration."""
        return self.config[key]
    
    def __setitem__(self, key: str, value: Any):
        """Allow dictionary-style setting of configuration."""
        self.config[key] = value
