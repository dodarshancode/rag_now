"""
Multi-Model Inference System for OpenSCENARIO 2.0 RAG
Supports CodeLlama-13B and open models with 8x A100 GPU optimization.
"""

import logging
import time
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from dataclasses import dataclass

import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    GenerationConfig
)
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import bitsandbytes as bnb

from .config_manager import ConfigManager

logger = logging.getLogger(__name__)

@dataclass
class GenerationResult:
    """Result from model generation."""
    generated_text: str
    input_tokens: int
    output_tokens: int
    generation_time: float
    model_used: str
    cached: bool = False

class ModelInferenceEngine:
    """Multi-model inference engine with GPU optimization."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.models = {}
        self.tokenizers = {}
        self.current_model = None
        
        # GPU configuration
        self.device_map = config.get('hardware.device_map', 'auto')
        self.gpu_count = config.get('hardware.gpu_count', 8)
        
        # Generation cache
        self.generation_cache = {}
        self.cache_enabled = config.get('cache.enable_disk_cache', True)
        
        logger.info(f"ModelInferenceEngine initialized for {self.gpu_count} GPUs")
    
    def load_model(self, model_name: str) -> bool:
        """Load a specific model.
        
        Args:
            model_name: Name of model to load ('codellama' or 'open_model')
            
        Returns:
            True if successful, False otherwise
        """
        if model_name in self.models:
            logger.info(f"Model {model_name} already loaded")
            self.current_model = model_name
            return True
        
        model_config = self.config.get_model_config(model_name)
        if not model_config:
            logger.error(f"No configuration found for model: {model_name}")
            return False
        
        logger.info(f"Loading model: {model_name}")
        
        try:
            # Load tokenizer
            tokenizer = self._load_tokenizer(model_config)
            
            # Load model with optimizations
            model = self._load_model_with_optimizations(model_config)
            
            # Store loaded components
            self.models[model_name] = model
            self.tokenizers[model_name] = tokenizer
            self.current_model = model_name
            
            logger.info(f"Successfully loaded model: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return False
    
    def _load_tokenizer(self, model_config: Dict[str, Any]) -> AutoTokenizer:
        """Load tokenizer for model."""
        model_path = model_config.get('path')
        model_name = model_config.get('name')
        
        # Try local path first, then model name
        if model_path and Path(model_path).exists():
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True,
                trust_remote_code=True
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
        
        # Set padding token if not present
        if tokenizer.pad_token is None:
            if tokenizer.eos_token:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        return tokenizer
    
    def _load_model_with_optimizations(self, model_config: Dict[str, Any]) -> AutoModelForCausalLM:
        """Load model with GPU and memory optimizations."""
        model_path = model_config.get('path')
        model_name = model_config.get('name')
        precision = model_config.get('precision', 'fp16')
        quantization = model_config.get('quantization', None)
        
        # Configure quantization
        quantization_config = None
        if quantization == '8bit':
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
        elif quantization == '4bit':
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        
        # Configure torch dtype
        torch_dtype = torch.float16 if precision == 'fp16' else torch.float32
        
        # Load model
        model_source = model_path if model_path and Path(model_path).exists() else model_name
        
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            quantization_config=quantization_config,
            torch_dtype=torch_dtype,
            device_map=self.device_map,
            trust_remote_code=True,
            local_files_only=model_path and Path(model_path).exists(),
            use_flash_attention_2=self.config.get('performance.use_flash_attention', True)
        )
        
        # Enable gradient checkpointing if configured
        if self.config.get('performance.gradient_checkpointing', False):
            model.gradient_checkpointing_enable()
        
        # Compile model if configured (PyTorch 2.0+)
        if self.config.get('performance.compile_model', True):
            try:
                model = torch.compile(model)
                logger.info("Model compiled with torch.compile")
            except Exception as e:
                logger.warning(f"Model compilation failed: {e}")
        
        return model
    
    def generate(self, prompt: str, model_name: str = None, **generation_kwargs) -> GenerationResult:
        """Generate text using specified model.
        
        Args:
            prompt: Input prompt
            model_name: Model to use (uses current if None)
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Generation result
        """
        if model_name is None:
            model_name = self.current_model
        
        if model_name is None:
            raise ValueError("No model loaded or specified")
        
        if model_name not in self.models:
            if not self.load_model(model_name):
                raise ValueError(f"Failed to load model: {model_name}")
        
        # Check cache
        cache_key = self._get_cache_key(prompt, model_name, generation_kwargs)
        if self.cache_enabled and cache_key in self.generation_cache:
            logger.debug("Using cached generation result")
            result = self.generation_cache[cache_key]
            result.cached = True
            return result
        
        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]
        
        start_time = time.time()
        
        try:
            # Prepare generation parameters
            gen_params = self._prepare_generation_params(model_name, generation_kwargs)
            
            # Tokenize input
            inputs = tokenizer(
                prompt, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=gen_params.get('max_length', 4096)
            )
            
            # Move to appropriate device
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            input_length = inputs['input_ids'].shape[1]
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    **gen_params,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True
                )
            
            # Decode output
            generated_ids = outputs[0][input_length:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            generation_time = time.time() - start_time
            
            result = GenerationResult(
                generated_text=generated_text,
                input_tokens=input_length,
                output_tokens=len(generated_ids),
                generation_time=generation_time,
                model_used=model_name,
                cached=False
            )
            
            # Cache result
            if self.cache_enabled:
                self.generation_cache[cache_key] = result
            
            logger.debug(f"Generated {len(generated_ids)} tokens in {generation_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    def _prepare_generation_params(self, model_name: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare generation parameters with defaults from config."""
        # Get base parameters from config
        base_params = self.config.get_generation_params()
        
        # Model-specific overrides
        model_config = self.config.get_model_config(model_name)
        model_params = {
            'max_length': model_config.get('max_length', 4096),
            'temperature': model_config.get('temperature', 0.3),
            'top_p': model_config.get('top_p', 0.9),
        }
        
        # Combine parameters (kwargs override config)
        gen_params = {**base_params, **model_params, **kwargs}
        
        # Ensure required parameters
        if 'max_new_tokens' not in gen_params:
            gen_params['max_new_tokens'] = gen_params.get('max_tokens', 1024)
        
        # Remove conflicting parameters
        if 'max_tokens' in gen_params:
            del gen_params['max_tokens']
        
        return gen_params
    
    def _get_cache_key(self, prompt: str, model_name: str, kwargs: Dict[str, Any]) -> str:
        """Generate cache key for generation request."""
        import hashlib
        
        # Create deterministic string from parameters
        cache_data = {
            'prompt': prompt,
            'model': model_name,
            'params': sorted(kwargs.items())
        }
        
        cache_str = str(cache_data)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def switch_model(self, model_name: str) -> bool:
        """Switch to a different loaded model.
        
        Args:
            model_name: Name of model to switch to
            
        Returns:
            True if successful, False otherwise
        """
        if model_name not in self.models:
            logger.warning(f"Model {model_name} not loaded, attempting to load")
            return self.load_model(model_name)
        
        self.current_model = model_name
        logger.info(f"Switched to model: {model_name}")
        return True
    
    def unload_model(self, model_name: str):
        """Unload a model to free memory.
        
        Args:
            model_name: Name of model to unload
        """
        if model_name in self.models:
            del self.models[model_name]
            del self.tokenizers[model_name]
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info(f"Unloaded model: {model_name}")
            
            # Switch to another model if current was unloaded
            if self.current_model == model_name:
                if self.models:
                    self.current_model = next(iter(self.models.keys()))
                    logger.info(f"Switched to model: {self.current_model}")
                else:
                    self.current_model = None
    
    def get_available_models(self) -> List[str]:
        """Get list of available models from config."""
        models_config = self.config.get('models', {})
        return list(models_config.keys())
    
    def get_loaded_models(self) -> List[str]:
        """Get list of currently loaded models."""
        return list(self.models.keys())
    
    def is_model_loaded(self, model_name: str) -> bool:
        """Check if a model is loaded."""
        return model_name in self.models
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a model."""
        if model_name not in self.models:
            return {'loaded': False}
        
        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]
        
        # Calculate memory usage
        param_count = sum(p.numel() for p in model.parameters())
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        
        info = {
            'loaded': True,
            'parameter_count': param_count,
            'memory_size_mb': param_size / (1024 * 1024),
            'device': str(next(model.parameters()).device),
            'vocab_size': tokenizer.vocab_size,
            'model_type': model.config.model_type if hasattr(model, 'config') else 'unknown'
        }
        
        return info
    
    def clear_cache(self):
        """Clear generation cache."""
        self.generation_cache.clear()
        logger.info("Generation cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get inference engine statistics."""
        stats = {
            'loaded_models': list(self.models.keys()),
            'current_model': self.current_model,
            'cache_size': len(self.generation_cache),
            'cache_enabled': self.cache_enabled,
            'gpu_count': self.gpu_count,
            'device_map': self.device_map
        }
        
        # Add model-specific stats
        for model_name in self.models:
            stats[f'{model_name}_info'] = self.get_model_info(model_name)
        
        # GPU memory stats
        if torch.cuda.is_available():
            gpu_stats = {}
            for i in range(torch.cuda.device_count()):
                memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)  # GB
                memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)   # GB
                gpu_stats[f'gpu_{i}'] = {
                    'memory_allocated_gb': memory_allocated,
                    'memory_reserved_gb': memory_reserved
                }
            stats['gpu_memory'] = gpu_stats
        
        return stats
