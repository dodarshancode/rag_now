"""
Production-Ready OpenSCENARIO 2.0 RAG System
Complete system integrating all components for secure offline code generation.
"""

import logging
import time
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from .core.config_manager import ConfigManager
from .core.document_processor import DocumentProcessor, DocumentChunk
from .core.retrieval_engine import HybridRetrievalEngine, RetrievalResult
from .core.model_inference import ModelInferenceEngine, GenerationResult
from .core.osc_validator import OSCValidator, ValidationResult
from .core.cache_manager import CacheManager

logger = logging.getLogger(__name__)

@dataclass
class OSCGenerationRequest:
    """Request for OpenSCENARIO code generation."""
    query: str
    model_name: Optional[str] = None
    max_retries: int = 3
    use_validation: bool = True
    generation_params: Optional[Dict[str, Any]] = None

@dataclass
class OSCGenerationResponse:
    """Response from OpenSCENARIO code generation."""
    query: str
    generated_code: str
    validation_result: Optional[ValidationResult] = None
    retrieval_results: List[RetrievalResult] = None
    generation_result: Optional[GenerationResult] = None
    total_time: float = 0.0
    retry_count: int = 0
    cached: bool = False

class OpenSCENARIORAGSystem:
    """Complete OpenSCENARIO 2.0 RAG system for production use."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the OpenSCENARIO RAG system.
        
        Args:
            config_path: Path to configuration file
        """
        # Initialize configuration
        self.config = ConfigManager(config_path)
        self.config.ensure_directories()
        
        # Initialize components
        self.document_processor = DocumentProcessor(self.config)
        self.retrieval_engine = HybridRetrievalEngine(self.config)
        self.inference_engine = ModelInferenceEngine(self.config)
        self.validator = OSCValidator(self.config)
        self.cache_manager = CacheManager(self.config)
        
        # System state
        self.initialized = False
        self.data_loaded = {
            'documentation': False,
            'code_examples': False
        }
        
        logger.info("OpenSCENARIO RAG System initialized")
    
    async def initialize_async(self):
        """Initialize the system asynchronously."""
        logger.info("Starting system initialization...")
        
        try:
            # Load cached indices if available
            if not self.retrieval_engine.load_indices():
                logger.info("No cached indices found, will need to process documents")
            else:
                logger.info("Loaded cached retrieval indices")
                self.data_loaded['documentation'] = True
                self.data_loaded['code_examples'] = True
            
            # Load default model
            available_models = self.inference_engine.get_available_models()
            if available_models:
                default_model = available_models[0]  # Use first available model
                logger.info(f"Loading default model: {default_model}")
                if not self.inference_engine.load_model(default_model):
                    logger.warning(f"Failed to load default model: {default_model}")
            
            # Check validator availability
            if not self.validator.check_parser_availability():
                logger.warning("OpenSCENARIO parser not available - validation disabled")
            
            self.initialized = True
            logger.info("System initialization completed")
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            raise
    
    def load_documentation(self, pdf_path: str) -> bool:
        """Load OpenSCENARIO documentation from PDF.
        
        Args:
            pdf_path: Path to ASAM OpenSCENARIO PDF
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Loading documentation from: {pdf_path}")
            
            # Process PDF documentation
            doc_chunks = self.document_processor.process_pdf_documentation(pdf_path)
            
            if not doc_chunks:
                logger.error("No chunks extracted from documentation")
                return False
            
            # Add to retrieval engine
            self.retrieval_engine.add_chunks(doc_chunks)
            
            # Save indices
            self.retrieval_engine.save_indices()
            
            self.data_loaded['documentation'] = True
            logger.info(f"Loaded {len(doc_chunks)} documentation chunks")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load documentation: {e}")
            return False
    
    def load_code_examples(self, examples_dir: str) -> bool:
        """Load code examples from directory.
        
        Args:
            examples_dir: Directory containing code examples
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Loading code examples from: {examples_dir}")
            
            # Process code examples
            code_chunks = self.document_processor.process_code_examples(examples_dir)
            
            if not code_chunks:
                logger.warning("No code examples found")
                return True  # Not an error, just no examples
            
            # Add to retrieval engine
            self.retrieval_engine.add_chunks(code_chunks)
            
            # Save indices
            self.retrieval_engine.save_indices()
            
            self.data_loaded['code_examples'] = True
            logger.info(f"Loaded {len(code_chunks)} code example chunks")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load code examples: {e}")
            return False
    
    async def generate_openscenario_code(self, request: OSCGenerationRequest) -> OSCGenerationResponse:
        """Generate OpenSCENARIO 2.0 code from natural language description.
        
        Args:
            request: Generation request
            
        Returns:
            Generation response with code and validation results
        """
        if not self.initialized:
            raise RuntimeError("System not initialized. Call initialize_async() first.")
        
        start_time = time.time()
        
        # Check cache first
        cache_key = self._get_cache_key(request)
        cached_response = self.cache_manager.get(cache_key, 'generation')
        if cached_response:
            cached_response.cached = True
            logger.info("Using cached generation result")
            return cached_response
        
        logger.info(f"Generating OpenSCENARIO code for query: {request.query[:100]}...")
        
        # Perform retrieval
        retrieval_results = self.retrieval_engine.search(
            query=request.query,
            top_k=self.config.get('retrieval.top_k_final', 10)
        )
        
        # Generate code with validation loop
        response = await self._generate_with_validation_loop(request, retrieval_results)
        
        response.total_time = time.time() - start_time
        
        # Cache successful results
        if response.validation_result is None or response.validation_result.is_valid:
            self.cache_manager.set(cache_key, response, namespace='generation')
        
        logger.info(f"Code generation completed in {response.total_time:.2f}s")
        return response
    
    async def _generate_with_validation_loop(self, request: OSCGenerationRequest, 
                                           retrieval_results: List[RetrievalResult]) -> OSCGenerationResponse:
        """Generate code with validation and retry loop."""
        current_prompt = self._build_initial_prompt(request.query, retrieval_results)
        retry_count = 0
        
        while retry_count <= request.max_retries:
            try:
                # Generate code
                generation_result = self.inference_engine.generate(
                    prompt=current_prompt,
                    model_name=request.model_name,
                    **(request.generation_params or {})
                )
                
                # Extract OpenSCENARIO code from generation
                generated_code = self._extract_openscenario_code(generation_result.generated_text)
                
                # Validate if requested
                validation_result = None
                if request.use_validation and self.validator.check_parser_availability():
                    validation_result = self.validator.validate_code(generated_code)
                    
                    # If validation failed and we have retries left, create feedback prompt
                    if not validation_result.is_valid and retry_count < request.max_retries:
                        feedback = self.validator.generate_error_feedback(
                            validation_result, request.query
                        )
                        current_prompt = self._build_retry_prompt(
                            request.query, retrieval_results, generated_code, feedback
                        )
                        retry_count += 1
                        logger.info(f"Validation failed, retrying ({retry_count}/{request.max_retries})")
                        continue
                
                # Success or max retries reached
                return OSCGenerationResponse(
                    query=request.query,
                    generated_code=generated_code,
                    validation_result=validation_result,
                    retrieval_results=retrieval_results,
                    generation_result=generation_result,
                    retry_count=retry_count
                )
                
            except Exception as e:
                logger.error(f"Generation attempt {retry_count} failed: {e}")
                if retry_count >= request.max_retries:
                    raise
                retry_count += 1
        
        # Should not reach here
        raise RuntimeError("Generation failed after all retries")
    
    def _build_initial_prompt(self, query: str, retrieval_results: List[RetrievalResult]) -> str:
        """Build initial prompt for code generation."""
        prompt_parts = [
            "# OpenSCENARIO 2.0 Code Generation",
            "",
            "You are an expert in OpenSCENARIO 2.0 DSL. Generate valid OpenSCENARIO 2.0 code based on the user's description.",
            "",
            "## User Request:",
            query,
            "",
            "## Relevant Examples and Documentation:",
        ]
        
        # Add retrieval results
        for i, result in enumerate(retrieval_results[:5], 1):
            chunk_type = result.chunk.metadata.get('type', 'unknown')
            source = result.chunk.metadata.get('source', 'unknown')
            
            prompt_parts.extend([
                f"### Example {i} ({chunk_type} from {source}):",
                result.chunk.content[:800] + ("..." if len(result.chunk.content) > 800 else ""),
                ""
            ])
        
        prompt_parts.extend([
            "## Instructions:",
            "1. Generate valid OpenSCENARIO 2.0 code that fulfills the user's request",
            "2. Use proper syntax and structure according to OpenSCENARIO 2.0 specification",
            "3. Include all required parameters and attributes",
            "4. Ensure entity references are properly defined",
            "5. Use appropriate data types and values",
            "",
            "## Generated OpenSCENARIO 2.0 Code:",
            "```openscenario"
        ])
        
        return "\n".join(prompt_parts)
    
    def _build_retry_prompt(self, query: str, retrieval_results: List[RetrievalResult], 
                           failed_code: str, feedback: str) -> str:
        """Build retry prompt with validation feedback."""
        prompt_parts = [
            "# OpenSCENARIO 2.0 Code Generation - Correction Required",
            "",
            "The previous attempt had validation errors. Please correct the code.",
            "",
            "## Original Request:",
            query,
            "",
            "## Previous Code (with errors):",
            "```openscenario",
            failed_code,
            "```",
            "",
            "## Validation Feedback:",
            feedback,
            "",
            "## Relevant Examples:",
        ]
        
        # Add fewer examples for retry to keep prompt concise
        for i, result in enumerate(retrieval_results[:3], 1):
            chunk_type = result.chunk.metadata.get('type', 'unknown')
            prompt_parts.extend([
                f"### Example {i} ({chunk_type}):",
                result.chunk.content[:500] + ("..." if len(result.chunk.content) > 500 else ""),
                ""
            ])
        
        prompt_parts.extend([
            "## Corrected OpenSCENARIO 2.0 Code:",
            "```openscenario"
        ])
        
        return "\n".join(prompt_parts)
    
    def _extract_openscenario_code(self, generated_text: str) -> str:
        """Extract OpenSCENARIO code from generated text."""
        # Look for code blocks
        import re
        
        # Try to find code in markdown code blocks
        code_block_pattern = r'```(?:openscenario|osc)?\n(.*?)\n```'
        matches = re.findall(code_block_pattern, generated_text, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        
        # If no code blocks, try to extract based on OpenSCENARIO keywords
        lines = generated_text.split('\n')
        code_lines = []
        in_code = False
        
        for line in lines:
            # Start collecting when we see OpenSCENARIO keywords
            if any(keyword in line.lower() for keyword in ['scenario', 'entity', 'action', 'condition', 'trigger']):
                in_code = True
            
            if in_code:
                code_lines.append(line)
                
                # Stop if we see obvious non-code content
                if line.strip().startswith(('##', '**', 'Note:', 'This code')):
                    code_lines.pop()  # Remove the non-code line
                    break
        
        if code_lines:
            return '\n'.join(code_lines).strip()
        
        # Fallback: return the entire generated text
        return generated_text.strip()
    
    def _get_cache_key(self, request: OSCGenerationRequest) -> str:
        """Generate cache key for generation request."""
        return self.cache_manager.get_cache_key_for_query(
            query=request.query,
            model_name=request.model_name or self.inference_engine.current_model or 'default',
            generation_params=request.generation_params or {}
        )
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        stats = {
            'system_info': {
                'initialized': self.initialized,
                'data_loaded': self.data_loaded,
                'config_file': str(self.config.config_path)
            },
            'document_processor': self.document_processor.get_processing_stats(),
            'retrieval_engine': self.retrieval_engine.get_stats(),
            'inference_engine': self.inference_engine.get_stats(),
            'validator': self.validator.get_stats(),
            'cache_manager': self.cache_manager.get_stats()
        }
        
        return stats
    
    def switch_model(self, model_name: str) -> bool:
        """Switch to a different model.
        
        Args:
            model_name: Name of model to switch to
            
        Returns:
            True if successful, False otherwise
        """
        return self.inference_engine.switch_model(model_name)
    
    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        return self.inference_engine.get_available_models()
    
    def clear_caches(self, namespace: str = None):
        """Clear system caches.
        
        Args:
            namespace: Specific namespace to clear (clears all if None)
        """
        self.cache_manager.clear(namespace)
        self.inference_engine.clear_cache()
        logger.info(f"Caches cleared: {namespace or 'all'}")
    
    def cleanup(self):
        """Cleanup system resources."""
        try:
            # Save indices
            self.retrieval_engine.save_indices()
            
            # Close cache connections
            self.cache_manager.close()
            
            # Clear GPU memory
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("System cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()
