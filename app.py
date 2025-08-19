"""
Production Flask API for OpenSCENARIO 2.0 RAG System
Provides REST API endpoints with monitoring and performance optimization.
"""

import logging
import asyncio
import time
from pathlib import Path
from typing import Dict, Any, Optional
import json

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import structlog
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from src.openscenario_rag_system import OpenSCENARIORAGSystem, OSCGenerationRequest

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Prometheus metrics
REQUEST_COUNT = Counter('openscenario_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('openscenario_request_duration_seconds', 'Request duration', ['endpoint'])
GENERATION_DURATION = Histogram('openscenario_generation_duration_seconds', 'Code generation duration')
VALIDATION_SUCCESS = Counter('openscenario_validation_success_total', 'Successful validations')
VALIDATION_FAILURES = Counter('openscenario_validation_failures_total', 'Failed validations')

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'openscenario_rag_production_key'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

# Enable CORS
CORS(app, origins=['http://localhost:3000', 'http://localhost:5000'])

# Global RAG system
rag_system: Optional[OpenSCENARIORAGSystem] = None
system_ready = False

def initialize_system():
    """Initialize the RAG system."""
    global rag_system, system_ready
    
    try:
        logger.info("Initializing OpenSCENARIO RAG system")
        rag_system = OpenSCENARIORAGSystem()
        
        # Run async initialization
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(rag_system.initialize_async())
        loop.close()
        
        system_ready = True
        logger.info("RAG system initialized successfully")
        
    except Exception as e:
        logger.error("Failed to initialize RAG system", error=str(e))
        system_ready = False

# Initialize system on startup
@app.before_first_request
def startup():
    """Initialize system before first request."""
    initialize_system()

@app.before_request
def before_request():
    """Log request start and set up timing."""
    request.start_time = time.time()
    logger.info("Request started", 
                method=request.method, 
                path=request.path,
                remote_addr=request.remote_addr)

@app.after_request
def after_request(response):
    """Log request completion and update metrics."""
    duration = time.time() - getattr(request, 'start_time', time.time())
    
    # Update metrics
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.endpoint or 'unknown',
        status=response.status_code
    ).inc()
    
    REQUEST_DURATION.labels(endpoint=request.endpoint or 'unknown').observe(duration)
    
    logger.info("Request completed",
                method=request.method,
                path=request.path,
                status_code=response.status_code,
                duration=duration)
    
    return response

@app.route('/')
def index():
    """Main application page."""
    return render_template('index.html')

@app.route('/api/health')
def health_check():
    """Health check endpoint."""
    if not system_ready or not rag_system:
        return jsonify({
            'status': 'unhealthy',
            'message': 'System not initialized',
            'timestamp': time.time()
        }), 503
    
    try:
        stats = rag_system.get_system_stats()
        
        health_status = {
            'status': 'healthy',
            'initialized': stats['system_info']['initialized'],
            'data_loaded': stats['system_info']['data_loaded'],
            'models_loaded': stats['inference_engine']['loaded_models'],
            'validator_available': stats['validator']['parser_available'],
            'timestamp': time.time()
        }
        
        return jsonify(health_status)
        
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return jsonify({
            'status': 'unhealthy',
            'message': str(e),
            'timestamp': time.time()
        }), 503

@app.route('/api/stats')
def get_stats():
    """Get comprehensive system statistics."""
    if not system_ready or not rag_system:
        return jsonify({'error': 'System not initialized'}), 503
    
    try:
        stats = rag_system.get_system_stats()
        return jsonify(stats)
        
    except Exception as e:
        logger.error("Failed to get stats", error=str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate', methods=['POST'])
def generate_code():
    """Generate OpenSCENARIO 2.0 code from natural language description."""
    if not system_ready or not rag_system:
        return jsonify({'error': 'System not initialized'}), 503
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        query = data.get('query', '').strip()
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        # Create generation request
        generation_request = OSCGenerationRequest(
            query=query,
            model_name=data.get('model_name'),
            max_retries=data.get('max_retries', 3),
            use_validation=data.get('use_validation', True),
            generation_params=data.get('generation_params', {})
        )
        
        # Generate code
        start_time = time.time()
        
        # Run async generation
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            response = loop.run_until_complete(
                rag_system.generate_openscenario_code(generation_request)
            )
        finally:
            loop.close()
        
        generation_time = time.time() - start_time
        GENERATION_DURATION.observe(generation_time)
        
        # Update validation metrics
        if response.validation_result:
            if response.validation_result.is_valid:
                VALIDATION_SUCCESS.inc()
            else:
                VALIDATION_FAILURES.inc()
        
        # Format response
        result = {
            'query': response.query,
            'generated_code': response.generated_code,
            'validation': {
                'is_valid': response.validation_result.is_valid if response.validation_result else None,
                'error_message': response.validation_result.error_message if response.validation_result else None,
                'suggestions': response.validation_result.suggestions if response.validation_result else []
            } if response.validation_result else None,
            'retrieval_context': [
                {
                    'content': result.chunk.content[:500] + ('...' if len(result.chunk.content) > 500 else ''),
                    'score': result.combined_score,
                    'source': result.chunk.metadata.get('source', 'unknown'),
                    'type': result.chunk.metadata.get('type', 'unknown'),
                    'tags': result.chunk.metadata.get('tags', [])
                }
                for result in (response.retrieval_results or [])[:5]
            ],
            'generation_info': {
                'model_used': response.generation_result.model_used if response.generation_result else None,
                'input_tokens': response.generation_result.input_tokens if response.generation_result else 0,
                'output_tokens': response.generation_result.output_tokens if response.generation_result else 0,
                'generation_time': response.generation_result.generation_time if response.generation_result else 0,
                'retry_count': response.retry_count,
                'total_time': response.total_time,
                'cached': response.cached
            }
        }
        
        logger.info("Code generation completed",
                   query_length=len(query),
                   code_length=len(response.generated_code),
                   validation_valid=response.validation_result.is_valid if response.validation_result else None,
                   retry_count=response.retry_count,
                   total_time=response.total_time,
                   cached=response.cached)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error("Code generation failed", error=str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/api/validate', methods=['POST'])
def validate_code():
    """Validate OpenSCENARIO 2.0 code."""
    if not system_ready or not rag_system:
        return jsonify({'error': 'System not initialized'}), 503
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        code = data.get('code', '').strip()
        if not code:
            return jsonify({'error': 'Code is required'}), 400
        
        # Validate code
        validation_result = rag_system.validator.validate_code(code)
        
        # Update metrics
        if validation_result.is_valid:
            VALIDATION_SUCCESS.inc()
        else:
            VALIDATION_FAILURES.inc()
        
        result = {
            'is_valid': validation_result.is_valid,
            'error_message': validation_result.error_message,
            'error_type': validation_result.error_type,
            'line_number': validation_result.line_number,
            'suggestions': validation_result.suggestions or [],
            'validation_time': validation_result.validation_time
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error("Code validation failed", error=str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/api/load-documentation', methods=['POST'])
def load_documentation():
    """Load OpenSCENARIO documentation from PDF."""
    if not system_ready or not rag_system:
        return jsonify({'error': 'System not initialized'}), 503
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        pdf_path = data.get('pdf_path', '').strip()
        if not pdf_path:
            return jsonify({'error': 'PDF path is required'}), 400
        
        if not Path(pdf_path).exists():
            return jsonify({'error': f'PDF file not found: {pdf_path}'}), 400
        
        # Load documentation
        success = rag_system.load_documentation(pdf_path)
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Documentation loaded successfully'
            })
        else:
            return jsonify({'error': 'Failed to load documentation'}), 500
            
    except Exception as e:
        logger.error("Documentation loading failed", error=str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/api/load-examples', methods=['POST'])
def load_examples():
    """Load code examples from directory."""
    if not system_ready or not rag_system:
        return jsonify({'error': 'System not initialized'}), 503
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        examples_dir = data.get('examples_dir', '').strip()
        if not examples_dir:
            return jsonify({'error': 'Examples directory is required'}), 400
        
        if not Path(examples_dir).exists():
            return jsonify({'error': f'Examples directory not found: {examples_dir}'}), 400
        
        # Load examples
        success = rag_system.load_code_examples(examples_dir)
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Code examples loaded successfully'
            })
        else:
            return jsonify({'error': 'Failed to load code examples'}), 500
            
    except Exception as e:
        logger.error("Examples loading failed", error=str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/api/models')
def get_models():
    """Get available and loaded models."""
    if not system_ready or not rag_system:
        return jsonify({'error': 'System not initialized'}), 503
    
    try:
        available_models = rag_system.get_available_models()
        loaded_models = rag_system.inference_engine.get_loaded_models()
        current_model = rag_system.inference_engine.current_model
        
        return jsonify({
            'available_models': available_models,
            'loaded_models': loaded_models,
            'current_model': current_model
        })
        
    except Exception as e:
        logger.error("Failed to get models", error=str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/api/models/<model_name>/switch', methods=['POST'])
def switch_model(model_name: str):
    """Switch to a different model."""
    if not system_ready or not rag_system:
        return jsonify({'error': 'System not initialized'}), 503
    
    try:
        success = rag_system.switch_model(model_name)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'Switched to model: {model_name}',
                'current_model': model_name
            })
        else:
            return jsonify({'error': f'Failed to switch to model: {model_name}'}), 500
            
    except Exception as e:
        logger.error("Model switch failed", error=str(e), model=model_name)
        return jsonify({'error': str(e)}), 500

@app.route('/api/cache/clear', methods=['POST'])
def clear_cache():
    """Clear system caches."""
    if not system_ready or not rag_system:
        return jsonify({'error': 'System not initialized'}), 503
    
    try:
        data = request.get_json() or {}
        namespace = data.get('namespace')
        
        rag_system.clear_caches(namespace)
        
        return jsonify({
            'success': True,
            'message': f'Cache cleared: {namespace or "all"}'
        })
        
    except Exception as e:
        logger.error("Cache clear failed", error=str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/metrics')
def metrics():
    """Prometheus metrics endpoint."""
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors."""
    return jsonify({'error': 'Method not allowed'}), 405

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error("Internal server error", error=str(error))
    return jsonify({'error': 'Internal server error'}), 500

@app.teardown_appcontext
def cleanup_context(error):
    """Cleanup on app context teardown."""
    if error:
        logger.error("App context error", error=str(error))

if __name__ == '__main__':
    import os
    
    # Configuration
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    # Setup logging
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Starting OpenSCENARIO RAG API server",
                host=host, port=port, debug=debug)
    
    # Run Flask app
    app.run(
        host=host,
        port=port,
        debug=debug,
        threaded=True
    )
