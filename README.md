# RAG System

A production-ready offline RAG (Retrieval-Augmented Generation) system for generating valid DSL code from natural language descriptions. Built for high-performance deployment on 8x A100 GPUs with complete offline operation.

## 🚀 Features

- **Complete Offline Operation**: No external API calls, all models and data stored locally
- **8x A100 GPU Optimization**: Multi-GPU model sharding, FP16/8-bit quantization, GPU-accelerated FAISS
- **Hybrid Retrieval**: Dense vector search (FAISS) + sparse keyword search (BM25) with tag-aware ranking
- **Multi-Model Support**: CodeLlama-13B-Instruct + WizardCoder-15B with automatic model switching
- **Validation Pipeline**: py-osc2 parser integration with automatic error correction and retry logic
- **Production Caching**: Multi-level encrypted caching (memory/Redis/disk) for optimal performance
- **Modern Web UI**: Responsive interface with real-time monitoring and metrics
- **Prometheus Monitoring**: Production-grade metrics and health checks

## 📋 System Requirements

### Hardware
- **GPU**: 8x NVIDIA A100 80GB (minimum 4x A100 40GB)
- **RAM**: 256GB+ system memory recommended
- **Storage**: 500GB+ SSD for models and cache
- **OS**: Ubuntu 20.04/22.04 LTS

### Software
- **CUDA**: 12.1 (installed on host)
- **Python**: 3.10 (optimal for CUDA 12.1 compatibility)
- **Conda**: Miniconda or Anaconda

## 🛠️ Installation

### 1. Clone Repository
```bash
git clone https://github.com/your-username/openscenario-rag-system.git
cd openscenario-rag-system
```

### 2. Create Conda Environment
```bash
# Create environment with Python 3.10 for CUDA 12.1 compatibility
conda create -n openscenario-rag python=3.10 -y
conda activate openscenario-rag

# Install PyTorch with CUDA 12.1 support
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

### 3. Install Dependencies
```bash
# Install all required packages
pip install -r requirements.txt

# Verify CUDA installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
```

### 4. Install OpenSCENARIO Parser
```bash
# Install py-osc2 parser for validation
pip install py-osc2

# Verify installation
py-osc2 --help
```

## 📁 Project Structure

```
openscenario-rag-system/
├── src/                              # Core system components
│   ├── core/                         # Core modules
│   │   ├── config_manager.py         # YAML configuration management
│   │   ├── document_processor.py     # PDF parsing with LangExtract
│   │   ├── vector_store.py           # GPU-accelerated FAISS vector store
│   │   ├── retrieval_engine.py       # Hybrid FAISS + BM25 retrieval
│   │   ├── model_inference.py        # Multi-model inference engine
│   │   ├── osc_validator.py          # py-osc2 validation integration
│   │   └── cache_manager.py          # Multi-level caching system
│   └── openscenario_rag_system.py    # Main system orchestrator
├── config/
│   └── openscenario_config.yaml      # System configuration
├── templates/
│   └── index.html                    # Web UI interface
├── data/                             # Data directory (create these)
│   ├── models/                       # Model storage
│   │   ├── codellama-13b/           # CodeLlama model files
│   │   ├── wizardcoder-15b/         # WizardCoder model files
│   │   └── codebert-base/           # CodeBERT embedding model
│   ├── docs/                        # Documentation files
│   │   └── ASAM-OpenSCENARIO-2.0.pdf # OpenSCENARIO specification
│   ├── examples/                    # Code examples directory
│   ├── processed/                   # Processed document chunks
│   ├── embeddings/                  # Cached embeddings
│   └── cache/                       # System cache
├── logs/                            # Log files
├── app.py                           # Flask API server
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## 📊 Data Setup

### 1. Create Data Directories
```bash
mkdir -p data/{models,docs,examples,processed,embeddings,cache}
mkdir -p logs
```

### 2. Download Models

#### CodeLlama-13B-Instruct
```bash
# Using Hugging Face CLI (install with: pip install huggingface_hub)
huggingface-cli download codellama/CodeLlama-13b-Instruct-hf --local-dir data/models/codellama-13b

# Or using Python
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained('codellama/CodeLlama-13b-Instruct-hf')
tokenizer = AutoTokenizer.from_pretrained('codellama/CodeLlama-13b-Instruct-hf')
model.save_pretrained('data/models/codellama-13b')
tokenizer.save_pretrained('data/models/codellama-13b')
"
```

#### WizardCoder-15B (Alternative Model)
```bash
huggingface-cli download WizardLM/WizardCoder-15B-V1.0 --local-dir data/models/wizardcoder-15b
```

#### CodeBERT Embedding Model
```bash
huggingface-cli download microsoft/codebert-base --local-dir data/models/codebert-base
```

### 3. Place Documentation
```bash
# Download ASAM OpenSCENARIO 2.0 specification PDF
# Place the PDF file at: data/docs/ASAM-OpenSCENARIO-2.0.pdf
wget -O data/docs/ASAM-OpenSCENARIO-2.0.pdf "https://www.asam.net/index.php?eID=dumpFile&t=f&f=4422&token=e590561f3c39aa2260e5442e29e93f6693d1cccd"
```

### 4. Prepare Code Examples
```bash
# Create example OpenSCENARIO code files in data/examples/
# Organize by scenario type:
mkdir -p data/examples/{cut-in,lane-change,overtaking,parking,intersection}

# Example structure:
# data/examples/cut-in/basic_cut_in.osc
# data/examples/lane-change/highway_lane_change.json
# data/examples/overtaking/rural_overtaking.osc
```

#### Example Code File Format (JSON)
```json
[
  {
    "instruction": "Create a cut-in scenario",
    "input": "Vehicle performs cut-in maneuver in front of ego vehicle",
    "output": "scenario CutInScenario {\n  // OpenSCENARIO 2.0 code here\n}",
    "tags": ["cut-in", "lane-change"],
    "description": "Basic cut-in scenario with lane change"
  }
]
```

## ⚙️ Configuration

### 1. Update Configuration File
Edit `config/openscenario_config.yaml`:

```yaml
# Update paths to match your setup
data:
  documentation_pdf: "data/docs/ASAM-OpenSCENARIO-2.0.pdf"
  code_examples_dir: "data/examples/"
  models_dir: "data/models/"

# Configure for your GPU setup
hardware:
  gpu_count: 8  # Adjust based on available GPUs
  
# Model paths
models:
  codellama:
    path: "data/models/codellama-13b"
  open_model:
    path: "data/models/wizardcoder-15b"

embedding:
  model_path: "data/models/codebert-base"
```

### 2. Optional: Redis Setup (for production caching)
```bash
# Install Redis
sudo apt update
sudo apt install redis-server -y

# Start Redis
sudo systemctl start redis-server
sudo systemctl enable redis-server

# Test Redis
redis-cli ping  # Should return PONG
```

## 🚀 Running the System

### 1. Start the System
```bash
# Activate environment
conda activate openscenario-rag

# Start Flask API server
python app.py
```

### 2. Access Web Interface
Open your browser and navigate to:
```
http://localhost:5000
```

### 3. Load Data (First Time Setup)
1. **Load Documentation**: Enter path to PDF: `data/docs/ASAM-OpenSCENARIO-2.0.pdf`
2. **Load Code Examples**: Enter path to examples: `data/examples/`
3. Wait for processing to complete (this may take several minutes)

### 4. Generate Code
1. Enter a natural language description of your OpenSCENARIO scenario
2. Configure generation parameters (model, retries, validation)
3. Click "Generate OpenSCENARIO Code"
4. Review the generated code and validation results

## 📝 Example Usage

### Natural Language Inputs
```
Create a cut-in scenario where a vehicle changes lanes in front of the ego vehicle on a highway

Generate a parallel driving scenario with two vehicles maintaining constant speed of 60 km/h

Create an intersection scenario with traffic light control and pedestrian crossing

Generate a parking scenario with obstacle avoidance in a parking lot

Create an overtaking maneuver on a rural road with oncoming traffic
```

### Generated Output
The system will produce valid OpenSCENARIO 2.0 DSL code with:
- Proper syntax and structure
- Required parameters and attributes
- Valid entity references
- Automatic validation with py-osc2 parser
- Error correction and retry if needed

## 🔧 API Endpoints

### Core Endpoints
- `GET /` - Web interface
- `GET /api/health` - System health check
- `GET /api/stats` - System statistics
- `POST /api/generate` - Generate OpenSCENARIO code
- `POST /api/validate` - Validate OpenSCENARIO code
- `POST /api/load-documentation` - Load PDF documentation
- `POST /api/load-examples` - Load code examples
- `GET /api/models` - List available models
- `POST /api/models/{model}/switch` - Switch active model
- `POST /api/cache/clear` - Clear system caches
- `GET /metrics` - Prometheus metrics

### Example API Usage
```bash
# Generate code via API
curl -X POST http://localhost:5000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Create a cut-in scenario with lane change",
    "use_validation": true,
    "max_retries": 3
  }'

# Validate existing code
curl -X POST http://localhost:5000/api/validate \
  -H "Content-Type: application/json" \
  -d '{
    "code": "scenario TestScenario { ... }"
  }'
```

## 🔍 Monitoring & Debugging

### 1. Check System Status
```bash
# Health check
curl http://localhost:5000/api/health

# System statistics
curl http://localhost:5000/api/stats
```

### 2. View Logs
```bash
# Application logs
tail -f logs/openscenario_rag.log

# Flask development logs
python app.py  # Logs to console in debug mode
```

### 3. Monitor GPU Usage
```bash
# Monitor GPU utilization
nvidia-smi -l 1

# Monitor GPU memory
watch -n 1 nvidia-smi
```

### 4. Prometheus Metrics
Access metrics at: `http://localhost:5000/metrics`

Key metrics:
- `openscenario_requests_total` - Total API requests
- `openscenario_generation_duration_seconds` - Code generation time
- `openscenario_validation_success_total` - Successful validations

## 🛠️ Troubleshooting

### Common Issues

#### CUDA Out of Memory
```bash
# Reduce batch size in config
# Enable gradient checkpointing
# Use 8-bit quantization
```

#### Model Loading Fails
```bash
# Check model paths in config
# Verify model files are complete
# Check disk space
```

#### Validation Errors
```bash
# Verify py-osc2 installation
py-osc2 --version

# Check parser availability
curl -X GET http://localhost:5000/api/stats | grep validator
```

#### Performance Issues
```bash
# Clear caches
curl -X POST http://localhost:5000/api/cache/clear

# Restart system
python app.py
```

### Environment Variables
```bash
# Optional environment variables
export FLASK_HOST=0.0.0.0
export FLASK_PORT=5000
export FLASK_DEBUG=False
export LOG_LEVEL=INFO
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

## 📈 Performance Optimization

### 1. GPU Memory Optimization
- Use 8-bit quantization for models
- Enable gradient checkpointing
- Adjust batch sizes based on available memory

### 2. Caching Strategy
- Enable Redis for production deployments
- Configure appropriate TTL values
- Monitor cache hit rates

### 3. Model Selection
- Use CodeLlama-13B for best OpenSCENARIO generation
- Switch to WizardCoder-15B for alternative results
- Consider model ensemble for critical applications

## 🔒 Security Considerations

### 1. Data Encryption
- All cached data is encrypted using Fernet
- Encryption keys stored securely with restricted permissions

### 2. Network Security
- Configure firewall rules for production deployment
- Use HTTPS in production environments
- Implement authentication if needed

### 3. Model Security
- Store models locally to prevent data leakage
- Validate all inputs before processing
- Monitor system access and usage

## 📚 Additional Resources

- [OpenSCENARIO 2.0 Specification](https://www.asam.net/standards/detail/openscenario/)
- [py-osc2 Parser Documentation](https://github.com/asam-ev/py-osc2)
- [FAISS Documentation](https://faiss.ai/)
- [Transformers Library](https://huggingface.co/docs/transformers/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review system logs
3. Open an issue on GitHub
4. Contact the development team

---

**Note**: This system is designed for production use with high-performance hardware. Ensure your system meets the minimum requirements before deployment.
