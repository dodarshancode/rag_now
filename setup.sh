#!/bin/bash
# OpenSCENARIO 2.0 RAG System Environment Setup
# For CUDA 12.1 and 8x A100 GPUs

set -e

echo "🚀 Setting up OpenSCENARIO RAG environment..."

# Create conda environment with Python 3.10
echo "📦 Creating conda environment with Python 3.10..."
conda create -n openscenario-rag python=3.10 -y

# Activate environment
echo "🔄 Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate openscenario-rag

# Install PyTorch with CUDA 12.1 support
echo "🔥 Installing PyTorch with CUDA 12.1..."
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install FAISS-GPU with CUDA support
echo "🔍 Installing FAISS-GPU..."
conda install -c conda-forge faiss-gpu=1.7.4 -y

# Install other GPU-accelerated packages via conda
echo "⚡ Installing additional GPU packages..."
conda install -c conda-forge cupy cuda-version=12.1 -y

# Install Python packages
echo "📚 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Verify installations
echo "✅ Verifying installations..."
python -c "
import torch
import faiss
import transformers
import sentence_transformers

print(f'Python version: {torch.version.__version__ if hasattr(torch.version, \"__version__\") else \"3.10+\"}')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU count: {torch.cuda.device_count()}')
print(f'FAISS version: {faiss.__version__}')
print(f'Transformers version: {transformers.__version__}')
print(f'Sentence-transformers version: {sentence_transformers.__version__}')

# Test GPU memory
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'GPU {i}: {props.name} - {props.total_memory / 1024**3:.1f} GB')
"

echo "🎉 Environment setup complete!"
echo "💡 To activate: conda activate openscenario-rag"
echo "🚀 To run: python app.py"
