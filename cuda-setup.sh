#!/bin/bash

# CUDA Environment Setup Script
# This script sets up the CUDA environment variables

echo "Setting up CUDA environment..."

# Add CUDA paths to PATH and LD_LIBRARY_PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Verify CUDA installation
echo "CUDA Version:"
nvcc --version

echo "GPU Status:"
nvidia-smi

echo "CUDA installation verified successfully!"
echo ""
echo "To make these changes permanent, add the following lines to your ~/.bashrc:"
echo "export PATH=/usr/local/cuda/bin:\$PATH"
echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH"
echo ""
echo "To activate the virtual environment, run:"
echo "source .venv/bin/activate"
