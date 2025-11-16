# Auto-generated C++ extension loader for SAM2
import os
os.environ.setdefault('CUDA_HOME', '/usr/local/cuda-12.2')

import torch
from torch.utils.cpp_extension import load
from pathlib import Path

# Get the csrc directory
sam2_dir = Path(__file__).parent
csrc_dir = sam2_dir / "csrc"

# Load the compiled extension
_C = load(
    name='sam2_ops',
    sources=[str(csrc_dir / 'connected_components.cu')],
    extra_cuda_cflags=['-O3'],
    verbose=False
)

# Export all functions from the extension
globals().update({k: v for k, v in _C.__dict__.items() if not k.startswith('_')})

