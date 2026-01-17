"""
Utilities for adaptive multi-GPU dataset processing.

Modules:
- gpu_utils: GPU detection, memory monitoring, worker assignment
- adaptive_workers: Incremental worker spawning with OOM recovery
"""

from .gpu_utils import detect_gpus, get_available_memory, assign_worker_to_gpu
from .adaptive_workers import AdaptiveWorkerPool

__all__ = [
    'detect_gpus',
    'get_available_memory',
    'assign_worker_to_gpu',
    'AdaptiveWorkerPool',
]
