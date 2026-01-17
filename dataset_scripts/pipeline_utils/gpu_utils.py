"""
GPU utilities for adaptive worker pool.

Functions for GPU detection, memory monitoring, and worker assignment.
"""

import subprocess
import threading
from typing import Dict, List, Tuple, Optional
import torch


def detect_gpus() -> Tuple[int, List[Dict]]:
    """
    Detect available GPUs and their specifications.

    Returns:
        num_gpus: Number of available GPUs
        gpu_info: List of dicts with {name, memory_gb, compute_capability, id}

    Example:
        >>> num_gpus, gpu_info = detect_gpus()
        >>> print(f"Found {num_gpus} GPUs")
        >>> for gpu in gpu_info:
        ...     print(f"GPU {gpu['id']}: {gpu['name']} ({gpu['memory_gb']:.1f}GB)")
    """
    if not torch.cuda.is_available():
        return 0, []

    num_gpus = torch.cuda.device_count()
    gpu_info = []

    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)

        gpu_info.append({
            'id': i,
            'name': props.name,
            'memory_gb': props.total_memory / (1024 ** 3),  # Convert to GB
            'compute_capability': f"{props.major}.{props.minor}",
        })

    return num_gpus, gpu_info


def get_available_memory(gpu_id: int = 0) -> float:
    """
    Get free GPU memory in GB using nvidia-smi.

    Args:
        gpu_id: GPU index to query

    Returns:
        Free memory in GB

    Example:
        >>> free_gb = get_available_memory(0)
        >>> print(f"GPU 0 has {free_gb:.1f}GB free")
    """
    try:
        result = subprocess.run(
            [
                'nvidia-smi',
                '--query-gpu=memory.free',
                '--format=csv,nounits,noheader',
                f'--id={gpu_id}'
            ],
            capture_output=True,
            text=True,
            check=True
        )

        free_memory_mb = float(result.stdout.strip())
        return free_memory_mb / 1024  # Convert MB to GB

    except (subprocess.CalledProcessError, ValueError, FileNotFoundError) as e:
        # Fallback to PyTorch if nvidia-smi fails
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)
            free_memory_bytes = torch.cuda.mem_get_info()[0]
            return free_memory_bytes / (1024 ** 3)
        return 0.0


def assign_worker_to_gpu(worker_id: int, num_gpus: int, strategy: str = 'round-robin') -> int:
    """
    Assign worker to GPU based on strategy.

    Args:
        worker_id: Worker index (0, 1, 2, ...)
        num_gpus: Total number of GPUs
        strategy: Assignment strategy
            - 'round-robin': Distribute evenly (worker_id % num_gpus)
            - 'fill-first': Fill GPU 0, then GPU 1, etc.
            - 'memory-aware': Assign to GPU with most free memory

    Returns:
        GPU ID (0-indexed)

    Example:
        >>> # 4 workers, 2 GPUs, round-robin
        >>> for worker_id in range(4):
        ...     gpu_id = assign_worker_to_gpu(worker_id, 2, 'round-robin')
        ...     print(f"Worker {worker_id} → GPU {gpu_id}")
        Worker 0 → GPU 0
        Worker 1 → GPU 1
        Worker 2 → GPU 0
        Worker 3 → GPU 1
    """
    if num_gpus == 0:
        raise ValueError("No GPUs available")

    if num_gpus == 1:
        return 0  # Only one GPU

    if strategy == 'round-robin':
        return worker_id % num_gpus

    elif strategy == 'fill-first':
        # Assign workers sequentially: 0,0,1,1,2,2...
        workers_per_gpu = 2  # Can be made configurable
        return min(worker_id // workers_per_gpu, num_gpus - 1)

    elif strategy == 'memory-aware':
        # Find GPU with most free memory
        max_free_memory = 0
        best_gpu = 0

        for gpu_id in range(num_gpus):
            free_memory = get_available_memory(gpu_id)
            if free_memory > max_free_memory:
                max_free_memory = free_memory
                best_gpu = gpu_id

        return best_gpu

    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'round-robin', 'fill-first', or 'memory-aware'")


class GPUMemoryMonitor:
    """
    Background thread to monitor peak GPU memory usage.

    Usage:
        monitor = GPUMemoryMonitor(gpu_id=0)
        monitor.start()
        # ... do GPU work ...
        monitor.stop()
        print(f"Peak memory: {monitor.peak_memory_gb:.2f}GB")
    """

    def __init__(self, gpu_id: int = 0, interval: float = 0.5):
        """
        Args:
            gpu_id: GPU to monitor
            interval: Polling interval in seconds
        """
        self.gpu_id = gpu_id
        self.interval = interval
        self.peak_memory_gb = 0.0
        self.current_memory_gb = 0.0
        self._stop_event = threading.Event()
        self._thread = None

    def _monitor_loop(self):
        """Background monitoring loop"""
        while not self._stop_event.is_set():
            try:
                # Get total and free memory
                if torch.cuda.is_available():
                    torch.cuda.set_device(self.gpu_id)
                    free_bytes, total_bytes = torch.cuda.mem_get_info()
                    used_bytes = total_bytes - free_bytes
                    used_gb = used_bytes / (1024 ** 3)

                    self.current_memory_gb = used_gb
                    if used_gb > self.peak_memory_gb:
                        self.peak_memory_gb = used_gb

            except Exception:
                pass  # Ignore errors during monitoring

            self._stop_event.wait(self.interval)

    def start(self):
        """Start monitoring"""
        if self._thread is None or not self._thread.is_alive():
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._thread.start()

    def stop(self):
        """Stop monitoring"""
        if self._thread and self._thread.is_alive():
            self._stop_event.set()
            self._thread.join(timeout=2.0)

    def reset(self):
        """Reset peak memory counter"""
        self.peak_memory_gb = 0.0


if __name__ == "__main__":
    # Test GPU detection
    print("Testing GPU detection...")
    num_gpus, gpu_info = detect_gpus()
    print(f"\nFound {num_gpus} GPU(s):")
    for gpu in gpu_info:
        print(f"  GPU {gpu['id']}: {gpu['name']} ({gpu['memory_gb']:.1f}GB, compute {gpu['compute_capability']})")
        free = get_available_memory(gpu['id'])
        print(f"    Free memory: {free:.1f}GB")

    # Test worker assignment
    if num_gpus > 0:
        print(f"\nTesting worker assignment (round-robin with {num_gpus} GPU(s)):")
        for worker_id in range(8):
            gpu_id = assign_worker_to_gpu(worker_id, num_gpus, 'round-robin')
            print(f"  Worker {worker_id} → GPU {gpu_id}")

    # Test memory monitor
    if num_gpus > 0:
        print("\nTesting GPU memory monitor...")
        monitor = GPUMemoryMonitor(gpu_id=0, interval=0.1)
        monitor.start()

        # Allocate some memory
        import time
        x = torch.randn(1000, 1000, device='cuda:0')
        time.sleep(0.5)
        y = torch.randn(2000, 2000, device='cuda:0')
        time.sleep(0.5)

        monitor.stop()
        print(f"  Peak memory: {monitor.peak_memory_gb:.2f}GB")
        print(f"  Current memory: {monitor.current_memory_gb:.2f}GB")

        # Cleanup
        del x, y
        torch.cuda.empty_cache()
