"""
Adaptive worker pool with incremental spawning and OOM recovery.

Incrementally spawns workers until OOM is detected, then continues with stable workers.
Supports multi-GPU work distribution with round-robin assignment.
"""

import os
import sys
import time
import pickle
import signal
import multiprocessing as mp
from multiprocessing import Queue, Event
from queue import Empty
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable, Any
from tqdm import tqdm

# Use spawn context for CUDA compatibility
mp_ctx = mp.get_context('spawn')


class AdaptiveWorkerPool:
    """
    Worker pool that spawns workers incrementally until OOM.

    Features:
    - Incremental spawning (1→2→3... until failure)
    - Shared work queue (workers pull items)
    - OOM detection and recovery
    - Progress tracking with tqdm
    - Multi-GPU support
    - Optional checkpointing
    """

    def __init__(
        self,
        num_gpus: int,
        max_workers_per_gpu: int,
        worker_fn: Callable,
        checkpoint_dir: Optional[Path] = None,
        checkpoint_every: int = 100,
        spawn_delay: float = 3.0,
        gpu_strategy: str = 'round-robin',
        verbose_workers: bool = False,
        save_worker_logs: bool = True,
        log_dir: Optional[Path] = None,
    ):
        """
        Args:
            num_gpus: Number of available GPUs (0 for CPU-only)
            max_workers_per_gpu: Maximum workers to attempt per GPU
            worker_fn: Worker function object with load_model() and process() methods
            checkpoint_dir: Directory for checkpoints (None = disabled)
            checkpoint_every: Save checkpoint every N successful items
            spawn_delay: Seconds to wait after spawning before spawning next worker
            gpu_strategy: GPU assignment strategy ('round-robin', 'fill-first', 'memory-aware')
            verbose_workers: Show worker output in main terminal (debug mode)
            save_worker_logs: Save worker logs to files
            log_dir: Directory for worker logs (defaults to checkpoint_dir or cwd)
        """
        self.num_gpus = num_gpus
        self.max_workers_per_gpu = max_workers_per_gpu
        self.worker_fn = worker_fn
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.checkpoint_every = checkpoint_every
        self.spawn_delay = spawn_delay
        self.gpu_strategy = gpu_strategy
        self.verbose_workers = verbose_workers
        self.save_worker_logs = save_worker_logs
        self.log_dir = Path(log_dir) if log_dir else (self.checkpoint_dir or Path.cwd())

        # State
        self.active_workers: List[Process] = []
        self.failed_worker_ids: List[int] = []
        self.max_stable_workers: int = 0
        self.processed_count: int = 0
        self._interrupted: bool = False
        self._current_results: Dict = {}

        # IPC (use spawn context for CUDA compatibility)
        self.work_queue = mp_ctx.Queue()
        self.result_queue = mp_ctx.Queue()
        self.stop_event = mp_ctx.Event()

        # Signal handlers for graceful shutdown
        self._original_sigint = signal.signal(signal.SIGINT, self._handle_interrupt)
        self._original_sigterm = signal.signal(signal.SIGTERM, self._handle_interrupt)

    def process_items(
        self,
        items: List[Any],
        desc: str = "Processing"
    ) -> Tuple[Dict[Any, Any], int]:
        """
        Main entry point: process items with adaptive worker pool.

        Args:
            items: List of items to process
            desc: Description for progress bar

        Returns:
            results: Dict mapping item -> result (None if failed)
            stable_workers: Number of workers that succeeded
        """
        # Check for existing checkpoint
        checkpoint = self._load_checkpoint()
        if checkpoint:
            print(f"🔄 Resuming from checkpoint ({checkpoint['processed_count']} items)")
            results = checkpoint['results']
            already_processed = set(results.keys())
            items = [item for item in items if item not in already_processed]
            self.processed_count = checkpoint['processed_count']
        else:
            results = {}

        if not items:
            print("✅ All items already processed!")
            return results, self.max_stable_workers

        # Fill work queue
        for item in items:
            self.work_queue.put(('process', item))

        # Create progress bar BEFORE spawning (timer includes full processing time)
        pbar = tqdm(total=len(items), desc=desc, unit="video")

        # Spawn workers incrementally
        self._spawn_workers_incrementally(desc=desc)

        if self.max_stable_workers == 0:
            pbar.close()
            print("❌ No workers could be spawned successfully")
            return results, 0

        print(f"📊 Spawned {self.max_stable_workers} stable workers")

        # Send stop messages to workers (one per worker)
        for _ in range(self.max_stable_workers):
            self.work_queue.put(('stop', None))

        # Collect results with progress bar
        new_results = self._collect_results(
            total=len(items),
            desc=desc,
            pbar=pbar
        )

        pbar.close()

        # Merge results
        results.update(new_results)

        # Shutdown workers
        self._shutdown_workers()

        return results, self.max_stable_workers

    def _spawn_workers_incrementally(self, desc: str):
        """Spawn workers one at a time until OOM or max reached"""
        max_total_workers = self.num_gpus * self.max_workers_per_gpu if self.num_gpus > 0 else self.max_workers_per_gpu

        print(f"🚀 Incrementally spawning workers (max {max_total_workers})...")

        for worker_id in range(max_total_workers):
            # Assign GPU
            if self.num_gpus > 0:
                from pipeline_utils.gpu_utils import assign_worker_to_gpu
                gpu_id = assign_worker_to_gpu(worker_id, self.num_gpus, self.gpu_strategy)
            else:
                gpu_id = -1  # CPU-only

            # Spawn worker
            success = self._spawn_worker(worker_id, gpu_id)

            if not success:
                print(f"  Worker {worker_id} (GPU {gpu_id}): ✗ Failed to spawn (OOM or crash)")
                self.failed_worker_ids.append(worker_id)
                break

            print(f"  Worker {worker_id} (GPU {gpu_id}): ✓ Spawned")
            self.max_stable_workers = worker_id + 1

            # Wait before spawning next worker
            if worker_id < max_total_workers - 1:
                time.sleep(self.spawn_delay)

    def _spawn_worker(self, worker_id: int, gpu_id: int) -> bool:
        """
        Spawn a single worker process.

        Returns:
            True if worker successfully spawned and alive after spawn_delay
        """
        # Prepare log file path if saving logs
        worker_log_file = None
        if self.save_worker_logs:
            if self.log_dir:
                self.log_dir.mkdir(parents=True, exist_ok=True)
                worker_log_file = self.log_dir / f"worker_{worker_id}.log"

        worker = mp_ctx.Process(
            target=_worker_main,
            args=(
                worker_id,
                gpu_id,
                self.work_queue,
                self.result_queue,
                self.stop_event,
                self.worker_fn,
                self.verbose_workers,
                worker_log_file,
            ),
            daemon=True,
        )

        worker.start()
        self.active_workers.append(worker)

        # Wait to see if worker crashes immediately (OOM at model load)
        time.sleep(self.spawn_delay)

        if not worker.is_alive():
            print(f"    Worker died with exit code: {worker.exitcode}")
            return False

        return True

    def _collect_results(self, total: int, desc: str, pbar: tqdm) -> Dict[Any, Any]:
        """Collect results from workers with progress bar"""
        results = {}
        finished_workers = 0
        successful = 0
        failed = 0

        while finished_workers < self.max_stable_workers:
            # Check for interrupt
            if self._interrupted:
                print("\n⚠️ Interrupt detected, stopping workers...")
                self.stop_event.set()
                break

            try:
                msg = self.result_queue.get(timeout=1.0)
            except Empty:
                # Check if all workers died
                alive = sum(1 for w in self.active_workers if w.is_alive())
                if alive == 0:
                    print("\n⚠️ All workers died unexpectedly")
                    break
                continue

            msg_type = msg[0]

            if msg_type == 'success':
                _, worker_id, item, result = msg
                results[item] = result
                self._current_results = results  # For checkpoint on interrupt
                successful += 1
                self.processed_count += 1
                pbar.update(1)

                # Checkpoint periodically
                if self.checkpoint_every and self.processed_count % self.checkpoint_every == 0:
                    self._save_checkpoint(results)

            elif msg_type == 'error':
                _, worker_id, item, error = msg
                results[item] = None  # Mark as failed
                self._current_results = results  # For checkpoint on interrupt
                failed += 1
                pbar.update(1)

                # Log error
                self._log_error(item, error)

                # Check if OOM
                if 'CUDA out of memory' in str(error) or 'out of memory' in str(error).lower():
                    print(f"\n⚠️ Worker {worker_id} encountered OOM during processing")

            elif msg_type == 'finished':
                _, worker_id, _, _ = msg
                finished_workers += 1

        # Summary
        if failed > 0:
            print(f"⚠️ {failed} items failed (see error log)")

        return results

    def _handle_interrupt(self, signum, frame):
        """Handle Ctrl+C gracefully"""
        if not self._interrupted:
            self._interrupted = True
            print("\n\n⚠️  Interrupt received (Ctrl+C)!")
            print("    Saving checkpoint...")
            print("    Stopping workers gracefully...")
            print("    (Press Ctrl+C again to force quit)\n")

            # Save checkpoint with current results
            if self._current_results:
                self._save_checkpoint(self._current_results)

            # Signal workers to stop (will be checked in _collect_results)
            self.stop_event.set()
        else:
            # Second Ctrl+C = force quit
            print("\n⚠️  Force quit! Terminating workers...")
            for worker in self.active_workers:
                if worker.is_alive():
                    worker.terminate()
            # Restore original handlers
            signal.signal(signal.SIGINT, self._original_sigint)
            signal.signal(signal.SIGTERM, self._original_sigterm)
            sys.exit(1)

    def _shutdown_workers(self):
        """Gracefully shutdown all workers"""
        self.stop_event.set()

        # Wait for workers to finish
        for worker in self.active_workers:
            worker.join(timeout=5.0)
            if worker.is_alive():
                worker.terminate()

        # Restore original signal handlers
        signal.signal(signal.SIGINT, self._original_sigint)
        signal.signal(signal.SIGTERM, self._original_sigterm)

    def _save_checkpoint(self, results: Dict):
        """Save checkpoint of processed items"""
        if not self.checkpoint_dir:
            return

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_file = self.checkpoint_dir / f"checkpoint_{self.processed_count}.pkl"

        with open(checkpoint_file, 'wb') as f:
            pickle.dump({
                'results': results,
                'processed_count': self.processed_count,
                'stable_workers': self.max_stable_workers,
            }, f)

        print(f"💾 Checkpoint saved: {checkpoint_file}")

    def _load_checkpoint(self) -> Optional[Dict]:
        """Load most recent checkpoint"""
        if not self.checkpoint_dir or not self.checkpoint_dir.exists():
            return None

        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.pkl"))
        if not checkpoints:
            return None

        latest = checkpoints[-1]
        with open(latest, 'rb') as f:
            return pickle.load(f)

    def _log_error(self, item: Any, error: str):
        """Log failed item for review"""
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            error_log = self.checkpoint_dir / "errors.log"
        elif self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            error_log = self.log_dir / "errors.log"
        else:
            error_log = Path("adaptive_workers_errors.log")

        import datetime
        timestamp = datetime.datetime.now().isoformat()

        with open(error_log, 'a') as f:
            f.write(f"[{timestamp}] {item}: {error}\n")


def _worker_main(
    worker_id: int,
    gpu_id: int,
    work_queue: Queue,
    result_queue: Queue,
    stop_event: Event,
    worker_fn: Callable,
    verbose: bool = False,
    log_file: Optional[Path] = None,
):
    """
    Worker process main loop.

    Args:
        worker_id: Worker index
        gpu_id: GPU to use (-1 for CPU)
        work_queue: Queue to pull work from
        result_queue: Queue to push results to
        stop_event: Event to signal shutdown
        worker_fn: Worker function object with load_model() and process() methods
        verbose: Show output in main terminal (else redirect to log file)
        log_file: Path to log file (if not verbose)
    """
    # Redirect stdout/stderr if not verbose
    if not verbose and log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        sys.stdout = open(log_file, 'w')
        sys.stderr = sys.stdout

        # Suppress third-party library logs
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        import warnings
        warnings.filterwarnings('ignore')

    print(f"[Worker {worker_id}] Starting...")

    # Set GPU for this worker
    if gpu_id >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        # Load model once per worker
        print(f"[Worker {worker_id}] Loading model...")
        model = worker_fn.load_model()
        print(f"[Worker {worker_id}] Model loaded, entering work loop...")
    except Exception as e:
        # Failed to load model (likely OOM)
        print(f"[Worker {worker_id}] Model load failed: {e}")
        result_queue.put(('error', worker_id, None, f"Model load failed: {e}"))
        return

    # Process items from queue
    print(f"[Worker {worker_id}] Checking for work...")
    while not stop_event.is_set():
        try:
            msg = work_queue.get(timeout=1.0)
        except Empty:
            continue

        msg_type, item = msg

        if msg_type == 'stop':
            # Received stop signal, exit loop
            break
        elif msg_type == 'process':
            try:
                result = worker_fn.process(model, item)
                result_queue.put(('success', worker_id, item, result))
            except Exception as e:
                result_queue.put(('error', worker_id, item, str(e)))

    # Signal finished
    result_queue.put(('finished', worker_id, None, None))

    # Restore stdout/stderr if redirected
    if not verbose and log_file:
        sys.stdout.close()
        sys.stderr = sys.__stderr__
        sys.stdout = sys.__stdout__


# ============================================================================
# CPU-Only Worker Pool (for create_h5_dataset.py)
# ============================================================================

class CPUWorkerPool(AdaptiveWorkerPool):
    """
    CPU-bound worker pool (no GPU).

    Finds optimal worker count before I/O bottleneck.
    """

    def __init__(
        self,
        max_workers: int,
        worker_fn: Callable,
        checkpoint_dir: Optional[Path] = None,
        checkpoint_every: int = 100,
    ):
        """
        Args:
            max_workers: Maximum CPU workers to test
            worker_fn: Worker function object
            checkpoint_dir: Directory for checkpoints
            checkpoint_every: Checkpoint frequency
        """
        super().__init__(
            num_gpus=0,
            max_workers_per_gpu=max_workers,
            worker_fn=worker_fn,
            checkpoint_dir=checkpoint_dir,
            checkpoint_every=checkpoint_every,
            spawn_delay=1.0,  # Faster spawning for CPU
        )
