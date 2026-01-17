#!/usr/bin/env python3
"""Test adaptive worker pool with dummy task."""

from dataset_scripts.utils.adaptive_workers import AdaptiveWorkerPool

class DummyWorker:
    def load_model(self):
        return None

    def process(self, model, x):
        return x * 2

pool = AdaptiveWorkerPool(
    num_gpus=1,
    max_workers_per_gpu=2,
    worker_fn=DummyWorker(),
)

items = [1, 2, 3, 4, 5]
results, stable_workers = pool.process_items(items, desc='Dummy Test')

print(f'\n✅ Test Complete!')
print(f'Stable workers: {stable_workers}')
print(f'Results: {results}')
