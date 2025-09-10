"""Parallel batch processing utilities for faster evaluation."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from functools import partial
from typing import Any

from loguru import logger


class ParallelBatchProcessor:
    """Process batches in parallel to significantly speed up evaluation."""

    def __init__(
        self,
        max_workers: int | None = None,
        use_threads: bool = False,
        timeout: float | None = None,
    ):
        """
        Initialize parallel batch processor.

        Args:
            max_workers: Maximum number of parallel workers. Defaults to CPU count.
            use_threads: Whether to use threads instead of processes.
                        Use threads for I/O-bound tasks, processes for CPU-bound.
            timeout: Timeout for individual batch processing in seconds.
        """
        self.max_workers = max_workers or min(os.cpu_count() or 1, 8)
        self.use_threads = use_threads
        self.timeout = timeout

        logger.info(
            f"Initialized ParallelBatchProcessor: {self.max_workers} workers, "
            f"{'threads' if use_threads else 'processes'}"
        )

    def process_batches_parallel(
        self, batch_processor_func: Callable, batches: list[Any], **kwargs
    ) -> list[Any]:
        """
        Process multiple batches in parallel.

        Args:
            batch_processor_func: Function to process a single batch.
                                Should accept (batch_idx, batch, **kwargs)
            batches: List of batch data to process
            **kwargs: Additional arguments to pass to batch_processor_func

        Returns:
            List of results from batch processing
        """
        if len(batches) <= 1:
            # Not worth parallelizing for single batch
            if batches:
                return [batch_processor_func(0, batches[0], **kwargs)]
            return []

        executor_class = ThreadPoolExecutor if self.use_threads else ProcessPoolExecutor

        # Create partial function with fixed kwargs
        process_func = partial(self._process_single_batch, batch_processor_func, **kwargs)

        results = [None] * len(batches)

        logger.info(
            f"Processing {len(batches)} batches in parallel with {self.max_workers} workers"
        )
        start_time = time.time()

        with executor_class(max_workers=self.max_workers) as executor:
            # Submit all batch jobs
            future_to_idx = {
                executor.submit(process_func, idx, batch): idx for idx, batch in enumerate(batches)
            }

            # Collect results as they complete
            for completed, future in enumerate(as_completed(future_to_idx, timeout=self.timeout), 1):
                batch_idx = future_to_idx[future]

                # Get result - let exceptions propagate to caller
                result = future.result()
                results[batch_idx] = result

                if completed % max(1, len(batches) // 10) == 0:
                    logger.info(
                        f"Completed {completed}/{len(batches)} batches "
                        f"({100 * completed / len(batches):.1f}%)"
                    )

        elapsed = time.time() - start_time
        logger.info(
            f"Parallel batch processing completed in {elapsed:.2f}s "
            f"({len(batches) / elapsed:.2f} batches/sec)"
        )

        return results

    @staticmethod
    def _process_single_batch(batch_processor_func, batch_idx: int, batch: Any, **kwargs):
        """Process a single batch - wrapper for multiprocessing."""
        return batch_processor_func(batch_idx, batch, **kwargs)


def parallel_batch_wrapper(
    original_process_func: Callable, parallel_config: dict[str, Any] | None = None
) -> Callable:
    """
    Decorator to add parallel processing to existing batch processing functions.

    Args:
        original_process_func: Original batch processing function
        parallel_config: Configuration for parallel processing

    Returns:
        Wrapped function that can process batches in parallel
    """
    parallel_config = parallel_config or {}

    def wrapped_processor(batches: list[Any], enable_parallel: bool = True, **kwargs):
        if not enable_parallel or len(batches) <= 1:
            # Fall back to sequential processing
            results = []
            for idx, batch in enumerate(batches):
                result = original_process_func(idx, batch, **kwargs)
                results.append(result)
            return results

        # Use parallel processing
        processor = ParallelBatchProcessor(**parallel_config)
        return processor.process_batches_parallel(original_process_func, batches, **kwargs)

    return wrapped_processor


def get_optimal_workers(task_type: str = "mixed") -> int:
    """
    Get optimal number of workers based on task type and system resources.

    Args:
        task_type: Type of task - "cpu", "io", or "mixed"

    Returns:
        Optimal number of workers
    """
    cpu_count = os.cpu_count() or 1

    if task_type == "cpu":
        # CPU-bound tasks: use number of cores
        return cpu_count
    elif task_type == "io":
        # I/O-bound tasks: can use more workers
        return min(cpu_count * 2, 16)
    else:
        # Mixed workload: conservative approach
        return min(cpu_count, 8)


def estimate_speedup(
    num_batches: int, batch_time_seconds: float, num_workers: int | None = None
) -> dict[str, float]:
    """
    Estimate speedup from parallel processing.

    Args:
        num_batches: Number of batches to process
        batch_time_seconds: Average time per batch in seconds
        num_workers: Number of parallel workers

    Returns:
        Dictionary with speedup estimates
    """
    if num_workers is None:
        num_workers = get_optimal_workers()

    sequential_time = num_batches * batch_time_seconds

    # Account for overhead and diminishing returns
    efficiency = 0.8  # Assume 80% efficiency due to overhead
    effective_workers = min(num_workers * efficiency, num_batches)

    parallel_time = sequential_time / effective_workers
    speedup = sequential_time / parallel_time

    return {
        "sequential_time": sequential_time,
        "parallel_time": parallel_time,
        "speedup": speedup,
        "workers": num_workers,
        "efficiency": efficiency,
    }
