"""Parallel kernel labeling for SVD decompositions.

Labels all kernels concurrently using ThreadPoolExecutor.
Expected 3-4× speedup for typical 8-12 kernels.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from ukt.kernels import KernelDecomposition, label_kernel
from ukt.registry import FeatureRegionRegistry


def label_all_kernels_parallel(
    decomposition: KernelDecomposition,
    registry: FeatureRegionRegistry,
    block_names: list[str],
    max_workers: int = 4,
    feature_meta: dict | None = None,
    timeframe_context: dict | None = None,
) -> list[dict]:
    """Label all kernels in parallel.

    Args:
        decomposition: SVD decomposition result
        registry: Feature region registry
        block_names: Block names (rows)
        max_workers: Number of parallel workers
        feature_meta: Optional per-feature metadata
        timeframe_context: Optional temporal context

    Returns:
        List of kernel label dicts (same order as kernel indices)
    """
    results = [None] * decomposition.n_kernels

    def label_kernel_task(k_idx: int) -> tuple[int, dict]:
        """Label a single kernel and return (index, result) tuple."""
        label = label_kernel(
            k_idx=k_idx,
            decomposition=decomposition,
            registry=registry,
            block_names=block_names,
            feature_meta=feature_meta,
            timeframe_context=timeframe_context,
        )
        return k_idx, label

    # Execute labeling in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(label_kernel_task, k_idx): k_idx
            for k_idx in range(decomposition.n_kernels)
        }

        for future in as_completed(futures):
            try:
                k_idx, label = future.result()
                results[k_idx] = label
            except Exception:
                # Fallback: use sequential labeling for this kernel
                k_idx = futures[future]
                results[k_idx] = label_kernel(
                    k_idx, decomposition, registry, block_names,
                    feature_meta, timeframe_context
                )

    return results
