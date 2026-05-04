#!/usr/bin/env python3
"""
Benchmark for fused comp-comm symmetric memory operations.

Usage:
python benchmarks/distributed/bench_fused_symm_mem_ops.py

Benchmarks:
  - fused_matmul_allreduce vs unfused (matmul + dist.all_reduce)
  - fused_allreduce_rmsnorm vs unfused (dist.all_reduce + F.rms_norm)
"""

import time

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import torch.nn.functional as F
from torch.testing._internal.common_distributed import MultiProcContinuousTest
from torch.testing._internal.common_utils import (
    requires_cuda_p2p_access,
    run_tests,
)


device_type = "cuda"
device_module = torch.get_device_module(device_type)


def _benchmark(fn, warmup_iters=5, bench_iters=20):
    for _ in range(warmup_iters):
        fn()
        torch.cuda.synchronize()

    dist.barrier()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(bench_iters):
        fn()
        torch.cuda.synchronize()
    end = time.perf_counter()

    return (end - start) / bench_iters * 1000  # ms


@requires_cuda_p2p_access()
class FusedSymmMemBenchmark(MultiProcContinuousTest):
    def _init_device(self) -> None:
        device_module.set_device(self.device)

    @property
    def device(self) -> torch.device:
        return torch.device(device_type, self.rank)

    def _bench_fused_matmul_allreduce(self, M, K, N, dtype=torch.float32):
        self._init_device()
        group = dist.group.WORLD
        group_name = group.group_name

        A = torch.randn(M, K, device=self.device, dtype=dtype)
        B = torch.randn(K, N, device=self.device, dtype=dtype)

        def unfused():
            out = A @ B
            dist.all_reduce(out)
            return out

        def fused():
            return torch.ops.symm_mem.fused_matmul_allreduce(
                A, B, "sum", group_name
            )

        unfused_ms = _benchmark(unfused)
        fused_ms = _benchmark(fused)

        return unfused_ms, fused_ms

    def _bench_fused_allreduce_rmsnorm(self, M, N, dtype=torch.float32):
        self._init_device()
        group = dist.group.WORLD
        group_name = group.group_name
        eps = 1e-5

        input_tensor = torch.randn(M, N, device=self.device, dtype=dtype)
        weight = torch.randn(N, device=self.device, dtype=dtype)

        def unfused():
            out = input_tensor.clone()
            dist.all_reduce(out)
            return F.rms_norm(out, [N], weight, eps)

        def fused():
            return torch.ops.symm_mem.fused_allreduce_rmsnorm(
                input_tensor, weight, eps, "sum", group_name
            )

        unfused_ms = _benchmark(unfused)
        fused_ms = _benchmark(fused)

        return unfused_ms, fused_ms

    def test_benchmark_fused_matmul_allreduce(self) -> None:
        configs = [
            (1, 4096, 4096),       # single-token decode (one_shot: 16KB)
            (8, 4096, 4096),       # small batch decode (one_shot: 128KB)
            (32, 4096, 4096),      # medium batch (two_shot: 512KB)
            (128, 4096, 4096),     # prefill (two_shot: 2MB)
            (256, 8192, 8192),     # large prefill (two_shot: 8MB)
            (1024, 4096, 4096),    # large batch (fallback: 16MB)
        ]

        if self.rank == 0:
            print("\n=== fused_matmul_allreduce Benchmark ===")
            print(
                f"{'M':>6} {'K':>6} {'N':>6} "
                f"{'Unfused (ms)':>14} {'Fused (ms)':>12} {'Speedup':>9}"
            )
            print("-" * 60)

        for M, K, N in configs:
            unfused_ms, fused_ms = self._bench_fused_matmul_allreduce(M, K, N)
            if self.rank == 0:
                speedup = unfused_ms / fused_ms if fused_ms > 0 else float("inf")
                print(
                    f"{M:>6} {K:>6} {N:>6} "
                    f"{unfused_ms:>14.3f} {fused_ms:>12.3f} {speedup:>8.2f}x"
                )

    def test_benchmark_fused_allreduce_rmsnorm(self) -> None:
        configs = [
            (1, 4096),        # single-token decode (one_shot: 16KB)
            (8, 4096),        # small batch decode (one_shot: 128KB)
            (32, 4096),       # medium batch (two_shot: 512KB)
            (128, 4096),      # prefill (two_shot: 2MB)
            (256, 8192),      # large prefill (two_shot: 8MB)
            (1024, 4096),     # large batch (fallback: 16MB)
        ]

        if self.rank == 0:
            print("\n=== fused_allreduce_rmsnorm Benchmark ===")
            print(
                f"{'M':>6} {'N':>6} "
                f"{'Unfused (ms)':>14} {'Fused (ms)':>12} {'Speedup':>9}"
            )
            print("-" * 50)

        for M, N in configs:
            unfused_ms, fused_ms = self._bench_fused_allreduce_rmsnorm(M, N)
            if self.rank == 0:
                speedup = unfused_ms / fused_ms if fused_ms > 0 else float("inf")
                print(
                    f"{M:>6} {N:>6} "
                    f"{unfused_ms:>14.3f} {fused_ms:>12.3f} {speedup:>8.2f}x"
                )


if __name__ == "__main__":
    run_tests()
