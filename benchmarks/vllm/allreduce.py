"""NCCL all-reduce on its own, with the NCCL that vLLM uses: two processes, one per GPU, as vLLM runs them.

usage: [NCCL_P2P_DISABLE=1] python allreduce.py

Prints the time per all-reduce of float32 data from 4 KiB to 64 MiB. Set NCCL_DEBUG=INFO to see the transport
(`via P2P/...` or `via SHM/...`).
"""
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

SIZES = [4 << 10, 64 << 10, 512 << 10, 1 << 20, 4 << 20, 16 << 20, 64 << 20]
PORT = os.environ.get("PORT", "29533")


def worker(rank):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{PORT}", rank=rank, world_size=2,
                            device_id=torch.device(f"cuda:{rank}"))
    for size in SIZES:
        x = torch.ones(size // 4, dtype=torch.float32, device=f"cuda:{rank}")
        iters = 200 if size <= (1 << 20) else 30
        for _ in range(10):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        dist.barrier()
        t = time.perf_counter()
        for _ in range(iters):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t) / iters
        assert torch.all(x > 0)
        if rank == 0:
            print(f"ALLREDUCE {size >> 10:6d} KiB: {dt * 1e6:9.1f} us, {size / dt / 1e9:5.2f} GB/s", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    print("NCCL_P2P_DISABLE =", os.environ.get("NCCL_P2P_DISABLE", "(unset)"), "torch", torch.__version__,
          "nccl", ".".join(map(str, torch.cuda.nccl.version())), flush=True)
    mp.spawn(worker, nprocs=2)
