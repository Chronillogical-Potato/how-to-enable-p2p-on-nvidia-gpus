"""Matrix-multiply speed of each GPU in fp16 and fp32, at Qwen3-1.7B's tensor-parallel-2 shapes.

usage: python gemm.py

The fp16 rows are what vLLM uses with --dtype half. "fp16 via fp32" converts fp16 inputs to fp32 for every
multiply, which is how a card without fast fp16 can still run an fp16 model at its fp32 speed.
"""
import time

import torch

SHAPES = [(4096, 2048, 2048), (4096, 2048, 6144), (4096, 3072, 2048)]  # qkv, gate_up, down for a 4096-token chunk


def tflops(fn, m, k, n, dev, iters=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize(dev)
    t = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize(dev)
    return 2 * m * k * n / ((time.time() - t) / iters) / 1e12


for d in range(torch.cuda.device_count()):
    dev = f"cuda:{d}"
    name = torch.cuda.get_device_name(d)
    for m, k, n in SHAPES:
        a16 = torch.randn(m, k, device=dev, dtype=torch.float16)
        b16 = torch.randn(k, n, device=dev, dtype=torch.float16)
        a32, b32 = a16.float(), b16.float()
        r16 = tflops(lambda: a16 @ b16, m, k, n, dev)
        r32 = tflops(lambda: a32 @ b32, m, k, n, dev)
        rup = tflops(lambda: (a16.float() @ b16.float()).half(), m, k, n, dev)
        print(f"GPU{d} {name:30s} {m}x{k}x{n}: fp16 {r16:6.2f}  fp32 {r32:6.2f}  fp16 via fp32 {rup:6.2f} TFLOPS")
