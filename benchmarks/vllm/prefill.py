"""Prompt processing (prefill) without the web server: single prompts of several lengths, and batches.

usage: [NCCL_P2P_DISABLE=1] [SHAPES=1x1024,8x1024] python prefill.py

  MODEL    default Qwen/Qwen3-0.6B         DTYPE    default float32
  TP       default 2                       KV       KV cache bytes per GPU, default 1100000000
  BACKEND  attention backend (empty: vLLM's choice; TRITON_ATTN on Turing)
  NO_CAR   1 turns vLLM's custom all-reduce off (it reset the test rig's mixed Turing pair; README.md)
  SHAPES   <prompts>x<tokens>, comma-separated, default 1x256,1x1024,1x2048,1x4000,1x8192,8x1024,16x512

Each shape is run once as warmup, then three times with fresh random prompts; the fastest is printed.

With TORCH_PROFILE_DIR=<dir>, vLLM's workers record the three timed runs of each shape with the PyTorch profiler
and write one trace per GPU into <dir> (profile.sh, profile_kernels.py).
"""
import os
import random
import time

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

kw = {}
if os.environ.get("BACKEND"):
    kw["attention_config"] = {"backend": os.environ["BACKEND"]}
PROFILE_DIR = os.environ.get("TORCH_PROFILE_DIR")
if PROFILE_DIR:
    kw["profiler_config"] = {
        "profiler": "torch",
        "torch_profiler_dir": os.path.abspath(PROFILE_DIR),
        "torch_profiler_with_stack": False,
        "torch_profiler_use_gzip": False,
    }
llm = LLM(
    os.environ.get("MODEL", "Qwen/Qwen3-0.6B"),
    dtype=os.environ.get("DTYPE", "float32"),
    tensor_parallel_size=int(os.environ.get("TP", "2")),
    max_num_seqs=32,
    max_model_len=8448,
    enable_prefix_caching=False,
    disable_custom_all_reduce=os.environ.get("NO_CAR") == "1",
    kv_cache_memory_bytes=int(os.environ.get("KV", "1100000000")),
    compilation_config={"max_cudagraph_capture_size": 64},
    **kw,
)
rng = random.Random(0)
one = SamplingParams(max_tokens=1)


def prompts(n, length):
    return [TokensPrompt(prompt_token_ids=[rng.randint(1000, 100000) for _ in range(length)]) for _ in range(n)]


def timed(n, length, reps=3):
    llm.generate(prompts(n, length), one, use_tqdm=False)  # warmup at this shape
    batches = [prompts(n, length) for _ in range(reps)]
    if PROFILE_DIR:
        llm.start_profile(f"prefill-{n}x{length}")
    best = None
    for p in batches:
        t = time.time()
        llm.generate(p, one, use_tqdm=False)
        dt = time.time() - t
        best = dt if best is None else min(best, dt)
    if PROFILE_DIR:
        llm.stop_profile()
    print(f"PREFILL {n}x{length}: {best * 1000:7.1f} ms, {n * length / best:7.0f} tok/s", flush=True)


shapes = os.environ.get("SHAPES", "1x256,1x1024,1x2048,1x4000,1x8192,8x1024,16x512")
for shape in shapes.split(","):
    n, length = (int(x) for x in shape.split("x"))
    timed(n, length)
