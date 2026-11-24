"""Smoke test: load a model in vLLM, generate once, and time one long prompt.

usage: [MODEL=..] [DTYPE=..] [NO_CAR=1] python smoke.py <tensor_parallel_size> [kv_cache_bytes] [max_model_len]

  MODEL   default Qwen/Qwen3-0.6B
  DTYPE   default float32 (half for Qwen3-1.7B)
  PROMPT  length of the timed prompt in tokens, default 1000 (must fit max_model_len)
  NO_CAR  1 turns vLLM's custom all-reduce off (it reset the test rig's mixed Turing pair; README.md)
"""
import os
import random
import sys
import time

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

tp = int(sys.argv[1]) if len(sys.argv) > 1 else 1
llm = LLM(
    os.environ.get("MODEL", "Qwen/Qwen3-0.6B"),
    dtype=os.environ.get("DTYPE", "float32"),
    tensor_parallel_size=tp,
    max_num_seqs=32,
    max_model_len=int(sys.argv[3]) if len(sys.argv) > 3 else 2048,
    kv_cache_memory_bytes=int(sys.argv[2]) if len(sys.argv) > 2 else 600_000_000,
    enable_prefix_caching=False,
    disable_custom_all_reduce=os.environ.get("NO_CAR") == "1",
    compilation_config={"max_cudagraph_capture_size": 64},
)

sp = SamplingParams(max_tokens=128, temperature=0)
prompt = "Explain in two sentences what PCIe peer-to-peer between two GPUs is."
llm.generate([prompt], sp, use_tqdm=False)  # warmup
t = time.time()
out = llm.generate([prompt], sp, use_tqdm=False)
dt = time.time() - t
o = out[0].outputs[0]
print("TEXT:", o.text.strip()[:400])
print(f"SMOKE tp={tp} tokens={len(o.token_ids)} seconds={dt:.2f} tok/s={len(o.token_ids) / dt:.1f}")

# Prompt processing: a fresh random prompt each time (a repeated prompt would hit vLLM's prefix cache).
n = int(os.environ.get("PROMPT", "1000"))
rng = random.Random(0)


def fresh():
    return TokensPrompt(prompt_token_ids=[rng.randint(1000, 100000) for _ in range(n)])


llm.generate([fresh()], SamplingParams(max_tokens=1), use_tqdm=False)
t = time.time()
llm.generate([fresh()], SamplingParams(max_tokens=1), use_tqdm=False)
dt = time.time() - t
print(f"PREFILL tp={tp} tokens={n} seconds={dt:.2f} tok/s={n / dt:.0f}")
