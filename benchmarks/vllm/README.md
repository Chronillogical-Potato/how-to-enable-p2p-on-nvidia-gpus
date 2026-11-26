# vLLM on two GPUs: P2P off vs on

Does GPU-to-GPU P2P make a real language model faster? This folder runs **vLLM** with **tensor parallel 2** (every
layer split across both cards, which then exchange their partial results twice per layer) on Qwen3 models. It compares
P2P off (the exchange goes through host RAM) with P2P on, and measures the exchange on its own and inside the model.

Everything here was run on the test rig's **Turing pair** on 2026-11-24 (the RTX 2060 in the B550 board's chipset slot,
PCIe 3.0 x4, which caps every transfer at about 3.3 GB/s each way; the GTX 1650 SUPER in a CPU slot, PCIe 3.0 x16). The scripts work on any two-GPU setup from
this guide.

## Short answer

On the Turing pair: **no meaningful speedup.** P2P off -> P2P on (through NCCL), two rounds each:

| | Result |
|---|---|
| Generating tokens, one user (Qwen3-0.6B fp32 / Qwen3-1.7B fp16) | **+1 %** / **+1 %** |
| Reading the prompt (prefill), without the web server | between 4 % slower and 3 % faster, depending on prompt size |
| Time to first token through the web server | 2-9 % worse |
| NCCL's data exchange on its own, same software | **30-47 % faster** for 0.5-64 MiB |
| vLLM's own P2P all-reduce (the path behind upstream's +8-31 % in generation) | **reset the machine**, possibly because of the two different chips or the chipset slot, see [below](#vllms-custom-all-reduce-reset-the-machine) |
| Repeatability: same setting, round 1 vs round 2 | 0.1-0.2 % median difference |

The exchange itself does get much faster. It does not show in the model because the GTX 1650 SUPER computes far
slower than the RTX 2060, and with tensor parallel the fast card waits for it on every layer
([where the time goes](#where-the-time-goes)). A pair of two equal, faster cards, like the RTX 3090s, is the setup
that can show a real difference.

## Test setup

| | |
|---|---|
| GPU0 | RTX 2060 6 GB (TU104), B550 chipset slot, PCIe 3.0 x4 |
| GPU1 | GTX 1650 SUPER 4 GB (TU116), CPU slot, PCIe 3.0 x16 |
| P2P | set up with [docs/turing.md](../../docs/turing.md): 8 GB and 4 GB memory windows, the patched driver loaded live (P2P on), the CUDA shim for the two different chips. Integrity test passed before the run and again after the reset below |
| Host | Ryzen 9 3900X, 8 GB RAM, Ubuntu 26.04.1, kernel 7.0.0-44-generic, driver 615.71.09 (the fork at `e5ad35d`) |
| vLLM | 0.34.0 in its own venv, Python 3.14.4, PyTorch 2.14.0+cu130, NCCL 2.30.7, Triton 3.8.0. Attention backend `TRITON_ATTN` (FlashAttention needs RTX 30 or newer) |
| Models | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) in float32 (everything), [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) in float16 (single-user generation only). Why: [choosing the model](#choosing-the-model) |

The settings compared, as in upstream's own vLLM test:

| Setting | P2P | How |
|---|---|---|
| `base` | off | `NCCL_P2P_DISABLE=1` and `--disable-custom-all-reduce`: NCCL goes through host RAM (`via SHM/direct`) |
| `nccl` | on | NCCL picks P2P by itself on this rig (`via P2P/CUMEM`; on an Intel CPU it needs `NCCL_P2P_LEVEL=PHB`, [troubleshooting](../../docs/troubleshooting.md#nccl-goes-via-shm-although-cuda-offers-peer-access)); vLLM's custom all-reduce off |
| `car` | on | as `nccl`, plus vLLM's custom all-reduce. **Not run**: it reset the machine on this pair of two different chips. `bench.sh` refuses it unless `ALLOW_CAR=1` |

NCCL's choice is recorded per run in `nccl-transport.txt`: `via SHM` for every `base` run, `via P2P/CUMEM` for every
`nccl` run. (Round 1 of the 0.6B model has no such file; it ran before the logging was fixed, with the same settings.)

## Choosing the model

With tensor parallel both cards hold the same half of every weight matrix, so the **smaller card sets the limit**:
about 3.5 GB usable on the GTX 1650 SUPER, so roughly 7 GB in total, not 10. It has to hold the weights, the KV cache
(the model's memory of the text so far), working memory and NCCL's buffers. A small model leaves the most room for long
prompts and many users at once, which is what moves the most data between the cards.

The first choice, Qwen3-1.7B in float16, turned out to be the wrong test. `gemm.py`, matrix multiply in TFLOPS:

| | float16 | float32 |
|---|---|---|
| RTX 2060 | 19-24 | 5.2-5.9 |
| GTX 1650 SUPER | **0.53** | 2.7 |

The GTX 1650 SUPER has no tensor cores, and every fp16 path in cuBLAS runs at 0.53 TFLOPS on it, five times slower
than its own fp32. In float16 the model read prompts at about 285 tokens per second, so the data exchange was a tiny
part of the time. Hence **Qwen3-0.6B in float32**: it avoids that path, fits (1.2 GB per card), and in float32 each
exchange carries twice the bytes, about as many per token as Qwen3-1.7B in float16. Qwen3-1.7B stayed in for
single-user generation, where the cards mostly read weights and the slow fp16 path does not dominate.

## Results

All numbers are the mean of two rounds, the second in reverse order. `results/2026-11-24-turing-pair/` has every raw
result.

### Through the web server (`vllm bench serve`)

Random prompts of fixed length with a fixed output length (`--ignore-eos`), so both settings get exactly the same work.
P2P off -> P2P on:

| Qwen3-0.6B float32 | Time to first token | Time per output token | Output tokens/s |
|---|---|---|---|
| 1 user, 1k prompt | 483 -> 499 ms | 12.1 -> 12.0 ms | 51.4 -> 51.1 |
| 1 user, 4k prompt | 3299 -> 3379 ms | 20.0 -> 19.9 ms | 14.0 -> 13.8 |
| 1 user, 8k prompt | 10986 -> 11153 ms | 30.4 -> 30.3 ms | 5.0 -> 4.9 |
| 1 user, generating 512 tokens | 95 -> 98 ms | 10.5 -> **10.4** ms | 93.9 -> **94.9** |
| 4 users, 2k prompts | 2767 -> 2872 ms | 35.7 -> 35.9 ms | 86.3 -> 85.1 |
| 8 users, 1k prompts | 1954 -> 2092 ms | 40.8 -> 40.7 ms | 153.8 -> 152.5 |
| 16 users, short prompts | 870 -> 955 ms | 38.8 -> 38.5 ms | 380.0 -> 380.0 |
| 32 users, short prompts | 4053 -> 4161 ms | 69.8 -> 69.8 ms | 344.5 -> 343.6 |

| Qwen3-1.7B float16 | Time to first token | Time per output token | Output tokens/s |
|---|---|---|---|
| 1 user, generating 512 tokens | 920 -> 915 ms | 13.1 -> **13.0** ms | 67.3 -> **67.9** |

Full table with total throughput: [summary.txt](results/2026-11-24-turing-pair/summary.txt).

### Reading prompts, without the web server (`prefill.py`)

Best of three, Qwen3-0.6B float32:

| Prompts | P2P off | P2P on | |
|---|---|---|---|
| 1 x 256 tokens | 86.8 ms | 89.8 ms | 3.5 % slower |
| 1 x 1024 | 450 ms | 469 ms | 4 % slower |
| 1 x 2048 | 1079 ms | 1125 ms | 4 % slower |
| 1 x 4000 | 3281 ms | 3247 ms | 1 % faster |
| 1 x 8192 | 11440 ms | 11243 ms | 1.7 % faster |
| 8 x 1024 | 3455 ms | 3438 ms | 0.5 % faster |
| 16 x 512 | 2948 ms | 2863 ms | 2.9 % faster |

Long single prompts are slow either way: attention grows with the square of the prompt length (8192 tokens: 716
tokens/s, against 2,950 at 256).

### NCCL's data exchange on its own (`allreduce.py`)

Same NCCL as vLLM (2.30.7), one process per card as vLLM runs it:

| All-reduce size | P2P off | P2P on | |
|---|---|---|---|
| 4 KiB (one token while generating, 0.6B fp32) | 35.7 µs | 34.1 µs | about the same |
| 512 KiB | 312 µs | 218 µs | 30 % faster |
| 4 MiB (a 1k-token prompt) | 2.32 ms | 1.59 ms | 32 % faster |
| 64 MiB | 1.83 GB/s | **2.69 GB/s** | **+47 %** |

The same gain as the guide's NCCL benchmark with NCCL 2.32 ([docs/benchmarks.md](../../docs/benchmarks.md)): P2P works
for vLLM's NCCL too.

### Where the time goes

`profile.sh` records the three timed runs of 8 prompts x 1024 tokens with vLLM's built-in PyTorch profiler, on both
GPUs ([profile-kernels.txt](results/2026-11-24-turing-pair/profile-kernels.txt)):

| 3 runs of 8 x 1024 tokens | Model compute | Inside NCCL kernels | Idle |
|---|---|---|---|
| RTX 2060, P2P off -> on | 4215 -> 4195 ms | 6081 -> 5884 ms | 0.3 % |
| GTX 1650 SUPER, P2P off -> on | **8077 -> 7968 ms** | **2221 -> 2114 ms** | 0.3 % |

- **The GTX 1650 SUPER sets the pace.** Its half of the math takes twice as long as the RTX 2060's (fp32: 2.7 against
  5.9 TFLOPS), and it spends 79 % of its time computing.
- **The RTX 2060's NCCL time is mostly waiting** for the slow card to reach each all-reduce, not moving data. P2P
  cannot shorten that.
- **On the slow card, P2P shortened the time in NCCL by only 5 %**, although the exchange on its own is 30-47 % faster.
  The likely reason, not verified further: the fast card arrives early and sends its part of each exchange while the
  slow card is still computing, so only the slow card's own part is left on the critical path. And the route through
  host RAM costs that card little: it sits on the x16 CPU link, and the slow x4 leg is done by the RTX 2060, which is
  waiting anyway.
- **Neither GPU is idle** (0.3 %): vLLM's own work on the CPU between steps is not what limits the speed.
- **While generating**, each exchange is 4 KiB, where NCCL takes about the same time with or without P2P. vLLM's custom
  all-reduce is the P2P path built for these small exchanges, and it is the one that reset the machine on this pair of two different chips.

The same 8 x 1024 batch, timed in four separate runs: P2P off 3287-3507 ms, P2P on 3334-3438 ms. The run-to-run spread
is about as large as the difference.

**Not explained:** single prompts up to 2k tokens were 3-4 % slower with P2P, and the web server's time to first token
2-9 % worse, while output speed was the same. It looks like a small fixed cost per prompt with P2P on.

### Compared with upstream

Upstream measured vLLM 0.31 on two RTX 3080 20GB cards with Qwen3.8-27B (4-bit), at 8k-128k prompts
([results/hyperqwen-long-context.md](https://github.com/Cinderella-Man/geforce-dynbar1-p2p/blob/main/results/hyperqwen-long-context.md)):
NCCL P2P gave +7-10 % on prompts and -1 to +32 % on generation, and the custom all-reduce +8-31 % on generation (with
speculative decoding, which upstream notes makes generation numbers noisy). Two equal cards with tensor cores, a larger
model and long prompts make the data exchange a bigger share of the time.

## vLLM's custom all-reduce reset the machine

vLLM's own all-reduce, which it switches on when P2P is available, was used for two short smoke tests that finished
normally. In the next test (prompt processing, P2P on, vLLM's defaults) the machine froze and reset. The kernel log
just stops; kdump saved nothing. The next boot logged:

```
x86/amd: Previous system reset reason [0x08000800]: an uncorrected error caused a data fabric sync flood event
mce: [Hardware Error]: CPU 16: Machine Check: 0 Bank 5: bea0000000000108
```

A data fabric sync flood is the Ryzen's reaction to an error it cannot recover from on its internal interconnect: it
stops and resets the whole machine. After that, the integrity test passed again and NCCL P2P ran for about an hour
(every result above) with no Xid, AER or machine-check errors.

**The trigger, reproduced without vLLM:** the kernels of both GPUs reading each other's memory at the same time. Why
the machine cannot take it is still open, and the two different chips are one of the two suspects
([below](#still-open-two-different-chips-or-the-chipset-slot)).
[`tools/p2p_patterns.cu`](../../tools/p2p_patterns.cu) runs one access pattern at a time
([p2p-patterns.txt](results/2026-11-24-turing-pair/p2p-patterns.txt)):

| Pattern | Result |
|---|---|
| One GPU's kernels read the other's memory, either direction | clean |
| One GPU's kernels read while the other's kernels write | clean |
| Copy engines, both directions at once, reading or writing | clean, 2.77 GB/s each |
| Both GPUs' kernels read each other's memory at once, 1 to 24 blocks x 512 threads | clean |
| Both GPUs' kernels read each other's memory at once, **36 blocks x 512 threads** | **reset within a second, twice**, with the same reason and machine-check status |

vLLM's custom all-reduce has each GPU read the other's buffer on every call, with exactly that grid: 36 x 512. While
generating, a message is 4 KiB and keeps only 256 threads busy, so the smoke tests got through. The 1 MiB messages of a
prompt fill the whole grid. NCCL's P2P transport mostly **writes** into the other card's memory, the pattern the
integrity test and the NCCL benchmarks cover. The integrity test reads too, but in one direction at a time.

### Still open: two different chips, or the chipset slot?

Everything above happened on one pair, and two things about it are unusual. Either may be the real reason:

- **Two different chips** (TU104 + TU116). CUDA offers peer access between them only through the shim, which makes it
  treat both cards as the same chip, and `docs/turing.md` warns that CUDA may rely on that chip identity for more than
  the one check the shim changes.
- **GPU0 behind the B550 chipset**, where every read from the other card needs an answer back through the chipset.

Every CUDA test on this pair needs the shim, so this pair cannot tell the two apart. `tools/p2p_patterns rbi 30` on a
pair of identical cards in the same slots would: if it resets too, the slot and the read pattern are to blame for any
pair; if not, the mixed chips are.

An outside report, not verified here, points to the chipset: on an Intel B460 board with two identical RTX 4060 Ti
cards (no shim), one behind the chipset, peer writes worked both ways but every peer read came back as `0xffffffff`
([troubleshooting](../../docs/troubleshooting.md#peer-reads-return-0xffffffff-writes-work)). Reads through a consumer
chipset fail there outright; on the B550 they work until both GPUs read heavily at once. Until then, run vLLM with `--disable-custom-all-reduce` on a setup like this one
([troubleshooting](../../docs/troubleshooting.md#the-machine-resets-data-fabric-sync-flood)).

## Running it yourself

You need P2P working first ([docs/procedure.md](../../docs/procedure.md), or [docs/turing.md](../../docs/turing.md) for
Turing cards) and the integrity test passed. With the live load, the desktop is stopped, which also frees RAM: 8 GB
was enough here.

### Install vLLM

In its own venv, so nothing touches the system Python. Triton, which vLLM uses, compiles a small C helper at run time and
needs the Python headers (without them: `fatal error: Python.h: No such file or directory`):

```bash
sudo apt install -y python3-venv python3-dev
python3 -m venv ~/vllm-venv
~/vllm-venv/bin/pip install uv
~/vllm-venv/bin/uv pip install --python ~/vllm-venv/bin/python vllm==0.34.0
```

The downloads and the finished venv take about 9 GB each. Then the models (1.5 and 4 GB):

```bash
~/vllm-venv/bin/hf download Qwen/Qwen3-0.6B
~/vllm-venv/bin/hf download Qwen/Qwen3-1.7B
```

### Two different Turing chips

CUDA offers peer access only through the shim ([docs/turing.md, T5](../../docs/turing.md#t5-cuda-and-the-integrity-test)).
Export it in the shell you run the scripts from; vLLM's processes inherit it. With your own implementation values:

```bash
export LD_PRELOAD=$HOME/how-to-enable-p2p-on-nvidia-gpus/tools/turing/libchipid-shim.so
export P2P_CHIP_IMPL_MAP=0x160:0x8:0x4
```

Without it NCCL uses host RAM in both settings, and `bench.sh` prints a warning after the `nccl` run.

### Run

```bash
cd ~/how-to-enable-p2p-on-nvidia-gpus/benchmarks/vllm
NO_CAR=1 ~/vllm-venv/bin/python smoke.py 2 1100000000 2048
```

Expected: a line of text, then `SMOKE tp=2 ... tok/s=...` and `PREFILL tp=2 ...` (here 111 and 2,188 tokens/s). Then the
whole benchmark, about 45 minutes, and the summary:

```bash
./run-all.sh
```

```bash
python3 analyze.py results/local
```

The profile, with vLLM's built-in PyTorch profiler (no other tools needed; about 3 minutes):

```bash
./profile.sh
```

Single pieces: `./bench.sh base run1` or `./bench.sh nccl run1` for one setting, `~/vllm-venv/bin/python gemm.py` for
each card's matrix-multiply speed, `NCCL_P2P_DISABLE=1 ~/vllm-venv/bin/python allreduce.py` (and `=0`) for the exchange
alone.

### Memory settings

The defaults fit the 4 GB card. Each is an environment variable of `bench.sh`:

| Variable | Default | What it sets |
|---|---|---|
| `KV_BYTES` | `1100000000` | KV cache per card, in bytes (`--kv-cache-memory-bytes`). 1.1 GB gave about 9,500 tokens for Qwen3-0.6B in float32 |
| `MAXLEN` | `8448` | longest prompt plus output (`--max-model-len`); the 8k test needs 8256 |
| `MODEL`, `DTYPE` | `Qwen/Qwen3-0.6B`, `float32` | `run-all.sh` also runs `Qwen/Qwen3-1.7B` in `half` with `MAXLEN=2048 KV_BYTES=600000000` |

`bench.sh` also caps CUDA graphs at batch size 64 and requests at 32. With the defaults, CUDA graphs alone took 1 GB
per card on the 4 GB card and it ran out of memory. On cards with 10 GB or more, raise `KV_BYTES` and use a larger
model. With tensor cores on both cards (RTX 20 except GTX 16, RTX 30), float16 is the normal choice.

## Files

| File | What it does |
|---|---|
| [bench.sh](bench.sh) | one setting: starts `vllm serve` with tensor parallel 2, sends the request suite, records NCCL's transport |
| [run-all.sh](run-all.sh) | the full run: both models, both settings, two rounds in alternating order, then `prefill.py` and `allreduce.py` |
| [smoke.py](smoke.py) | loads a model, generates once, times one prompt |
| [prefill.py](prefill.py) | prompt processing without the web server, several prompt sizes |
| [allreduce.py](allreduce.py) | NCCL all-reduce alone, 4 KiB to 64 MiB, one process per card |
| [gemm.py](gemm.py) | matrix-multiply speed per card in fp16 and fp32 |
| [profile.sh](profile.sh), [profile_kernels.py](profile_kernels.py) | profile of one prompt batch with vLLM's PyTorch profiler, split into model compute, NCCL and idle per card |
| [analyze.py](analyze.py) | the P2P off vs on table from `bench.sh` results |
| [results/2026-11-24-turing-pair/](results/2026-11-24-turing-pair) | the test rig's results: every `vllm bench serve` JSON, NCCL's transport per run, and the outputs of the other scripts |

Logs (`server.log`, `nccl.*.log`) and profiler traces stay in `results/local`, which git ignores: the logs contain the
machine's hostname and paths, and the traces are large.
