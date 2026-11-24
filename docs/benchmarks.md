# Benchmarks: P2P on vs off

How to measure what P2P gives you, with NVIDIA's own tools, and what it gave on the test rig with two different pairs
of cards, in the same two slots:

- **RTX 3090 pair**: two RTX 3090 24 GB (original install, 24.04 upgraded to 26.04).
- **RTX 3080 pair**: an RTX 3080 12 GB (GPU0) and an RTX 3080 10 GB (GPU1), on the fresh 26.04.1 install.
- **Turing pair**: an RTX 2060 6 GB (GPU0) and a GTX 1650 SUPER 4 GB (GPU1), set up with [turing.md](turing.md). Its
  numbers are in [their own section](#turing-pair-rtx-2060--gtx-1650-super), because the smaller cards change a few
  test sizes.

With NVIDIA's stock driver: [two Tesla V100s](pascal-and-volta.md#results-with-passthrough) on another machine, and
[two P106-090s (Pascal)](pascal-and-volta.md#p2p-on-the-test-rig-two-p106-090s-verified) in the test rig's slots.

All results below are from 2026-11-23, with the patched driver installed for boot ([procedure.md](procedure.md) step 16),
the desktop running, and the GPUs otherwise idle. Every "P2P off" number was measured with the same patched driver,
P2P switched off inside the benchmark (`p2pBandwidthLatencyTest` does both halves, NCCL takes `NCCL_P2P_DISABLE=1`).
With P2P off, data takes the same path as on the stock driver: GPU -> host RAM -> GPU.

## In plain words

- **Latency** is the delay before a transfer arrives. **Bandwidth** (GB/s) is how much data per second gets through
  once it flows.
- Without P2P, every GPU-to-GPU transfer is two trips: GPU -> the computer's main memory -> the other GPU. With P2P it
  is one trip, straight into the other card's memory.
- **NCCL** is the library PyTorch, vLLM and most multi-GPU software use to move data between GPUs. Its operations:

  | Operation | What happens | Typical use |
  |---|---|---|
  | `all_reduce` | each GPU has partial results; they are added up and every GPU gets the total | after every layer when a model is split across GPUs (tensor parallel); averaging gradients in training |
  | `all_gather` | each GPU has one piece; afterwards every GPU has all pieces | collecting split weights or outputs |
  | `reduce_scatter` | the reverse of `all_gather`: results are added up and each GPU keeps one part | sharded training (FSDP, ZeRO) |
  | `broadcast` | one GPU sends the same data to the others | handing out weights or inputs |
  | `send/recv` | one GPU sends to another | pipeline parallelism, passing activations along |

- Faster data exchange is not the same as a faster model. A workload speeds up only by the share of its time it spends
  exchanging data, so expect a smaller gain end to end than the numbers below. A real model on the Turing pair (vLLM,
  Qwen3): [benchmarks/vllm](../benchmarks/vllm/README.md).

## The test rig's PCIe layout matters more than anything

| | Slot | Link | Host <-> GPU, 3090 pair | Host <-> GPU, 3080 pair |
|---|---|---|---|---|
| GPU0 `04:00.0` | B550 chipset slot | PCIe 3.0 x4 | 2.84 / 3.34 GB/s | 2.86 / 3.36 GB/s |
| GPU1 `2b:00.0` | CPU slot | PCIe 4.0 x16 | 26.37 / 26.86 GB/s | 26.37 / 26.94 GB/s |

Every GPU0 <-> GPU1 transfer crosses GPU0's x4 chipset link, so bandwidth tops out around 3.3 GB/s each way with or
without P2P. On a board that gives both cards CPU lanes (x8/x8 or x16/x16) the bandwidth numbers scale up accordingly.
The **relative** gains below carry over; the absolute numbers do not.

## Summary

Each cell is P2P off (via host RAM) -> **P2P on**.

| What | RTX 3090 pair | RTX 3080 pair | Gain |
|---|---|---|---|
| GPU-to-GPU copy latency (`p2pBandwidthLatencyTest`) | 18.2 / 19.6 -> **1.90 / 1.06 µs** | 18.4 / 19.5 -> **1.79 / 1.00 µs** | 10-20x lower |
| Bidirectional copy bandwidth (`p2pBandwidthLatencyTest`) | 3.49 -> **5.52 GB/s** | 3.50 -> **5.46 GB/s** | +56-58 % |
| Unidirectional copy bandwidth (`p2pBandwidthLatencyTest`) | 2.80 / 2.47 -> 3.27 / 2.96 GB/s | 2.80 / 2.48 -> 3.28 / 2.91 GB/s | +17-20 % |
| NCCL all_reduce, 1 GiB, bus bandwidth | 1.88 -> **2.75 GB/s** | 1.85 -> **2.72 GB/s** | +46-47 % |
| NCCL all_gather / reduce_scatter, 1 GiB | 1.61 / 1.60 -> **2.74 / 2.72 GB/s** | 1.57 / 1.55 -> **2.71 / 2.69 GB/s** | +70-74 % |
| NCCL send/recv, 1 GiB | 1.80 -> **2.75 GB/s** | 1.79 -> **2.72 GB/s** | +52-53 % |
| NCCL all_reduce latency, 8 KiB / 512 KiB | 19.5 / 320 -> **14.6 / 214 µs** | 19.2 / 310 -> **13.5 / 214 µs** | 25-33 % faster |
| NCCL latency, tiny messages (8-512 B) | 10-12 µs either way | 10-12 µs either way | about the same |
| Integrity test (264 cases, 1.4 TiB byte-exact) | ALL PASS | ALL PASS | |

Where two numbers are given, they are GPU0 -> GPU1 / GPU1 -> GPU0.

Takeaways:

- **Latency of a GPU-to-GPU copy drops by an order of magnitude** (18-20 µs to 1-2 µs), because the copy no longer
  bounces through host RAM.
- **Anything that moves data both ways gains 46-74 %** (all_reduce, all_gather, reduce_scatter, send/recv,
  bidirectional copies). Through host RAM the slow x4 link is crossed in both directions for both halves; with P2P
  each direction crosses it once.
- **One-way traffic gains little** (unidirectional copies +17-20 %, NCCL broadcast 3.27 GB/s either way): it is capped
  by the x4 link either way.
- **Tiny NCCL messages (8-512 B) are about the same** (10-12 µs): NCCL's own fixed overhead dominates there. The
  differences between P2P and host RAM at these sizes (up to about ±1 µs) are run-to-run noise: a second run on the
  3080 pair moved the 8-byte timings by up to 0.9 µs. On the upstream X99 machine P2P was 1-2 µs slower for these.
- **The two pairs behave the same.** Card model and memory size make no difference here, and neither does a mixed
  10 GB + 12 GB pair: the x4 chipset slot sets the pace.
- NCCL 2.32 picks **P2P by default** on this topology (`via P2P/direct`); no `NCCL_P2P_LEVEL` needed.

## How repeatable these numbers are

- **RTX 3080 pair:** everything on this page was run twice. Across 44 key numbers the second run matched the first
  within 2 % for every bandwidth figure and within 3.5 % for every latency figure, except NCCL's 8-byte timings, which
  moved by up to 0.9 µs (8 %). The integrity test passed both times, with identical bandwidth (3.27 / 2.91 / 5.46 GB/s).
- **RTX 3090 pair:** the integrity test passed five times across the two installs, and `p2pBandwidthLatencyTest` gave
  the same results twice. `nvbandwidth` and NCCL ran once.
- **Turing pair:** everything was run twice, with the integrity test in between. Across the same 44 numbers (NCCL at
  512 MiB) every bandwidth figure matched within 2.1 %, except the slow kernel read of GPU0's memory (0.85 / 0.88 GB/s).
  The P2P latencies matched within 1 %, NCCL latencies within 2.8 %. The host-route copy latencies moved the most:
  39.4 / 19.3 µs, then 40.7 / 17.6 µs. The integrity test passed both times.

The numbers in the tables are from the first run on each pair.

## Getting the tools

The NVIDIA projects are git submodules of this repository, each pinned to a release and fetched shallow (one commit).
They are only downloaded when you ask for them. In the guide checkout from [procedure.md](procedure.md) step 6:

```bash
cd ~/how-to-enable-p2p-on-nvidia-gpus
git submodule update --init tools/cuda-samples tools/nvbandwidth tools/nccl tools/nccl-tests
```

| Submodule | Pinned at | Size | Used for |
|---|---|---|---|
| [`tools/cuda-samples`](https://github.com/NVIDIA/cuda-samples) | `v13.3` | ~215 MB | `p2pBandwidthLatencyTest` (removed from the samples in v13.4) |
| [`tools/nvbandwidth`](https://github.com/NVIDIA/nvbandwidth) | `v0.11.0` | ~1 MB | detailed bandwidth and pointer-chase latency |
| [`tools/nccl`](https://github.com/NVIDIA/nccl) | `v2.32.6-1` | ~20 MB | the NCCL library, built from source for CUDA 13.5 |
| [`tools/nccl-tests`](https://github.com/NVIDIA/nccl-tests) | `v2.21.1` | ~1 MB | `all_reduce_perf` and friends |

Fetch only the ones you need by naming them. Or clone everything at once with
`git clone --recurse-submodules https://github.com/Cinderella-Man/how-to-enable-p2p-on-nvidia-gpus.git`.

Everything below is compiled for the installed cards (`-arch=native`, `sm_86`). That is required: the 615.71.09 driver
supports CUDA 13.4 and the toolkit is 13.5 (see [troubleshooting](troubleshooting.md#a-cuda-program-fails-the-provided-ptx-was-compiled-with-an-unsupported-toolchain)).
For other cards use `sm_75` (Turing) or `sm_89` (Ada) in the NCCL commands.

## 1. Quick checks: this repository's `tools/`

Built in procedure step 6. Each takes seconds.

| Program | Measures | RTX 3090 pair | RTX 3080 pair |
|---|---|---|---|
| `tools/p2p_access` | `cudaDeviceCanAccessPeer` for every pair | `YES` both ways (stock driver: `NO`) | `YES` both ways |
| `tools/p2p_bandwidth` | `cudaMemcpyPeer`, 1 GiB x 20, 0 -> 1 / 1 -> 0 | 3.27 / 2.98 GB/s | 3.28 / 2.91 GB/s |
| `tools/pcie_bandwidth` | host <-> GPU, 1 GiB pinned, per card | GPU0 2.85 / 3.28, GPU1 26.18 / 26.43 GB/s | GPU0 2.87 / 3.37, GPU1 26.16 / 27.01 GB/s |

## 2. `p2pBandwidthLatencyTest` (CUDA samples)

The standard P2P benchmark. It runs every measurement twice, with peer access disabled (copies staged through host RAM)
and enabled, so one run gives you both sides.

```bash
nvcc -O2 -arch=native -I tools/cuda-samples/Common \
  tools/cuda-samples/cpp/5_Domain_Specific/p2pBandwidthLatencyTest/p2pBandwidthLatencyTest.cu \
  -o tools/p2pBandwidthLatencyTest
./tools/p2pBandwidthLatencyTest
```

It takes about a minute. How to read it:

- `P2P Connectivity Matrix`: must be `1` between the cards. With `0` both halves measure the host path.
- Diagonal entries (~830 GB/s) are each GPU copying within its own memory. Ignore them.
- `P2P=Disabled ...` is the route through host RAM, `P2P=Enabled ...` the direct route.
- Latency tables have a `GPU` half (the copy itself, timed on the GPU) and a `CPU` half (time per copy as the calling
  program sees it, launch overhead included).

Test rig:

| | 3090 pair, P2P off | 3090 pair, P2P on | 3080 pair, P2P off | 3080 pair, P2P on |
|---|---|---|---|---|
| Unidirectional, 0 -> 1 / 1 -> 0 | 2.80 / 2.47 GB/s | 3.27 / 2.96 GB/s | 2.80 / 2.48 GB/s | 3.28 / 2.91 GB/s |
| Bidirectional | 3.49 GB/s | 5.52 GB/s | 3.50 GB/s | 5.46 GB/s |
| Latency, GPU-timed, 0 -> 1 / 1 -> 0 | 18.15 / 19.62 µs | 1.90 / 1.06 µs | 18.42 / 19.52 µs | 1.79 / 1.00 µs |
| Latency, CPU-timed, 0 -> 1 / 1 -> 0 | 6.83 / 6.90 µs | 2.04 / 2.17 µs | 6.89 / 6.81 µs | 2.02 / 2.09 µs |

It does not check the copied data. That is what the integrity test (procedure step 14) is for.

## 3. `nvbandwidth`

NVIDIA's dedicated interconnect benchmark. It measures each copy pattern separately, both with the copy engines (`_ce`,
what `cudaMemcpy` uses) and with GPU kernels (`_sm`), and has pointer-chase latency tests: a kernel follows a chain of
dependent loads through the other device's memory, so you get the cost of a single remote access.

It needs CMake, and **internet during the first `cmake`**: the build downloads one dependency,
[argparse](https://github.com/p-ranav/argparse) v3.2, pinned by its SHA-256.

```bash
sudo apt install -y cmake
cd tools/nvbandwidth
cmake -DCMAKE_CUDA_ARCHITECTURES=native .
make -j"$(nproc)"
```

```bash
./nvbandwidth -t host_device_latency_sm device_to_device_latency_sm
```

```bash
./nvbandwidth          # every test; about 16 minutes on the test rig
cd ../..
```

It warns that the driver's CUDA 13.4 is below the toolkit's 13.5. That is expected and harmless with native code. The
`_tma` tests are skipped on RTX 30 cards (they need Hopper or newer). The output prints the machine's hostname, so
check before posting it. `SUM` lines add up all GPUs and directions; read the matrices above them for per-direction
numbers.

Test rig. The columns give the direction the **data** flows. Careful when reading the raw output: its arrows
(`GPU(row) <- GPU(column)` on the write tests, `GPU(row) -> GPU(column)` on the read tests) do not show the data
direction. The row is the GPU that runs the copy: write tests move data from row to column, read tests from column to
row. This was checked against `tools/p2p_bandwidth`, whose code copies GPU0 -> GPU1 and measures 3.27 GB/s that way,
and against the PCIe layout (GPU0's x4 chipset link sends faster than it receives).

| Test | 3090 pair, data 0 -> 1 / 1 -> 0 | 3080 pair, data 0 -> 1 / 1 -> 0 | Note |
|---|---|---|---|
| `device_to_device_memcpy_write_ce` | 3.27 / 2.96 GB/s | 3.27 / 2.91 GB/s | P2P copy, the source GPU pushes |
| `device_to_device_memcpy_read_ce` | 3.27 / 2.95 GB/s | 3.27 / 2.90 GB/s | P2P copy, the destination GPU pulls |
| `device_to_device_bidirectional_memcpy_write_ce` | 2.87 + 2.76 = **5.62 GB/s** | 2.89 + 2.73 = **5.62 GB/s** | both directions at once; the output's `Write1`, `Write2` and `Total` matrices |
| `device_to_device_memcpy_write_sm` | 3.27 / 2.96 GB/s | 3.27 / 2.91 GB/s | a kernel writes into the peer |
| `device_to_device_memcpy_read_sm` | **0.85** / 2.86 GB/s | **0.88** / 2.88 GB/s | a kernel reads from the peer |
| `device_to_device_latency_sm` | 1852-1857 ns | 1854-1856 ns | one dependent load from the peer's memory |

| Test | 3090 pair: GPU0 (chipset x4) / GPU1 (CPU x16) | 3080 pair: GPU0 / GPU1 |
|---|---|---|
| `host_to_device_memcpy_ce` | 2.84 / 26.37 GB/s | 2.86 / 26.37 GB/s |
| `device_to_host_memcpy_ce` | 3.34 / 26.86 GB/s | 3.36 / 26.94 GB/s |
| `host_device_latency_sm` | 1435 / 638 ns | 1449 / 635 ns |

What stands out:

- **A kernel on GPU1 reading GPU0's memory gets only 0.85-0.88 GB/s** (GPU0 is the card behind the chipset). Writes are
  fine in both directions, and so are copy-engine reads. Software that pulls data from the peer with kernels, rather
  than pushing it or using `cudaMemcpy`, pays for that. Upstream reports the same: P2P reads are much slower than
  writes.
- **A direct access to the peer's memory costs ~1.85 µs**, a bit more than GPU0 reading host RAM (1.44 µs): both go
  over the same chipset link. Without P2P there is no direct access at all, only staged copies (18-20 µs above).

## 4. NCCL and `nccl-tests`

NCCL is the library PyTorch, vLLM and most multi-GPU software use to move data between GPUs, so these numbers are the
closest to real workloads. Ubuntu's own `libnccl2` (2.22) predates CUDA 13, so build NCCL from the submodule. It stays
in the checkout; nothing is installed system-wide.

```bash
cd ~/how-to-enable-p2p-on-nvidia-gpus/tools
GEN="-gencode=arch=compute_86,code=sm_86"
make -C nccl -j"$(nproc)" src.build NVCC_GENCODE="$GEN"
make -C nccl-tests -j"$(nproc)" MPI=0 NCCL_HOME="$PWD/nccl/build" NVCC_GENCODE="$GEN"
export LD_LIBRARY_PATH="$PWD/nccl/build/lib"
```

On the test rig NCCL took 3.2-3.7 minutes to build, `nccl-tests` 1.4 minutes.

P2P on (NCCL's default here) and off:

```bash
./nccl-tests/build/all_reduce_perf -b 8 -e 1G -f 2 -g 2
```

```bash
NCCL_P2P_DISABLE=1 ./nccl-tests/build/all_reduce_perf -b 8 -e 1G -f 2 -g 2
```

The same works for `all_gather_perf`, `reduce_scatter_perf`, `broadcast_perf` and `sendrecv_perf`. Read the `busbw`
column for bandwidth and `time` (µs) for latency, out-of-place half. To see which route NCCL picked:

```bash
NCCL_DEBUG=INFO ./nccl-tests/build/all_reduce_perf -b 8 -e 8 -g 2 2>&1 | grep -m2 ' via '
```

`via P2P/direct` is P2P. `via SHM/direct` is through host RAM. For timing runs, leave `NCCL_DEBUG` off: its log lines
get mixed into the results table.

Test rig, NCCL 2.32.6 built for CUDA 13.5. NCCL used `via P2P/direct` by default on both pairs. Bus bandwidth in
GB/s, `-b 8 -e 1G -f 2`, P2P / host RAM:

| Operation | 3090 pair, 64 MiB | 3090 pair, 1 GiB | 3080 pair, 64 MiB | 3080 pair, 1 GiB |
|---|---|---|---|---|
| all_reduce | 2.73 / 1.82 | **2.75** / 1.88 | 2.71 / 1.82 | **2.72** / 1.85 |
| all_gather | 2.71 / 1.53 | **2.74** / 1.61 | 2.67 / 1.53 | **2.71** / 1.57 |
| reduce_scatter | 2.68 / 1.53 | **2.72** / 1.60 | 2.64 / 1.53 | **2.69** / 1.55 |
| send/recv | 2.75 / 1.80 | **2.75** / 1.80 | 2.72 / 1.80 | **2.72** / 1.79 |
| broadcast | 3.27 / 3.27 | 3.27 / 3.27 | 3.27 / 3.27 | 3.27 / 3.27 |

Latency in µs, `-b 8 -e 1M -f 4 -n 500 -w 50`, P2P / host RAM. RTX 3090 pair:

| Operation | 8 B | 512 B | 8 KiB | 32 KiB | 512 KiB |
|---|---|---|---|---|---|
| all_reduce | 11.9 / 12.5 | 11.9 / 12.0 | 14.6 / 19.5 | 32.5 / 44.9 | 214 / 320 |
| all_gather | - | 11.7 / 11.8 | 11.6 / 14.1 | 19.4 / 28.8 | 120 / 189 |
| reduce_scatter | - | 11.5 / 11.6 | 11.4 / 14.2 | 19.6 / 29.3 | 118 / 189 |
| send/recv | 11.0 / 11.2 | 10.8 / 11.0 | 14.1 / 19.2 | 32.0 / 52.2 | 206 / 320 |
| broadcast | 11.9 / 12.0 | 11.6 / 11.8 | 11.4 / 12.0 | 26.1 / 28.3 | 177 / 180 |

RTX 3080 pair:

| Operation | 8 B | 512 B | 8 KiB | 32 KiB | 512 KiB |
|---|---|---|---|---|---|
| all_reduce | 11.8 / 12.1 | 11.7 / 12.0 | 13.5 / 19.2 | 31.4 / 44.6 | 214 / 310 |
| all_gather | - | 11.0 / 11.7 | 10.8 / 13.8 | 19.2 / 28.1 | 120 / 188 |
| reduce_scatter | - | 11.6 / 10.7 | 11.5 / 13.9 | 19.4 / 28.8 | 118 / 186 |
| send/recv | 11.1 / 10.2 | 10.8 / 9.9 | 13.9 / 19.1 | 31.8 / 51.8 | 204 / 319 |
| broadcast | 11.0 / 11.9 | 10.9 / 11.7 | 10.8 / 11.9 | 25.5 / 27.7 | 177 / 179 |

Every run reported `#wrong` 0 and `Out of bounds values : 0 OK`.

(all_gather and reduce_scatter have no 8 B row: the smallest size they run with two GPUs is larger.)

## Turing pair: RTX 2060 + GTX 1650 SUPER

Same rig, same two slots, set up with [turing.md](turing.md). Two different Turing chips, so every CUDA program ran
with the preload shim from [turing.md step T5](turing.md#t5-cuda-and-the-integrity-test); NCCL picked `via P2P/direct`
by itself. The tools were built with `-arch=native` / `sm_75`: in the NCCL commands above use
`GEN="-gencode=arch=compute_75,code=sm_75"`. If they were built for other cards before, build from clean: `make -C nccl
clean` and `make -C nccl-tests clean`, and delete `CMakeCache.txt` and `CMakeFiles` in `tools/nvbandwidth`. Otherwise
the old architecture stays in the cached build files.

Two differences from the RTX 30 runs:

- **NCCL's largest size is 512 MiB, not 1 GiB.** `nccl-tests` lowers `-e 1G` to what fits in the 4 GB card. The RTX
  3080 pair's numbers below are at 512 MiB too, from the same runs.
- **The GTX 1650 SUPER has PCIe 3.0** (Turing has no PCIe 4.0), so its own host link is half the 3080's. It does not
  matter for GPU-to-GPU traffic, which is capped by GPU0's x4 chipset link either way.

Each cell is P2P off (via host RAM) -> **P2P on**. Where two numbers are given, they are GPU0 -> GPU1 / GPU1 -> GPU0.

| What | Turing pair | RTX 3080 pair |
|---|---|---|
| GPU-to-GPU copy latency, GPU-timed (`p2pBandwidthLatencyTest`) | 39.4 / 19.3 -> **1.86 / 1.04 µs** | 18.4 / 19.5 -> 1.79 / 1.00 µs |
| GPU-to-GPU copy latency, CPU-timed | 8.03 / 8.06 -> **2.27 / 2.34 µs** | 6.89 / 6.81 -> 2.02 / 2.09 µs |
| Bidirectional copy bandwidth | 3.26 -> **5.49 GB/s** | 3.50 -> 5.46 GB/s |
| Unidirectional copy bandwidth | 2.51 / 2.27 -> 3.28 / 2.93 GB/s | 2.80 / 2.48 -> 3.28 / 2.91 GB/s |
| NCCL all_reduce, 512 MiB, bus bandwidth | 1.89 -> **2.74 GB/s** | 1.85 -> 2.72 GB/s |
| NCCL all_gather / reduce_scatter, 512 MiB | 1.53 / 1.54 -> **2.73 / 2.70 GB/s** | 1.56 / 1.55 -> 2.69 / 2.69 GB/s |
| NCCL send/recv, 512 MiB | 1.86 -> **2.49 GB/s** | 1.79 -> 2.72 GB/s |
| NCCL broadcast, 512 MiB | 3.27 either way | 3.27 either way |
| NCCL all_reduce latency, 8 KiB / 512 KiB | 20.4 / 306 -> **13.6 / 217 µs** | 19.2 / 310 -> 13.5 / 214 µs |
| Integrity test | ALL PASS twice (240 cases, 636 GiB each) | ALL PASS twice (264 cases, 1.4 TiB each) |

`tools/`: `p2p_access` `YES` both ways (with the shim; `NO` without it), `p2p_bandwidth` 3.28 / 2.93 GB/s,
`pcie_bandwidth` GPU0 2.87 / 3.38, GPU1 13.14 / 13.55 GB/s (host -> GPU / GPU -> host).

`nvbandwidth`, data 0 -> 1 / 1 -> 0 (see section 3 for how to read the raw output):

| Test | Turing pair | RTX 3080 pair |
|---|---|---|
| `device_to_device_memcpy_write_ce` | 3.28 / 2.93 GB/s | 3.27 / 2.91 GB/s |
| `device_to_device_memcpy_read_ce` | 3.28 / 2.93 GB/s | 3.27 / 2.90 GB/s |
| `device_to_device_bidirectional_memcpy_write_ce` | 2.89 + 2.75, total **5.63 GB/s** | 2.89 + 2.73 = 5.62 GB/s |
| `device_to_device_memcpy_write_sm` | 3.28 / 2.93 GB/s | 3.27 / 2.91 GB/s |
| `device_to_device_memcpy_read_sm` | **0.85** / 2.89 GB/s | **0.88** / 2.88 GB/s |
| `device_to_device_latency_sm` | 2162 / 2301 ns | 1854-1856 ns |
| `host_to_device_memcpy_ce`, GPU0 / GPU1 | 2.87 / 13.14 GB/s | 2.86 / 26.37 GB/s |
| `device_to_host_memcpy_ce`, GPU0 / GPU1 | 3.38 / 13.55 GB/s | 3.36 / 26.94 GB/s |
| `host_device_latency_sm`, GPU0 / GPU1 | 1568 / 785 ns | 1449 / 635 ns |

NCCL latency in µs, `-b 8 -e 1M -f 4 -n 500 -w 50`, P2P / host RAM:

| Operation | 8 B | 512 B | 8 KiB | 32 KiB | 512 KiB |
|---|---|---|---|---|---|
| all_reduce | 12.2 / 12.7 | 12.8 / 12.6 | 13.6 / 20.4 | 31.4 / 46.7 | 217 / 306 |
| all_gather | - | 12.1 / 12.2 | 12.0 / 14.2 | 19.2 / 28.5 | 120 / 189 |
| reduce_scatter | - | 12.1 / 12.1 | 12.1 / 14.0 | 19.0 / 28.3 | 120 / 186 |
| send/recv | 11.7 / 11.5 | 11.5 / 11.3 | 13.1 / 20.5 | 31.5 / 49.4 | 207 / 330 |
| broadcast | 12.6 / 12.3 | 12.4 / 12.2 | 12.2 / 11.8 | 26.7 / 28.9 | 177 / 180 |

Every NCCL run reported `#wrong` 0 and `Out of bounds values : 0 OK`.

What stands out:

- **P2P on Turing performs like P2P on the RTX 3080s.** Every GPU-to-GPU bandwidth figure is within 1-2 % of the 3080
  pair: the x4 chipset link sets the pace, not the cards.
- **Without P2P, a copy from the RTX 2060 to the GTX 1650 SUPER is slow: 39-41 µs** in both rounds, twice the 3080
  pair's 18 µs; the other direction takes 18-19 µs, as on the 3080s. The cause was not investigated. With P2P both
  directions take 1-2 µs, as on the 3080s, so on this pair P2P cuts the average delay 20x (13x on the RTX 30 pairs).
- **NCCL send/recv is the one exception:** 2.49 GB/s with P2P against 2.72 GB/s on the 3080s. Still +34 % over the
  host route. NCCL tunes its protocols per GPU generation, which is the likely reason.
- **A direct access to the peer's memory costs ~2.2-2.3 µs**, about 0.4 µs more than on the 3080s.

## Reproducing on the test rig

The RTX 3080 run followed this page as written, on the fresh 26.04.1 install: the submodules arrived in 20 s, and the
builds took 3 s (`p2pBandwidthLatencyTest`), 14 s (`nvbandwidth`), 3.2 min (NCCL) and 1.4 min (`nccl-tests`).
`git status` stayed clean afterwards. The RTX 3090 run used the same pinned commits, fetched directly by SHA, because
this repository was not pushed yet. Nothing on this page changes the driver or system configuration, apart from
installing `cmake`.
