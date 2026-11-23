# Benchmarks: P2P on vs off

How to measure what P2P gives you, with NVIDIA's own tools, and what it gave on the test rig.

All results below are from 2026-11-23, with the patched driver installed for boot ([procedure.md](procedure.md) step 16),
the desktop running, and the GPUs otherwise idle. Every "P2P off" number was measured with the same patched driver,
P2P switched off inside the benchmark (`p2pBandwidthLatencyTest` does both halves, NCCL takes `NCCL_P2P_DISABLE=1`).
With P2P off, data takes the same path as on the stock driver: GPU -> host RAM -> GPU.

## The test rig's PCIe layout matters more than anything

| | Slot | Link | Host <-> GPU (measured) |
|---|---|---|---|
| GPU0 `04:00.0` | B550 chipset slot | PCIe 3.0 x4 | 2.84 / 3.34 GB/s |
| GPU1 `2b:00.0` | CPU slot | PCIe 4.0 x16 | 26.37 / 26.86 GB/s |

Every GPU0 <-> GPU1 transfer crosses GPU0's x4 chipset link, so bandwidth tops out around 3.3 GB/s each way with or
without P2P. On a board that gives both cards CPU lanes (x8/x8 or x16/x16) the bandwidth numbers scale up accordingly.
The **relative** gains below carry over; the absolute numbers do not.

## Summary

| What | P2P off (via host RAM) | P2P on | Gain |
|---|---|---|---|
| GPU-to-GPU copy latency (`p2pBandwidthLatencyTest`) | 18.2 / 19.6 µs | **1.90 / 1.06 µs** | 10-18x lower |
| Bidirectional copy bandwidth (`p2pBandwidthLatencyTest`) | 3.49 GB/s | **5.52 GB/s** | +58 % |
| Unidirectional copy bandwidth (`p2pBandwidthLatencyTest`) | 2.80 / 2.47 GB/s | 3.27 / 2.96 GB/s | +17-20 % |
| NCCL all_reduce, 1 GiB, bus bandwidth | 1.88 GB/s | **2.75 GB/s** | +46 % |
| NCCL all_gather / reduce_scatter, 1 GiB | 1.61 / 1.60 GB/s | **2.74 / 2.72 GB/s** | +70 % |
| NCCL send/recv, 1 GiB | 1.80 GB/s | **2.75 GB/s** | +53 % |
| NCCL all_reduce latency, 8 KiB / 512 KiB | 19.5 / 320 µs | **14.6 / 214 µs** | -25 % / -33 % |
| NCCL all_reduce latency, 8 B | 12.5 µs | 11.9 µs | about the same |

Where two numbers are given, they are GPU0 -> GPU1 / GPU1 -> GPU0.

Takeaways:

- **Latency of a GPU-to-GPU copy drops by an order of magnitude** (18-20 µs to 1-2 µs), because the copy no longer
  bounces through host RAM.
- **Anything that moves data both ways gains 46-70 %** (all_reduce, all_gather, reduce_scatter, send/recv,
  bidirectional copies). Through host RAM the slow x4 link is crossed in both directions for both halves; with P2P
  each direction crosses it once.
- **One-way traffic gains little** (unidirectional copies +17-20 %, NCCL broadcast 3.27 GB/s either way): it is capped
  by the x4 link either way.
- **Tiny NCCL messages (8-512 B) are about the same** (11-12 µs): NCCL's own fixed overhead dominates there. On the
  upstream X99 machine P2P was even 1-2 µs slower for these; here it is 0.1-0.6 µs faster.
- NCCL 2.32 picks **P2P by default** on this topology (`via P2P/direct`); no `NCCL_P2P_LEVEL` needed.

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

| Program | Measures | Test rig |
|---|---|---|
| `tools/p2p_access` | `cudaDeviceCanAccessPeer` for every pair | `YES` both ways (stock driver: `NO`) |
| `tools/p2p_bandwidth` | `cudaMemcpyPeer`, 1 GiB x 20, each way | 3.27 / 2.98 GB/s |
| `tools/pcie_bandwidth` | host <-> GPU, 1 GiB pinned, per card | GPU0 2.85 / 3.28, GPU1 26.18 / 26.43 GB/s |

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

| | P2P off | P2P on |
|---|---|---|
| Unidirectional, 0 -> 1 / 1 -> 0 | 2.80 / 2.47 GB/s | 3.27 / 2.96 GB/s |
| Bidirectional | 3.49 GB/s | 5.52 GB/s |
| Latency, GPU-timed, 0 -> 1 / 1 -> 0 | 18.15 / 19.62 µs | 1.90 / 1.06 µs |
| Latency, CPU-timed, 0 -> 1 / 1 -> 0 | 6.83 / 6.90 µs | 2.04 / 2.17 µs |

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

Test rig. The columns give the direction the **data** flows. Careful when reading the raw output: the write tests
print `GPU(row) -> GPU(column)`, the read tests `GPU(row) <- GPU(column)`.

| Test | data 0 -> 1 | data 1 -> 0 | Note |
|---|---|---|---|
| `device_to_device_memcpy_write_ce` | 3.27 GB/s | 2.96 GB/s | P2P copy, the source GPU pushes |
| `device_to_device_memcpy_read_ce` | 3.27 GB/s | 2.95 GB/s | P2P copy, the destination GPU pulls |
| `device_to_device_bidirectional_memcpy_write_ce` | 2.87 GB/s | 2.87 GB/s | both ways at once: 5.73 GB/s total |
| `device_to_device_memcpy_write_sm` | 3.27 GB/s | 2.96 GB/s | a kernel writes into the peer |
| `device_to_device_memcpy_read_sm` | **0.85 GB/s** | 2.86 GB/s | a kernel reads from the peer |
| `device_to_device_latency_sm` | 1852-1857 ns | 1852-1857 ns | one dependent load from the peer's memory (the two pairs differ by 5 ns) |

| Test | GPU0 (chipset x4) | GPU1 (CPU x16) |
|---|---|---|
| `host_to_device_memcpy_ce` | 2.84 GB/s | 26.37 GB/s |
| `device_to_host_memcpy_ce` | 3.34 GB/s | 26.86 GB/s |
| `host_device_latency_sm` | 1435 ns | 638 ns |

What stands out:

- **A kernel on GPU1 reading GPU0's memory gets only 0.85 GB/s** (GPU0 is the card behind the chipset). Writes are
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

On the test rig NCCL took 3.7 minutes to build, `nccl-tests` 1.4 minutes.

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

Test rig, NCCL 2.32.6 built for CUDA 13.5. Bus bandwidth, `-b 8 -e 1G -f 2`:

| Operation | P2P, 64 MiB | P2P, 1 GiB | Host RAM, 64 MiB | Host RAM, 1 GiB |
|---|---|---|---|---|
| all_reduce | 2.73 | **2.75** | 1.82 | 1.88 GB/s |
| all_gather | 2.71 | **2.74** | 1.53 | 1.61 GB/s |
| reduce_scatter | 2.68 | **2.72** | 1.53 | 1.60 GB/s |
| send/recv | 2.75 | **2.75** | 1.80 | 1.80 GB/s |
| broadcast | 3.27 | 3.27 | 3.27 | 3.27 GB/s |

Latency in µs, `-b 8 -e 1M -f 4 -n 500 -w 50` (P2P / host RAM):

| Operation | 8 B | 512 B | 8 KiB | 32 KiB | 512 KiB |
|---|---|---|---|---|---|
| all_reduce | 11.9 / 12.5 | 11.9 / 12.0 | 14.6 / 19.5 | 32.5 / 44.9 | 214 / 320 |
| all_gather | - | 11.7 / 11.8 | 11.6 / 14.1 | 19.4 / 28.8 | 120 / 189 |
| reduce_scatter | - | 11.5 / 11.6 | 11.4 / 14.2 | 19.6 / 29.3 | 118 / 189 |
| send/recv | 11.0 / 11.2 | 10.8 / 11.0 | 14.1 / 19.2 | 32.0 / 52.2 | 206 / 320 |
| broadcast | 11.9 / 12.0 | 11.6 / 11.8 | 11.4 / 12.0 | 26.1 / 28.3 | 177 / 180 |

(all_gather and reduce_scatter have no 8 B row: the smallest size they run with two GPUs is larger.)

## Reproducing on the test rig

The rig ran exactly these pinned commits, fetched directly by SHA, because this repository was not pushed yet. The
benchmark binaries stayed in the user's home directory. Nothing above changes the driver or system configuration, apart
from installing `cmake`.
