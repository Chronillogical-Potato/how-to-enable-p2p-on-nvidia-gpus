# How to enable P2P on NVIDIA GeForce GPUs (Ubuntu 26.04)

A step-by-step guide to **GPU-to-GPU peer-to-peer (P2P) over PCIe** on consumer GeForce cards (tested on two RTX
3090s, on two RTX 3080s, and, with extra steps, on an RTX 2060 + GTX 1650 SUPER and on an RTX 2060 + RTX 3060, two
generations) with Ubuntu 26.04, the
Ubuntu-packaged NVIDIA 615 driver, and CUDA 13.5.

The stock GeForce driver reports no peer access (`cudaDeviceCanAccessPeer` = 0, `nvidia-smi topo -p2p r` = `GNS`), so
CUDA, NCCL, PyTorch and vLLM stage every GPU-to-GPU copy through host RAM. This guide swaps NVIDIA's open **kernel
modules** for a patched build of the same version:
[Cinderella-Man/geforce-dynbar1-p2p](https://github.com/Cinderella-Man/geforce-dynbar1-p2p).
That is a fork of [noon-at-cgn/geforce-dynbar1-p2p](https://github.com/noon-at-cgn/geforce-dynbar1-p2p) (615.71.09 plus
a dynamic BAR1 P2P patch series) with the two fixes Ubuntu 26.04 / Linux 7.0 needs. The userspace driver (libcuda,
nvidia-smi) stays Ubuntu's stock package. Copies then go card to card.

## Which path is yours

| Your two cards | Chip | Driver | Follow | Tested here |
|---|---|---|---|---|
| RTX 3080, 3080 Ti, 3090, 3090 Ti | GA102 (Ampere) | the patched open driver, 615.71.09 | [procedure.md](docs/procedure.md) | **verified**: 2x RTX 3090; RTX 3080 12 GB + 10 GB |
| RTX 20 series, GTX 16 series | Turing | the patched driver, plus a bigger memory window at every boot and P2P switched on per card | [procedure.md](docs/procedure.md), then [turing.md](docs/turing.md) | **works, experimental**: RTX 2060 + GTX 1650 SUPER |
| GTX 10 series, P106 / P104 mining cards | Pascal | NVIDIA's stock driver 580, no patch | [pascal-and-volta.md](docs/pascal-and-volta.md) | **verified**: 2x P106-090 |
| Tesla V100 | Volta | NVIDIA's stock driver 580, no patch | [pascal-and-volta.md](docs/pascal-and-volta.md) | **verified**: 2x V100 PCIe 32 GB, on another machine |
| Two generations: RTX 2060 + RTX 3060 | Turing + Ampere | the patched driver; then a patched copy of `libcuda` (CUDA peer access), or the fork's key for writes through the other card's memory window | [procedure.md](docs/procedure.md), [turing.md](docs/turing.md) T1-T3, then [mixed-generations.md](docs/mixed-generations.md) | **works, experimental**: RTX 2060 + RTX 3060 |
| Tesla V100 + a GeForce card | Volta + Ampere | NVIDIA's stock driver 580 with one opt-in `nvidia-drm` option: **memory sharing through a dma-buf, not CUDA P2P** (Vulkan programs only) | [volta-dmabuf.md](docs/volta-dmabuf.md) | **works, experimental**: V100 + RTX 3090 |
| RTX 3050 to 3070 Ti, RTX 40 series | other Ampere chips, Ada | the patched driver; P2P opt-in per card upstream | [procedure.md](docs/procedure.md), then upstream's [opt-in](https://github.com/Cinderella-Man/geforce-dynbar1-p2p#opt-in-for-other-chips) | not tested |
| RTX 50 series | Blackwell | upstream leaves these to NVIDIA's own driver, which offers BAR1 P2P (untested upstream too) | upstream's [README](https://github.com/Cinderella-Man/geforce-dynbar1-p2p) | not tested |

Two cards of **different generations** (say an RTX 3090 with an RTX 2080) get no P2P from CUDA as it is: CUDA itself
refuses peer access between architectures. Two ways around it worked on an RTX 2060 + RTX 3060, byte-exact:
[docs/mixed-generations.md](docs/mixed-generations.md). Two different cards of one generation can work: an RTX 3080
12 GB + 10 GB (one chip, GA102), and an RTX 2060 + GTX 1650 SUPER (two different chips, with a workaround for CUDA).
A Tesla V100 with a GeForce card can use neither, because the V100 needs NVIDIA's closed driver: there the two cards
share video memory through a dma-buf instead, which is not CUDA P2P and works for Vulkan programs only
([docs/volta-dmabuf.md](docs/volta-dmabuf.md)).

Whatever the cards, the **motherboard** decides whether the traffic gets through and how fast: both cards on CPU lanes
is best, a chipset slot works with limits, cheap PCIe switches do not work. What is known per board, and how to check
yours: [docs/boards.md](docs/boards.md).

## Documents

| | |
|---|---|
| [docs/procedure.md](docs/procedure.md) | **The procedure.** From a fresh Ubuntu 26.04 install to P2P working, with a check after each step. |
| [docs/turing.md](docs/turing.md) | **Turing cards (RTX 20 / GTX 16), experimental.** The extra steps they need: a bigger memory window at every boot, P2P switched on per card, and a CUDA workaround for two different chips. |
| [docs/mixed-generations.md](docs/mixed-generations.md) | **Two generations: RTX 2060 (Turing) + RTX 3060 (Ampere), experimental.** CUDA refuses P2P between architectures; two ways around it, with every step: a patched copy of `libcuda` (CUDA peer access, so NCCL and PyTorch use P2P), or writes through the other card's memory window with the fork's `RMPeerIoMemBar1` key. |
| [docs/volta-dmabuf.md](docs/volta-dmabuf.md) | **Memory sharing between a Tesla V100 and an RTX 3090, experimental.** Not CUDA P2P: one opt-in option in `nvidia-drm` on NVIDIA's own driver 580 lets Vulkan on one card import the other's video memory through a dma-buf. Byte-exact, as fast as CUDA P2P in the same slots, and llama.cpp runs Qwen tensor-parallel across both cards with an all-reduce through the shared memory. |
| [docs/pascal-and-volta.md](docs/pascal-and-volta.md) | **Pascal (GTX 10, P106) and Volta (V100)**, on NVIDIA's stock driver 580, no patch. Both get P2P with the IOMMU in passthrough mode, on a board that routes the traffic: two V100s, and two P106-090s in the test rig. Without passthrough, data is silently lost. |
| [docs/boards.md](docs/boards.md) | **Boards and slots**: what is known per motherboard (CPU lanes, chipset slots, mining-board switches), which traffic gets through, and how to check yours. |
| [docs/troubleshooting.md](docs/troubleshooting.md) | Each failure hit on the way, with its cause and fix. The big one: Ubuntu's udev rules reloading the stock driver every few seconds. |
| [docs/benchmarks.md](docs/benchmarks.md) | P2P on vs off with NVIDIA's tools (`p2pBandwidthLatencyTest`, `nvbandwidth`, NCCL): how to run them, and the test rig's numbers. |
| [benchmarks/vllm](benchmarks/vllm/README.md) | **A language model on both cards**: vLLM with tensor parallel 2 on Qwen3, P2P off vs on, with every script and raw result. On the Turing pair P2P made generation 1 % faster, and vLLM's custom all-reduce reset the machine (possibly because of the two different chips, or the card behind the chipset). |
| [docs/security-review.md](docs/security-review.md) | What is in the fork beyond NVIDIA's official source, and whether it is safe to load. |
| [docs/test-rig-session.md](docs/test-rig-session.md) | What was actually done on the test rig, reconstructed from shell history and logs, dead ends included. |
| [tools/](tools) | Six small CUDA programs (peer access matrix, P2P bandwidth, host-GPU bandwidth, peer access patterns, data checked, that find traffic a pair cannot carry, P2P through I/O memory for pairs CUDA will not pair, and a copy through system memory as the baseline), a Vulkan test of memory sharing through a dma-buf, the two Turing helpers in [tools/turing](tools/turing), plus NVIDIA's `cuda-samples`, `nvbandwidth`, `nccl` and `nccl-tests`, aikitoria's `open-gpu-kernel-modules` (for its `libcuda` patch), and the two forks of [docs/volta-dmabuf.md](docs/volta-dmabuf.md) (`nvidia-drm` 580 with `peer_vidmem_dmabuf`, llama.cpp with the all-reduce through shared video memory) as pinned submodules. |

## The procedure in short

For RTX 30 cards (GA102). Turing cards need the extra steps in [docs/turing.md](docs/turing.md). Pascal and Volta cards
use NVIDIA's stock driver instead: [docs/pascal-and-volta.md](docs/pascal-and-volta.md).

1. BIOS: Above 4G Decoding and Resizable BAR on (BAR1 must be at least VRAM), Secure Boot off.
2. Ubuntu 26.04, then **`nvidia-driver-615-open`**. Not the `595` that `ubuntu-drivers` recommends. Hold the 615
   packages.
3. Kernel command line `iommu=pt` (IOMMU passthrough; Intel: `intel_iommu=on iommu=pt`). CUDA Toolkit 13.5 (`cuda-toolkit-13-5` only).
4. Clone the fork (pinned to `25c929e`), install `pahole`, run `make modules`.
5. Set `RMForceStaticBar1=0` for each card in modprobe.d. Set the card addresses and build path in `contrib/config.sh`.
6. **Disable the `remove` rules in udev's `71-nvidia.rules` and mask `nvidia-persistenced`.** Otherwise udev puts the
   stock driver back right after the swap.
7. Stop the desktop, run `p2p-load.sh`, run `p2p-check.sh --integrity`. The result must be `ALL PASS`.
8. Keep the NVIDIA driver out of the initramfs (needed on a fresh 26.04 install), then run `p2p-install.sh` and
   reboot, so the patched modules load at every boot. Then put the udev rules and
   `nvidia-persistenced` back to default and reboot once more.

## Status

| | |
|---|---|
| Whole procedure on a **fresh Ubuntu 26.04.1 install**, followed in order by a reader with only this repository (blind run) | **verified** 2026-11-23, after two fixes that are now in the docs |
| Whole procedure on the original test rig (24.04 upgraded to 26.04) | **verified** 2026-11-23 |
| Live load + integrity test (264 cases, 1.4 TiB byte-exact, both directions) | **passed** on both RTX 30 pairs |
| RTX 3080 pair (a 12 GB and a 10 GB card, swapped in for the 3090s) | **works**: P2P on at boot, integrity test passed twice, all benchmarks run twice with the same results, same speeds as the 3090s |
| Turing pair: RTX 2060 + GTX 1650 SUPER, two different chips ([docs/turing.md](docs/turing.md)) | **works, experimental**: memory windows grown from 256 MB to 8 GB and 4 GB at every boot, P2P on at boot, integrity test passed twice (240 cases), benchmarks run twice, same speeds as the RTX 30 pairs. CUDA needs a small preload shim for two different chips |
| Two generations: RTX 2060 + RTX 3060, Turing + Ampere ([docs/mixed-generations.md](docs/mixed-generations.md)) | **works, experimental**, trial load only: with a patched copy of `libcuda`, CUDA peer access both ways, integrity test passed (264 cases, 1.4 TiB byte-exact) and NCCL `via P2P` (all-reduce 2.72 vs 1.83 GB/s); without it, as root, writes through the other card's memory window with the fork's `RMPeerIoMemBar1` key, byte-exact both ways, a 4-byte copy in 1.8 µs instead of 13.5-18 µs |
| Tesla V100 pair (another machine, NVIDIA's stock driver 580, no patch; [docs/pascal-and-volta.md](docs/pascal-and-volta.md)) | **works** with the IOMMU in passthrough mode: integrity test passed (264 cases). Without passthrough, CUDA reported P2P while the IOMMU silently dropped the data |
| Memory sharing: Tesla V100 + RTX 3090, Volta + Ampere ([docs/volta-dmabuf.md](docs/volta-dmabuf.md)) | **works, experimental**, not CUDA P2P: NVIDIA's driver 580 with an opt-in `nvidia-drm` option, loaded by hand. 20 x 1 GiB byte-exact; 2.96 / 3.26 GB/s one way and 5.62 GB/s both ways at once, as fast as CUDA P2P in those slots; llama.cpp tensor-parallel Qwen3.8-27B bit-identical to stock, prompts 18 % faster, generation 28.9 tokens/s (27.9 with upstream's all-reduce, 23.3 split by layer) |
| Pascal: 2x P106-090 in the test rig (NVIDIA's stock driver 580, no patch, no override; [docs/pascal-and-volta.md](docs/pascal-and-volta.md)) | **works**: integrity test passed (264 cases, 1.4 TiB byte-exact). CUDA 12.4 and PyTorch 2.14.1 verified on a GTX 1070 Ti. A P106-090 mining board could not carry P2P: its switches and chipset reject the traffic |
| Boot install: P2P on at every boot, with Ubuntu's default udev rules and `nvidia-persistenced` | **verified** over two reboots |
| vLLM end to end on the Turing pair (Qwen3, tensor parallel 2, [benchmarks/vllm](benchmarks/vllm/README.md)) | **run**: NCCL P2P made generation 1 % faster and prompts no faster, because the GTX 1650 SUPER's compute sets the pace. vLLM's custom all-reduce reset the machine; use `--disable-custom-all-reduce` there. Possibly caused by the two different chips or the chipset slot, see the next row |
| Both GPUs' kernels reading each other's memory at once (Turing pair, [troubleshooting](docs/troubleshooting.md#the-machine-resets-data-fabric-sync-flood)) | **resets the machine** at 36 x 512 threads (reproduced twice with `tools/p2p_patterns`, no vLLM); clean up to 24 x 512. One direction at a time, and copy engines both ways, are clean. Seen only on this pair of two different chips (which need the CUDA shim), with GPU0 behind the chipset: either may be the reason. Identical cards in the same slots: not tested yet. An outside report with identical cards on an Intel B460 board saw every read through the chipset come back as `0xffffffff` ([troubleshooting](docs/troubleshooting.md#peer-reads-return-0xffffffff-writes-work)), which points to the chipset |
| Kernel-update and rollback steps (17-18) | not run yet |
| Security review of everything in the fork that is not NVIDIA's code | [no malicious code found](docs/security-review.md) |

Test rig: Ryzen 9 3900X on a B550 board (one card in the chipset x4 slot), Ubuntu 26.04.1, kernel 7.0.0-44-generic,
driver 615.71.09, CUDA 13.5. Cards: 2x RTX 3090 (24 GiB, 32 GiB BAR1); then an RTX 3080 12 GB + RTX 3080 10 GB (16 GiB
BAR1 each) in the same slots, both GA102; then an RTX 2060 6 GB (TU104, 8 GiB BAR1) + GTX 1650 SUPER 4 GB (TU116,
4 GiB BAR1), see [docs/turing.md](docs/turing.md); then two P106-090 mining cards (GP106, Pascal) on NVIDIA's stock driver
580, see [docs/pascal-and-volta.md](docs/pascal-and-volta.md).

### What P2P changed on that rig

Without P2P, every transfer between the two cards makes a detour: GPU -> the computer's main memory -> the other GPU.
With P2P, one card writes straight into the other's memory. Without P2P -> **with P2P**:

| | RTX 3090 pair | RTX 3080 pair | |
|---|---|---|---|
| Delay for one GPU to send a copy to the other | 19 -> **1.5 µs** | 19 -> **1.4 µs** | about 13x less |
| Both GPUs sending to each other at the same time | 3.49 -> **5.52 GB/s** | 3.50 -> **5.46 GB/s** | +56-58 % |
| Syncing a model split across both GPUs: they add up their partial results ¹ | 1.88 -> **2.75 GB/s** | 1.85 -> **2.72 GB/s** | +46-47 % |
| Collecting each other's data, so each GPU has a full copy ² | 1.61 -> **2.74 GB/s** | 1.57 -> **2.71 GB/s** | +70-73 % |
| Errors in 1.5 TB of GPU-to-GPU test copies | **0** | **0** | |

The latency figures are the average of both directions.

¹ NCCL `all_reduce`: when one model runs split across two GPUs, each computes half of every layer, and after the layer
the two halves are added up so both GPUs have the result. ² NCCL `all_gather`: each GPU holds a different chunk (for
example its half of the weights), and afterwards each has all of it. NCCL is the library that PyTorch, vLLM and most
multi-GPU software use to move data between GPUs.

These are speeds of the data exchange itself, **not** of a whole AI workload: how much faster a model runs depends on
how much of its time it spends exchanging data. (On upstream's own two-GPU machine, vLLM generated tokens anywhere from
no faster to 32 % faster, depending on prompt length. On this rig's Turing pair it was 1 % faster:
[benchmarks/vllm](benchmarks/vllm/README.md).) Absolute speeds on this rig are also held back by one card sitting in a slow x4 slot; with
both cards on CPU lanes they are several times higher. Full numbers and how to measure them:
[docs/benchmarks.md](docs/benchmarks.md).

## Disclaimer

This loads an **experimental, patched kernel driver**. It can hang the machine or corrupt GPU memory. The upstream
project has verified it only on RTX 3080 20GB pairs; other GA102 cards are "on by default, untested" there, and Turing
cards are opt-in. Use at your own risk, run the integrity test on your own pair, and keep a way back to stock
(`p2p-fallback.sh`).

## Credits

- [noon-at-cgn/geforce-dynbar1-p2p](https://github.com/noon-at-cgn/geforce-dynbar1-p2p): the patched driver and its
  scripts, which [the fork](https://github.com/Cinderella-Man/geforce-dynbar1-p2p) builds on. It in
  turn builds on the P2P work of tinygrad, aikitoria, Duanyll and dingiv (see its README).
- [noon-at-cgn/geforce-bar1-resize](https://github.com/noon-at-cgn/geforce-bar1-resize): enlarges the Turing cards'
  memory window at every boot ([docs/turing.md](docs/turing.md)).
- [aikitoria/open-gpu-kernel-modules](https://github.com/aikitoria/open-gpu-kernel-modules): the `libcuda` patch
  script that lets CUDA pair two GPU generations ([docs/mixed-generations.md](docs/mixed-generations.md)).
- Christian König and the [amdgpu](https://sources.debian.org/src/linux/6.19.6-2/drivers/gpu/drm/amd/amdgpu/amdgpu_dma_buf.c)
  developers: the dma-buf technique for sharing video memory between devices ([docs/volta-dmabuf.md](docs/volta-dmabuf.md)).
- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp): tensor parallelism, and the Vulkan all-reduce by
  pwilkin ([PR #25051](https://github.com/ggml-org/llama.cpp/pull/25051)) that the shared-memory all-reduce builds on.
- [TheBlueMatt](https://github.com/TheBlueMatt/llama.cpp/commit/5eb904aaad6cc865bed2b30aad2559ef90558563): the first
  direct GPU-to-GPU Vulkan all-reduce for llama.cpp, on Intel Xe.
- [cmp50hx-unlock](https://github.com/xrip/cmp50hx-unlock): the earlier same-chip Turing P2P report (TU102), with a
  different route to large windows.
- [Ubsefor](https://github.com/Ubsefor/open-gpu-kernel-modules/commit/0e23c574b430c27d7b5ae394bc6ca488aff346cd): the
  CXL build fix for Linux 7.0.
- [NVIDIA open-gpu-kernel-modules](https://github.com/NVIDIA/open-gpu-kernel-modules): the driver, dual MIT/GPLv2.
