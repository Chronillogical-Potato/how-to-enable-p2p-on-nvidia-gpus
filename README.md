# How to enable P2P on NVIDIA GeForce GPUs (Ubuntu 26.04)

A step-by-step guide to **GPU-to-GPU peer-to-peer (P2P) over PCIe** on consumer GeForce cards (tested on two RTX
3090s, on two RTX 3080s, and, with extra steps, on an RTX 2060 + GTX 1650 SUPER) with Ubuntu 26.04, the
Ubuntu-packaged NVIDIA 615 driver, and CUDA 13.5.

The stock GeForce driver reports no peer access (`cudaDeviceCanAccessPeer` = 0, `nvidia-smi topo -p2p r` = `GNS`), so
CUDA, NCCL, PyTorch and vLLM stage every GPU-to-GPU copy through host RAM. This guide swaps NVIDIA's open **kernel
modules** for a patched build of the same version:
[Cinderella-Man/geforce-dynbar1-p2p](https://github.com/Cinderella-Man/geforce-dynbar1-p2p).
That is a fork of [noon-at-cgn/geforce-dynbar1-p2p](https://github.com/noon-at-cgn/geforce-dynbar1-p2p) (615.71.09 plus
a dynamic BAR1 P2P patch series) with the two fixes Ubuntu 26.04 / Linux 7.0 needs. The userspace driver (libcuda,
nvidia-smi) stays Ubuntu's stock package. Copies then go card to card.

## Documents

| | |
|---|---|
| [docs/procedure.md](docs/procedure.md) | **The procedure.** From a fresh Ubuntu 26.04 install to P2P working, with a check after each step. |
| [docs/turing.md](docs/turing.md) | **Turing cards (RTX 20 / GTX 16), experimental.** The extra steps they need: a bigger memory window at every boot, P2P switched on per card, and a CUDA workaround for two different chips. |
| [docs/pascal-and-volta.md](docs/pascal-and-volta.md) | **Pascal (GTX 10, P106) and Volta (V100)**, on NVIDIA's stock driver 580. Two V100s work once the IOMMU is in passthrough mode (without it, data is silently lost). Pascal: CUDA 12 and PyTorch verified, P2P not yet. |
| [docs/troubleshooting.md](docs/troubleshooting.md) | Each failure hit on the way, with its cause and fix. The big one: Ubuntu's udev rules reloading the stock driver every few seconds. |
| [docs/benchmarks.md](docs/benchmarks.md) | P2P on vs off with NVIDIA's tools (`p2pBandwidthLatencyTest`, `nvbandwidth`, NCCL): how to run them, and the test rig's numbers. |
| [docs/security-review.md](docs/security-review.md) | What is in the fork beyond NVIDIA's official source, and whether it is safe to load. |
| [docs/test-rig-session.md](docs/test-rig-session.md) | What was actually done on the test rig, reconstructed from shell history and logs, dead ends included. |
| [tools/](tools) | Three small CUDA programs (peer access matrix, P2P bandwidth, host-GPU bandwidth), the two Turing helpers in [tools/turing](tools/turing), plus NVIDIA's `cuda-samples`, `nvbandwidth`, `nccl` and `nccl-tests` as pinned submodules. |

## The procedure in short

For RTX 30 cards (GA102). Turing cards need the extra steps in [docs/turing.md](docs/turing.md). Pascal and Volta cards
use NVIDIA's stock driver instead: [docs/pascal-and-volta.md](docs/pascal-and-volta.md).

1. BIOS: Above 4G Decoding and Resizable BAR on (BAR1 must be at least VRAM), Secure Boot off.
2. Ubuntu 26.04, then **`nvidia-driver-615-open`**. Not the `595` that `ubuntu-drivers` recommends. Hold the 615
   packages.
3. Kernel command line `iommu=pt` (IOMMU passthrough; Intel: `intel_iommu=on iommu=pt`). CUDA Toolkit 13.5 (`cuda-toolkit-13-5` only).
4. Clone the fork (pinned to `e5ad35d`), install `pahole`, run `make modules`.
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
| Tesla V100 pair (another machine, NVIDIA's stock driver 580, no patch; [docs/pascal-and-volta.md](docs/pascal-and-volta.md)) | **works** with the IOMMU in passthrough mode: integrity test passed (264 cases). Without passthrough, CUDA reported P2P while the IOMMU silently dropped the data |
| Pascal: GTX 1070 Ti on the test rig, 9x P106-090 on a mining board (stock driver 580) | CUDA 12.4 and PyTorch 2.14.1 **verified** on the GTX 1070 Ti. P2P **not verified**: the driver can be told to allow it, but the mining board's switches and chipset reject the traffic |
| Boot install: P2P on at every boot, with Ubuntu's default udev rules and `nvidia-persistenced` | **verified** over two reboots |
| Kernel-update and rollback steps (17-18) | not run yet |
| Security review of everything in the fork that is not NVIDIA's code | [no malicious code found](docs/security-review.md) |

Test rig: Ryzen 9 3900X on a B550 board (one card in the chipset x4 slot), Ubuntu 26.04.1, kernel 7.0.0-44-generic,
driver 615.71.09, CUDA 13.5. Cards: 2x RTX 3090 (24 GiB, 32 GiB BAR1); then an RTX 3080 12 GB + RTX 3080 10 GB (16 GiB
BAR1 each) in the same slots, both GA102; then an RTX 2060 6 GB (TU104, 8 GiB BAR1) + GTX 1650 SUPER 4 GB (TU116,
4 GiB BAR1), see [docs/turing.md](docs/turing.md).

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
no faster to 32 % faster, depending on prompt length.) Absolute speeds on this rig are also held back by one card sitting in a slow x4 slot; with
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
- [cmp50hx-unlock](https://github.com/xrip/cmp50hx-unlock): the earlier same-chip Turing P2P report (TU102), with a
  different route to large windows.
- [Ubsefor](https://github.com/Ubsefor/open-gpu-kernel-modules/commit/0e23c574b430c27d7b5ae394bc6ca488aff346cd): the
  CXL build fix for Linux 7.0.
- [NVIDIA open-gpu-kernel-modules](https://github.com/NVIDIA/open-gpu-kernel-modules): the driver, dual MIT/GPLv2.
