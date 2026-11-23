# How to enable P2P on NVIDIA GeForce GPUs (Ubuntu 26.04)

A step-by-step guide to **GPU-to-GPU peer-to-peer (P2P) over PCIe** on consumer GeForce cards (tested on two RTX
3090s) with Ubuntu 26.04, the Ubuntu-packaged NVIDIA 615 driver, and CUDA 13.5.

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
| [docs/troubleshooting.md](docs/troubleshooting.md) | Each failure hit on the way, with its cause and fix. The big one: Ubuntu's udev rules reloading the stock driver every few seconds. |
| [docs/benchmarks.md](docs/benchmarks.md) | P2P on vs off with NVIDIA's tools (`p2pBandwidthLatencyTest`, `nvbandwidth`, NCCL): how to run them, and the test rig's numbers. |
| [docs/security-review.md](docs/security-review.md) | What is in the fork beyond NVIDIA's official source, and whether it is safe to load. |
| [docs/test-rig-session.md](docs/test-rig-session.md) | What was actually done on the test rig, reconstructed from shell history and logs, dead ends included. |
| [tools/](tools) | Three small CUDA programs (peer access matrix, P2P bandwidth, host-GPU bandwidth), plus NVIDIA's `cuda-samples`, `nvbandwidth`, `nccl` and `nccl-tests` as pinned submodules. |

## The procedure in short

1. BIOS: Above 4G Decoding and Resizable BAR on (BAR1 must be at least VRAM), Secure Boot off.
2. Ubuntu 26.04, then **`nvidia-driver-615-open`**. Not the `595` that `ubuntu-drivers` recommends. Hold the 615
   packages.
3. Kernel command line `amd_iommu=on iommu=pt` (IOMMU passthrough). CUDA Toolkit 13.5 (`cuda-toolkit-13-5` only).
4. Clone the fork (pinned to `e5ad35d`), install `pahole`, run `make modules`.
5. Set `RMForceStaticBar1=0` for each card in modprobe.d. Set the card addresses and build path in `contrib/config.sh`.
6. **Disable the `remove` rules in udev's `71-nvidia.rules` and mask `nvidia-persistenced`.** Otherwise udev puts the
   stock driver back right after the swap.
7. Stop the desktop, run `p2p-load.sh`, run `p2p-check.sh --integrity`. The result must be `ALL PASS`.
8. Run `p2p-install.sh` and reboot, so the patched modules load at every boot. Then put the udev rules and
   `nvidia-persistenced` back to default and reboot once more.

## Status

| | |
|---|---|
| Whole procedure, followed word for word from a fresh folder with the fork | **verified** on the test rig, 2026-11-23 |
| Live load + integrity test (264 cases, 1.4 TiB byte-exact, both directions) | **passed** |
| Boot install: P2P on at every boot, with Ubuntu's default udev rules and `nvidia-persistenced` | **verified** over two reboots |
| Kernel-update and rollback steps (17-18) | not run yet |
| Security review of everything in the fork that is not NVIDIA's code | [no malicious code found](docs/security-review.md) |

Test rig: 2x RTX 3090 (GA102, 24 GiB, 32 GiB BAR1), Ryzen 9 3900X on a B550 board (one card in the chipset x4 slot),
Ubuntu 26.04.1, kernel 7.0.0-44-generic, driver 615.71.09, CUDA 13.5.

### What P2P changed on that rig

Without P2P, every transfer between the two cards makes a detour: GPU -> the computer's main memory -> the other GPU.
With P2P, one card writes straight into the other's memory.

| | Without P2P | With P2P | |
|---|---|---|---|
| Delay for one GPU to send a copy to the other | 18-20 µs | **1-2 µs** | about 12x less |
| Both GPUs sending to each other at the same time | 3.49 GB/s | **5.52 GB/s** | +58 % |
| GPUs adding up their results so each has the total ¹ | 1.88 GB/s | **2.75 GB/s** | +46 % |
| GPUs sharing their pieces so each has all of them ² | 1.61 GB/s | **2.74 GB/s** | +70 % |
| Errors in 1.5 TB of GPU-to-GPU test copies | | **0** | |

¹ NCCL `all_reduce`. ² NCCL `all_gather`. NCCL is the library that PyTorch, vLLM and most multi-GPU software use to move
data between GPUs. When a model is split across two GPUs, they do this kind of exchange after every layer.

These are speeds of the data exchange itself, **not** of a whole AI workload: how much faster a model runs depends on
how much of its time it spends exchanging data. (On upstream's own two-GPU machine, vLLM generated tokens anywhere from
no faster to 32 % faster, depending on prompt length.) Absolute speeds on this rig are also held back by one card sitting in a slow x4 slot; with
both cards on CPU lanes they are several times higher. Full numbers and how to measure them:
[docs/benchmarks.md](docs/benchmarks.md).

## Disclaimer

This loads an **experimental, patched kernel driver**. It can hang the machine or corrupt GPU memory. The upstream
project has verified it only on RTX 3080 20GB pairs; other GA102 cards are "on by default, untested" there. Use at your
own risk, run the integrity test on your own pair, and keep a way back to stock (`p2p-fallback.sh`).

## Credits

- [noon-at-cgn/geforce-dynbar1-p2p](https://github.com/noon-at-cgn/geforce-dynbar1-p2p): the patched driver and its
  scripts, which [the fork](https://github.com/Cinderella-Man/geforce-dynbar1-p2p) builds on. It in
  turn builds on the P2P work of tinygrad, aikitoria, Duanyll and dingiv (see its README).
- [Ubsefor](https://github.com/Ubsefor/open-gpu-kernel-modules/commit/0e23c574b430c27d7b5ae394bc6ca488aff346cd): the
  CXL build fix for Linux 7.0.
- [NVIDIA open-gpu-kernel-modules](https://github.com/NVIDIA/open-gpu-kernel-modules): the driver, dual MIT/GPLv2.
