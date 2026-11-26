# Pascal (GTX 10, P106) and Volta (V100) cards

The patched driver in this guide cannot run these cards: it is built on NVIDIA's open kernel modules, which need the
GSP processor that came with Turing. On the test rig the open driver rejected a GTX 1070 Ti with `not supported by
open nvidia.ko because it does not include the required GPU System Processor (GSP)`. Pascal and Volta run on NVIDIA's
**closed** driver instead, and its **580** branch is the last that supports them. CUDA 13 dropped both, so programs for
them are built with CUDA 12.

Everything here uses that stock 580 driver, unpatched, and was done on 2026-11-23:

| | Volta: 2x Tesla V100 PCIe 32 GB | Pascal: 2x P106-090 |
|---|---|---|
| P2P | **works** with the stock driver, once the IOMMU is in passthrough mode. Without that, data is silently lost | **works** with the stock driver, no override, in the test rig (B550). A mining board could not carry the traffic |
| Integrity test | passed: 264 cases, 1403.6 GiB byte-exact | passed in the test rig: 264 cases, 1403.6 GiB byte-exact. Failed on the mining board's first copy (hardware) |
| CUDA, PyTorch | CUDA 12.9 | CUDA 12.4 and PyTorch 2.14.1 (CUDA 12.6 build) verified on a GTX 1070 Ti |

Both need the IOMMU in passthrough mode and a motherboard that routes traffic between its slots. An outside report
found the same on Intel: four Tesla P40s on a dual-socket X99 board, where every peer copy came back wrong until
`iommu=pt`, and P2P works only between two cards on the same socket
([below](#outside-report-four-tesla-p40s-on-a-dual-socket-x99)).

**A Pascal or Volta card with a card of another generation gets no P2P:** CUDA refuses peer access between two
architectures, and this guide's patched driver cannot run Pascal or Volta. For that pair use memory sharing with
[DRMShare](https://github.com/Cinderella-Man/drmshare/blob/main/docs/volta-dmabuf.md) instead: not P2P, Vulkan programs only, on the same stock driver 580.
Tested with a Tesla V100 next to an RTX 3090, an RTX 2060 and a GTX 1070 Ti
([when to use it](../README.md#when-p2p-is-not-possible-drmshare)).

## Volta: two Tesla V100s

| | |
|---|---|
| GPUs | 2x Tesla V100 PCIe 32 GB (`10de:1db6`, GV100), 32 GiB BAR1 each from the factory (no resizing needed) |
| Board, CPU | ASUS TUF GAMING B450-PLUS II, Ryzen 7 5700X, 32 GB RAM |
| Slots | GPU0: B450 chipset slot, **PCIe 2.0 x4** (B450's chipset lanes are Gen2). GPU1: CPU slot, PCIe 3.0 x16 |
| Software | Ubuntu 24.04.5, kernel 7.0.0-43, `nvidia-driver-580` 580.178.04, CUDA 12.9 |

V100s are data-center cards, so NVIDIA's driver offers them P2P without any patch: `nvidia-smi topo -p2p r` showed `OK`
and CUDA reported peer access both ways, out of the box.

### As found: P2P "works", but data is lost

The IOMMU was in Ubuntu's default translated mode (`iommu: Default domain type: Translated`). The copy test then gave:

```
P2P GPU0 -> GPU1: 1.4963 GB/s
P2P GPU1 -> GPU0: 6.63388 GB/s
```

6.6 GB/s into GPU0 is impossible: its PCIe 2.0 x4 link carries at most 2 GB/s (its host copies measure 1.44 / 1.63
GB/s). The kernel log shows why:

```
nvidia 0000:07:00.0: AMD-Vi: Event logged [IO_PAGE_FAULT domain=0x0012 address=0x7000000000 flags=0x0020]
nvidia 0000:07:00.0: AMD-Vi: Event logged [IO_PAGE_FAULT domain=0x0012 address=0xf3139068 flags=0x0020]
```

The IOMMU blocked GPU1's writes into GPU0's memory (`0x7000000000` is GPU0's BAR1, `0xf3...` its register window). The
copies "finished" without the data arriving, and neither the driver nor CUDA reported an error. Any program using P2P
on this setup would have worked on garbage. See also
[troubleshooting](troubleshooting.md#peer-copies-faster-than-the-link-allows-or-io_page_fault-from-nvidia).

### The fix: IOMMU passthrough

```bash
printf 'GRUB_CMDLINE_LINUX_DEFAULT="$GRUB_CMDLINE_LINUX_DEFAULT iommu=pt"\n' | sudo tee /etc/default/grub.d/99-p2p-iommu-pt.cfg
sudo update-grub && sudo reboot
```

On Intel use `intel_iommu=on iommu=pt`, as in [procedure.md step 4](procedure.md#4-iommu-in-passthrough-mode). After the
reboot every IOMMU group must be `identity`:

```bash
cat /sys/kernel/iommu_groups/*/type | sort | uniq -c
```

Undo: delete the file and run `sudo update-grub`.

### Results with passthrough

That board puts GPU0 in a chipset slot at **PCIe 2.0 x4** (about 1.7 GB/s each way at most) and GPU1 on CPU lanes at
PCIe 3.0 x16: the x4 link sets every speed below ([boards](boards.md)).

Integrity test (built for Volta, see [below](#building-the-test-programs-for-pascal-and-volta)):

```
bandwidth (cudaMemcpyPeerAsync, 256 MiB x 20, contents checked): A->B 1.62 GB/s, B->A 1.67 GB/s, both ways at once 2.90 GB/s
RESULT PASS 0000:06:00.0 0000:07:00.0 max-mib 4096 cases 264 transfers 2112 copied 1403.6 GiB
```

17.6 minutes, no IOMMU faults, AER or Xid lines. `p2pBandwidthLatencyTest`, P2P off (through host RAM) -> **on**:

| | GPU0 -> GPU1 / GPU1 -> GPU0 |
|---|---|
| Copy latency, GPU-timed | 82.3 / 27.3 -> **2.72 / 2.73 µs** |
| Copy latency, CPU-timed | 6.24 / 7.35 -> **1.66 / 1.66 µs** |
| Both directions at once | 2.12 -> **2.90 GB/s** (+37 %) |
| One direction | 1.61 / 1.43 -> 1.62 / 1.67 GB/s |

The chipset slot's PCIe 2.0 x4 link caps the bandwidth here. With both cards on CPU lanes the numbers would be far
higher; the latency gain does not depend on it.

## Pascal

### The software path on Ubuntu 26.04 (verified)

Tested on the test rig with a GTX 1070 Ti (GP104, CPU slot) next to an RTX 2060 (chipset slot):

| Layer | What works | Notes |
|---|---|---|
| Driver | `nvidia-driver-580` 580.178.04 from the Ubuntu archive (the closed one, not `-open`) | DKMS built it for kernel 7.0 in 2.5 minutes. Ubuntu also ships prebuilt modules (`linux-modules-nvidia-580-generic`) |
| CUDA | Ubuntu's `nvidia-cuda-toolkit`, CUDA 12.4.1 | NVIDIA's own repository for 26.04 only has CUDA 13, which cannot build for Pascal. Ubuntu's package brings GCC 13 along: 73 packages, about 2 GB to download |
| Test programs | this repository's `tools/` and `p2pBandwidthLatencyTest`, built with `/usr/bin/nvcc -arch=native` | |
| PyTorch | 2.14.1, CUDA 12.6 build (`--index-url https://download.pytorch.org/whl/cu126`) | Its architecture list includes `sm_60`, which runs on every Pascal card. The CUDA 13 builds do not |

Coming from this guide's driver 615: unhold and remove every 615 package first (`apt-mark unhold`, `apt purge`), then
`sudo apt install nvidia-driver-580` and hold its packages. Two things on Ubuntu 26.04 that stop a PyTorch install:

- `python3 -m venv` needs `sudo apt install python3-venv` first.
- `pip install torch` fails with `No space left on device`: `/tmp` is in RAM on 26.04 (3.6 GB on the test rig) and
  PyTorch's downloads do not fit. Point pip elsewhere: `mkdir -p ~/pip-tmp && TMPDIR=~/pip-tmp pip install ...`.

PyTorch on the GTX 1070 Ti: float32 matrix multiply and convolution matched a float64 CPU reference (largest relative
error 1.2e-6), and a small training run converged. Speed:

| | GTX 1070 Ti | RTX 2060 |
|---|---|---|
| float32 matrix multiply | **6.54** TFLOPS | 5.90 TFLOPS |
| float16 matrix multiply | 6.66 TFLOPS (no gain: Pascal has no tensor cores) | **26.19** TFLOPS |

### P2P on the test rig: two P106-090s (verified)

| | |
|---|---|
| GPUs | 2x P106-090 (`10de:1c09`, GP106, 6 GB), mining cards without video outputs and with passive heatsinks |
| Slots | the test rig's usual two: B550 chipset slot and CPU slot. Both cards link at **PCIe 1.0 x4**, the most a P106-090 supports (`lspci`: `LnkCap ... 2.5GT/s, Width x4`) |
| Software | Ubuntu 26.04.1, kernel 7.0.0-44, `nvidia-driver-580` 580.178.04, CUDA 12.4, `iommu=pt` ([procedure.md step 4](procedure.md#4-iommu-in-passthrough-mode)). No `ForceP2P`, `RegistryDwords` empty |

`nvidia-smi topo -p2p r` showed `OK` straight away and CUDA reported peer access both ways. The integrity test, built for
`sm_61` ([below](#building-the-test-programs-for-pascal-and-volta)):

```
bandwidth (cudaMemcpyPeerAsync, 256 MiB x 20, contents checked): A->B 0.84 GB/s, B->A 0.83 GB/s, both ways at once 1.48 GB/s
RESULT PASS 0000:04:00.0 0000:2b:00.0 max-mib 4096 cases 264 transfers 2112 copied 1403.6 GiB
```

33.7 minutes, no IOMMU faults, AER or Xid lines. `p2pBandwidthLatencyTest`, P2P off (through host RAM) -> **on**:

| | GPU0 -> GPU1 / GPU1 -> GPU0 |
|---|---|
| Copy latency, GPU-timed | 22.4 / 22.2 -> **3.11 / 3.10 µs** |
| Copy latency, CPU-timed | 5.71 / 5.62 -> **1.72 / 1.73 µs** |
| Both directions at once | 1.09 -> **1.48 GB/s** (+36 %) |
| One direction | 0.82 / 0.74 -> 0.84 / 0.83 GB/s |

One direction stays at the cards' PCIe 1.0 x4 limit; P2P cannot change that. The gain is in latency and in traffic
both ways, as on every other pair.

Three things to know before trying this with P106 cards:

- **The board needs a display card to boot.** A P106 has no video outputs and no display firmware, and the Ryzen 9
  3900X has no integrated graphics: with only the two P106s the board stopped at POST with its VGA debug LED on. A GTX
  1070 Ti on a mining riser in a PCIe x1 slot fixed POST. (That card then dropped off the bus when the driver loaded,
  likely for lack of riser power, which stops every CUDA program:
  [troubleshooting](troubleshooting.md#cudagetdevicecount-invalid-device-ordinal-with-one-gpu-that-failed-to-start).)
- **Passive cooling.** These cards rely on a mining frame's airflow. The test ran with the power limit at its minimum
  (`sudo nvidia-smi -i 0,1 -pl 60`, until the next reboot) and a small script that logged the temperatures every 2 s and
  would have stopped the test at 75 °C. The peak was 46 °C: copies at PCIe 1.0 x4 keep the GPUs mostly idle. Heavier
  work would heat them far more.
- **The cards themselves cap the bandwidth** at about 0.84 GB/s each way, in any slot.

### Outside report: four Tesla P40s on a dual-socket X99

From an outside report, not reproduced here
([issue 1](https://github.com/Cinderella-Man/how-to-enable-p2p-on-nvidia-gpus/issues/1), with a
[full write-up](https://gist.github.com/Cavemanmining/896bd4fa826da665525c9d2f9742bcff)).

| | |
|---|---|
| GPUs | 4x Tesla P40 24 GB (GP102), each on PCIe 3.0 x16, 32 GiB BAR1 from the factory |
| Board | dual-socket Intel X99, VT-d on, two cards on each socket |
| Software | Ubuntu, kernel 7.0, NVIDIA's stock driver 580.178.04 (closed, unpatched), CUDA 12.0 |

The driver offered P2P inside each socket's pair (`nvidia-smi topo -p2p r`: `OK`, CUDA peer access both ways) and not
across the sockets (`TNS`, topology not supported: P2P does not cross the link between the two CPUs). Pair cards by
socket.

**As found**, with the IOMMU in Ubuntu's default translated mode (`DMA-FQ`): every peer copy came back wrong, 12 of 12
cases (1-16 MiB, both directions), while every CUDA call returned success. The only sign was in the kernel log:

```
DMAR: [DMA Write NO_PASID] Request device [xx:00.0] fault addr ... [fault reason 0x05] PTE Write access is not set
```

It is Intel's version of the V100 machine's `IO_PAGE_FAULT` ([above](#as-found-p2p-works-but-data-is-lost)): the
IOMMU blocks the writes into the other card. Here both directions were lost, and on PCIe 3.0 x16 links the speed gave
nothing away. Copies with peer access off, through host RAM, were byte-exact.

**With `iommu=pt`** ([the fix above](#the-fix-iommu-passthrough)): 192 of 192 cases byte-exact, 128 GiB copied, no
`DMAR` or `Xid` lines. That was the reporter's own test (random buffers from 1 MiB to 4 GiB, each copy compared byte by
byte, peer access off and on), not this repository's integrity test, on one pair. `p2pBandwidthLatencyTest` and the
reporter's copies, P2P off (through host RAM) -> **on**:

| | Both directions of the pair |
|---|---|
| Copy latency, GPU-timed | 13.8-14.3 -> **1.0 µs** |
| Copy latency, CPU-timed | 7.3-7.9 -> **2.3-2.4 µs** |
| Both directions at once | 16.1 -> **19.2 GB/s** (+19 %) |
| One direction | 11.6 -> 10.3 GB/s (slower with P2P) |

Passthrough also sped up the copies through host RAM, from 9.90 / 8.72 to 11.7 / 11.6 GB/s.

In llama.cpp the pair ran a 32B dense model (Q4_K_M) about two-thirds faster than one card with tensor split
(`-sm tensor`: 12.9 -> 21.6 tokens/s generated), while layer split gave nothing (12.9). Most of that gain does not need
P2P: `GGML_CUDA_P2P=1` (llama.cpp enables peer access only when it is set) added 4-6 % of it (20.8 -> 21.6 tokens/s on
the 32B, 63.9 -> 67.9 on an 8B). Doing that exchange through NCCL (llama.cpp's `GGML_CUDA_ALLREDUCE=nccl`) added
another 6-9 % generation and 14-23 % prompt speed, with `NCCL_P2P_LEVEL=PHB` set: on an Intel board NCCL uses no P2P
between two CPU slots without it ([troubleshooting](troubleshooting.md#nccl-goes-via-shm-although-cuda-offers-peer-access)).
`tools/p2p_patterns` was not run.

### P2P: what the stock driver says

- **It does not block Pascal GPUs.** Two P106-090s got P2P in the test rig without any override (above). On the mining
  box (below), `nvidia-smi topo -p2p r` said `CNS` (chipset not supported) for every pair, not `GNS` (GPU not
  supported). On the test rig the driver's P2P capability answer for a GTX 1070 Ti with itself was `0x67`: writes, reads,
  PCIe and loopback supported.
- **It refuses Pascal + Turing pairs** (`NS`, capability `0x0800`, none of the read, write or PCIe bits). That refusal
  comes from the kernel driver itself, so the preload shim from [turing.md](turing.md) cannot help, and the closed driver
  cannot be patched. Different generations should not share memory this way anyway.
- **Two different Pascal chips** (for example GP104 + GP106) were not tested. The driver refuses different chips unless
  `ForceP2P` (below) is set, and CUDA would then still need the shim, as on Turing.

### `ForceP2P`: overriding the chipset check

The driver reads a registry key `ForceP2P` ([`nvrm_registry.h`](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/580.178.04/src/nvidia/interface/nvrm_registry.h)).
Its PCIe P2P check returns "supported" at once when it is set, before the chipset table, the topology checks and the
same-chip check ([`p2p_caps.c`](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/580.178.04/src/nvidia/src/kernel/platform/p2p/p2p_caps.c),
`_kp2pCapsCheckStatusOverridesForPcie`). `0x11` allows reads and writes. NVIDIA marks it as a test switch ("only be
used to test on internal issues"): it removes the driver's safety check, so only the integrity test tells whether the
hardware actually carries the traffic.

```bash
echo 'options nvidia NVreg_RegistryDwords="ForceP2P=0x11"' | sudo tee /etc/modprobe.d/zz-p2p-test.conf
sudo update-initramfs -u
sudo reboot
```

Check: `grep RegistryDwords /proc/driver/nvidia/params` shows `ForceP2P=0x11`, and `nvidia-smi topo -p2p r` shows `OK`.
The IOMMU must be in passthrough mode as well ([above](#the-fix-iommu-passthrough)). Undo: delete the file, run
`sudo update-initramfs -u`, reboot.

### The test on a mining box: the driver said yes, the hardware said no

| | |
|---|---|
| GPUs | 9x P106-090 (`10de:1c09`, GP106, 6 GB), every one on a **PCIe 1.0 x1** link |
| Board | mining board with a Celeron 3865U (Intel Sunrise Point-LP chipset), 3 GB RAM. Two ASMedia ASM1184e switches with four GPUs each, one GPU straight on a chipset root port |
| Software | Ubuntu 24.04.5, kernel 7.0.0-42, `nvidia-driver-580` 580.178.04, CUDA 12.9 |

With `ForceP2P=0x11` and `iommu=pt`, the driver reported `OK` and CUDA peer access for all 72 pairs. The integrity test
on two GPUs behind the **same** switch then failed on its first copy: none of the 512 words arrived. The kernel log:

```
pcieport 0000:00:1c.0: AER: Multiple Uncorrectable (Non-Fatal) error message received from 0000:00:1c.0
pcieport 0000:00:1c.0: AER:   TLP Header: 0x40701002 0x030000ff 0xdc139068 0x00000000
```

The header is a memory write from GPU0 (`03:00.0`) to `0xdc139068`, inside GPU1's register window
(`dc000000-dcffffff`), and the chipset root port `00:1c.0` logged it as an Unsupported Request. The driver addressed the
write correctly; the ASM1184e switch passed it up to the chipset instead of across to its neighbour port, and the
chipset does not send traffic back down the port it came from. So this board cannot carry P2P at all, as NVIDIA's `CNS`
verdict said. The four GPUs behind that root port needed a reboot afterwards. Both settings were removed again.

### Not tested yet

- Other Pascal models (GTX 1060, 1070, 1080, 1080 Ti, P102/P104 mining cards). Same driver code path as the P106-090,
  so they should behave the same on a board that routes the traffic; run the integrity test before relying on it.
  (The Tesla P40 has the outside report [above](#outside-report-four-tesla-p40s-on-a-dual-socket-x99).)
- Two different Pascal chips (for example GTX 1070 Ti + P106-090). The driver needs `ForceP2P` for that, and CUDA the
  chip-ID shim.

## Building the test programs for Pascal and Volta

Use CUDA 12 (`/usr/bin/nvcc` from Ubuntu's `nvidia-cuda-toolkit`, or `/usr/local/cuda-12.x/bin/nvcc`). `-arch=sm_61`
for Pascal (`sm_60` for P100), `sm_70` for Volta, or `-arch=native`. In this repository's checkout:

```bash
NV=/usr/bin/nvcc
$NV -O2 -arch=native tools/p2p_access.cu -o tools/p2p_access
$NV -O2 -arch=native tools/p2p_bandwidth.cu -o tools/p2p_bandwidth
$NV -O2 -arch=native tools/pcie_bandwidth.cu -o tools/pcie_bandwidth -lnvidia-ml
$NV -O2 -arch=native -I tools/cuda-samples/Common \
  tools/cuda-samples/cpp/5_Domain_Specific/p2pBandwidthLatencyTest/p2pBandwidthLatencyTest.cu \
  -o tools/p2pBandwidthLatencyTest
```

The byte-exact integrity test comes from the fork; its own `build.sh` only targets Turing and newer, so build it
directly. One file, pinned to the same commit as [procedure.md](procedure.md) step 8:

```bash
curl -fLO https://raw.githubusercontent.com/Cinderella-Man/geforce-dynbar1-p2p/25c929e8cfebe69f98fd07267aa28cab52b9a6b8/contrib/integrity/p2p-integrity.cu
$NV -O2 -std=c++17 -arch=native -o p2p-integrity p2p-integrity.cu -ldl
./p2p-integrity --list
./p2p-integrity --pair 0000:06:00.0 0000:07:00.0 --max-mib 4096
```

`--max-mib` sets the largest buffer: 4096 for the full 264-case run, smaller on slow links (`--max-mib 64` runs 168
cases). The last line must be `RESULT PASS`. Check the kernel log afterwards for `IO_PAGE_FAULT`, `DMAR`, `AER` and `Xid`
lines.
