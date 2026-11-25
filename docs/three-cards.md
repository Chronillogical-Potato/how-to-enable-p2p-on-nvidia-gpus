# Three or more cards: three RTX 3090s

P2P between three RTX 3090s, one of them behind the chipset, on a second machine. Everything was done on 2026-11-25 as a
**trial load** (procedure step 13): the machine had to stay up, so nothing was installed for boot and it was not
rebooted. The boot install and reboots are not tested on this machine.

| | |
|---|---|
| Board, CPU | Gigabyte X570 AORUS ULTRA, Ryzen 9 5900X |
| GPU0 `04:00.0` | RTX 3090 24 GB, behind the X570 chipset, PCIe 4.0 x4 |
| GPU1 `0a:00.0` | RTX 3090 24 GB, CPU lanes, PCIe 4.0 x8 |
| GPU2 `0b:00.0` | RTX 3090 24 GB, CPU lanes, PCIe 4.0 x8 |
| Software | Ubuntu 26.04.1, kernel 7.0.0-43-generic, Ubuntu's 615.71.09 open modules, the fork at `b70b69c`, CUDA 13.5 |

**Result:** the integrity test passed on all three pairs, and every card could allocate 23 GiB of its 24 with every
other card mapping it. Between the two cards on CPU lanes, 13.1 GB/s each way and
16.65 GB/s both ways at once; with the card behind the chipset, 6.60 GB/s each way.

## What differs from two cards

1. **List every card.** `GPUS` in `contrib/config.sh` ([procedure step 11a](procedure.md#11a-scripts-which-cards-which-build))
   and the per-card line of [step 11b](procedure.md#11b-driver-dynamic-bar1-for-each-card-rmforcestaticbar10) take
   all three; the commands there already do that. The integrity test then runs every pair, one after the other: three
   pairs, about 1.4 TiB of copies each.
2. **Every card needs a large BAR1, even one whose vBIOS has no Resizable BAR.** See [below](#bar1-on-cards-without-resizable-bar).
3. **Peer access between all cards at once needs fork commit `b70b69c`** or later (the guide pins it). Before it, the
   patched driver let each card map only one card's BAR1 of all the other cards' memory together
   ([troubleshooting](troubleshooting.md#three-or-more-cards-cudamalloc-runs-out-of-memory-with-peer-access-between-all-of-them)).
   With two cards that limit never showed. Programs that turn on peer access for whole cards hit it: llama.cpp with
   `GGML_CUDA_P2P=1`, for example. [Checked here](#peer-memory-old-and-new-build), old build against new.
4. **NCCL on AMD with three or more GPUs** goes through host RAM by default between cards that meet in the CPU, like
   on Intel: set `NCCL_P2P_LEVEL=PHB` ([troubleshooting](troubleshooting.md#nccl-goes-via-shm-although-cuda-offers-peer-access);
   from NCCL's source, not tested on this machine).

## BAR1 on cards without Resizable BAR

Step 1 asks for a BAR1 at least as large as each card's memory. On this machine all three showed 256 MB: Resizable BAR
was off in the BIOS, and the vBIOS of two cards did not offer more:

```bash
for d in $(lspci -D -d 10de::0300 | awk '{print $1}'); do echo "$d $(sudo lspci -vv -s $d | grep -oE 'BAR 1: current size.*')"; done
```

```
0000:04:00.0 BAR 1: current size: 256MB, supported: 64MB 128MB 256MB
0000:0a:00.0 BAR 1: current size: 256MB, supported: 64MB 128MB 256MB
0000:0b:00.0 BAR 1: current size: 256MB, supported: 64MB 128MB 256MB 512MB 1GB 2GB 4GB 8GB 16GB 32GB
```

A card that lists `32GB` gets it from the BIOS setting. A card that does not would need a vBIOS update from its maker.
Instead, [geforce-bar1-resize](https://github.com/noon-at-cgn/geforce-bar1-resize), the tool the Turing cards use
([turing.md T1](turing.md#t1-install-geforce-bar1-resize-pinned)), gave all three cards 32 GiB: by a plain resize on
`0b:00.0`, and through the frame-buffer size override in config space (`0x610`) on the other two, including the card
behind the chipset. Its read-only check shows the route per card:

```bash
sudo scripts/geforce-bar1-resize check --vram 24G 0000:04:00.0
```

```
  ok   S1 VSEC (ID 000b v1) at 0x600 in the capability chain
  ...
  ok   S6 type 0 endpoint
  switch  behind a PCIe switch: yes
  PLAN    arm 0x610 VAL=9, resize to 32768 MiB
  note    behind a PCIe switch: the plain resize usually fails with ENOSPC there
```

The note did not apply here: the kernel grew the chipset's windows by itself. Probably because an RTX 3090 has no USB-C
function next to the GPU, which is what blocked the Turing cards ([turing.md T2](turing.md#t2-add-the-usb-c-pre-step)).
The tool lists every card but the RTX 3080 20GB as untested; on the 3090 it verified every step as it ran.

**How it was run here.** The tool normally runs at every boot, before the driver loads
([turing.md T1-T3](turing.md#t1-install-geforce-bar1-resize-pinned)). Here, to avoid a reboot, it ran once by hand
with the driver unloaded (after [step 12](procedure.md#12-prepare-the-live-swap-stop-everything-that-reloads-the-stock-driver),
`rmmod` of the stock modules), as `geforce-bar1-resize record` while the driver was still loaded, then
`geforce-bar1-resize boot --card <address>` per card. The windows stay until the next reboot. For a machine that keeps
P2P, install the tool as in turing.md T1 (without T2, which only Turing cards need) and reboot. Not tested here.

**IOMMU, also without a reboot.** This machine had no `iommu=pt`
([step 4](procedure.md#4-iommu-in-passthrough-mode)). With the driver unloaded and each card's HDMI audio function
unbound, the three cards' IOMMU groups were switched to passthrough through sysfs, until the next reboot:

```bash
echo identity | sudo tee /sys/kernel/iommu_groups/<group>/type
```

The integrity test then passed. For a lasting setup use step 4.

**Power limits.** This machine caps its cards at 250 W with a unit that runs `nvidia-smi -pm 1` and `-pl 250` at boot.
The driver swap dropped both, as [step 13](procedure.md#13-trial-load-nothing-is-installed-a-reboot-returns-to-stock)
warns: the tests above ran at the cards' default limits (350-420 W), and after the return to stock the unit had to run
again (`systemctl restart <unit>`).

## Results

### Integrity test

`p2p-check.sh --integrity`, the build at `b70b69c`, all three pairs (264 cases each, both directions, 1.4 TiB
byte-exact each):

| Pair | Path | One way | Both ways at once | Result |
|---|---|---|---|---|
| GPU0 <-> GPU1 | chipset x4 <-> CPU x8 | 6.60 / 6.60 GB/s | 12.75 GB/s | **PASS** |
| GPU0 <-> GPU2 | chipset x4 <-> CPU x8 | 6.60 / 6.60 GB/s | 12.76 GB/s | **PASS** |
| GPU1 <-> GPU2 | CPU x8 <-> CPU x8 | 13.12 / 13.17 GB/s | 16.65 GB/s | **PASS** |

The kernel log had no `Xid`, `AER` or IOMMU lines. The chipset card's x4 link sets the pace for its two pairs.

### Peer memory, old and new build

[`tools/p2p_peermap`](../tools/p2p_peermap.cu) turns on peer access between all three cards, then allocates the same
size on each card in turn, so each allocation is mapped into the other two. Then every card writes into the end of
every other card's allocation, and the owner checks it word by word:

```bash
nvcc -O2 -arch=native tools/p2p_peermap.cu -o tools/p2p_peermap -lnvidia-ml
```

```bash
./tools/p2p_peermap 20480
```

| MiB per card | Old build (`25c929e`) | New build (`b70b69c`) |
|---|---|---|
| 15000, 16000 | all three ok, data exact | |
| 16352 | all three ok, data exact | |
| 16400 | **GPU1: out of memory** (23.9 GB free) | all three ok, data exact |
| 17408, 20480 | **GPU1: out of memory** | all three ok, data exact |
| 22528, 23552 | | all three ok, data exact |

16352 MiB is exactly what the old check allowed: each card may map 32768 - 64 MiB of the other two cards together,
so 16352 MiB of each. With the new build each card can map the whole of the others' memory. Each card's BAR1 holds
its own allocation once, however many cards map it (`BAR1 used` 23.5 of 32 GiB per card at 23552 MiB), so 24 GB cards
with 32 GiB windows have room to spare.

Not tested here: the boot install, reboots, NCCL and real workloads on three cards, `tools/p2p_patterns`, and the
driver's new log line for a refused window (a 3090 cannot fill another card's 32 GiB window).
