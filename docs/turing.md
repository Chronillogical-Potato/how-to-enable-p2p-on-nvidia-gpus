# Turing cards (RTX 20 / GTX 16): experimental

P2P on Turing GeForce cards works, but it takes more than the RTX 30 route in [procedure.md](procedure.md): the cards
need a bigger memory window first, P2P has to be switched on per card, and a pair of two *different* chips needs a
workaround for CUDA itself. Everything here was done on the test rig on 2026-11-23 with:

| | Card | Chip | Memory | Slot |
|---|---|---|---|---|
| GPU0 `04:00.0` | EVGA GeForce RTX 2060 | TU104 (`10de:1e89`) | 6 GB | B550 chipset slot, PCIe 3.0 x4 |
| GPU1 `2b:00.0` | MSI GeForce GTX 1650 SUPER | TU116 (`10de:2187`) | 4 GB | CPU slot, PCIe 3.0 x16 |

Result: P2P on at every boot, the integrity test passed twice (240 cases, 636 GiB of byte-exact copies each time),
and the benchmarks match the RTX 3090 and 3080 pairs in the same slots ([results](#results)). As far as we could find, this is the first
integrity-verified P2P between two different Turing chips. Same-chip Turing P2P (RTX 2080 Ti with CMP 50HX cards, all
TU102) was reported earlier with other patches: [cmp50hx-unlock](https://github.com/xrip/cmp50hx-unlock/blob/master/docs/P2P.md).

**Experimental.** Every piece below was verified on one pair, one board and one driver version. Run the integrity
test on your own pair before trusting it, and keep a way to reach the machine if it hangs.

**Not for GTX 10 (Pascal) or older.** NVIDIA's open kernel modules need the GSP processor that came with Turing. On
the test rig the driver rejected a GTX 1070 Ti with `not supported by open nvidia.ko because it does not include the
required GPU System Processor (GSP)`. Driver 615 and CUDA 13 no longer support Pascal at all. What does work on
Pascal and Volta with NVIDIA's stock driver: [pascal-and-volta.md](pascal-and-volta.md).

## How it differs from RTX 30 cards

| | RTX 3080 / 3090 (GA102) | Turing |
|---|---|---|
| Memory window (BAR1) | Resizable BAR from the firmware: 16 or 32 GB | **256 MB**; the cards never got NVIDIA's Resizable BAR. Grown at every boot by [geforce-bar1-resize](https://github.com/noon-at-cgn/geforce-bar1-resize) plus a pre-step from this repository (T1-T3) |
| P2P in the patched driver | on by default | **opt-in** per card: trial load, integrity test, then `p2p-enable.sh` (T4-T6) |
| Integrity test | full run, 264 cases | cards with 6 GB or less skip the 4 GB sizes, so `p2p-enable.sh` needs `--skip-integrity` |
| CUDA | peer access as soon as the driver offers it | for **two different chips** (here TU104 + TU116) CUDA refuses; a small preload shim gets around it (T5). Two identical chips do not need it |

## Before you start

Do [procedure.md](procedure.md) steps 1-10, 11a and 16a: BIOS, Ubuntu, driver 615, IOMMU, CUDA, the fork, the build,
`contrib/config.sh`, and the initramfs fix. Skip 11b: the resize tool writes that file. Then come back here.

Check that your cards are Turing and stuck at 256 MB:

```bash
for d in $(lspci -D -d 10de::0300 | awk '{print $1}'); do lspci -nns $d; sudo lspci -vv -s $d | grep -E 'Region 1:|BAR 1:'; done
```

Expected: `TU1xx` in the names, `[size=256M]` and `BAR 1: current size: 256MB, supported: 64MB 128MB 256MB`.

## T1. Install geforce-bar1-resize (pinned)

[geforce-bar1-resize](https://github.com/noon-at-cgn/geforce-bar1-resize) is from the same author as the patched
driver. It enlarges each card's BAR1 at every boot, before the NVIDIA driver loads, by arming a frame-buffer size
override in the card's PCI config space (`0x610`). Nothing is flashed; a reboot without it returns the stock window.
We reviewed its code before running it (no network access, no writes outside its documented files; its optional
"layout" kernel module stays unused here). Pin the reviewed commit:

```bash
cd ~
git clone https://github.com/noon-at-cgn/geforce-bar1-resize.git
cd geforce-bar1-resize
git checkout 53466608917ff5d8c660eefcee7c7e6c4261ef1b
```

Read-only check, one card at a time with its memory size:

```bash
sudo scripts/geforce-bar1-resize check --vram 6G 0000:04:00.0
```

```bash
sudo scripts/geforce-bar1-resize check --vram 4G 0000:2b:00.0
```

Each card must show six lines starting with `ok` (S1 to S6) and a `PLAN` line `arm 0x610 VAL=7, resize to 8192 MiB`
(6 GB card) or `arm 0x610 VAL=6, resize to 4096 MiB` (4 GB card). A card behind a chipset or other PCIe switch also
prints `behind a PCIe switch: yes` and a note that the plain resize usually fails there: T2 is what makes it work.

Install it with the layout module off, dry run first:

```bash
sudo ./install.sh --dry-run --layout off --enable
```

```bash
sudo ./install.sh --layout off --enable --yes
```

## T2. Add the USB-C pre-step

RTX 20 and GTX 16 cards have a USB-C controller (and a UCSI function) next to the GPU. Their address ranges sit in
the same window as the GPU's BAR1, and the kernel refuses to grow a window that holds another device's range: the
resize fails with `No space left on device`. Remove those two functions first and the kernel's own resize grows every
window up to the root port by itself, also behind the B550 chipset, and without touching the chipset's other devices
(network, SATA, USB).

[`tools/turing/gpu-usbc-off`](../tools/turing/gpu-usbc-off) does exactly that: it removes only NVIDIA functions of
class USB xHCI (`0c0330`) or UCSI (`0c8000`) that share a slot with an NVIDIA GPU. Run it before the resize tool,
through a systemd drop-in (the tool's own files stay unchanged):

```bash
cd ~/how-to-enable-p2p-on-nvidia-gpus
sudo install -m 0755 tools/turing/gpu-usbc-off /usr/local/sbin/gpu-usbc-off
sudo mkdir -p /etc/systemd/system/geforce-bar1-resize.service.d
printf '[Service]\nExecStartPre=/usr/local/sbin/gpu-usbc-off\n' | sudo tee /etc/systemd/system/geforce-bar1-resize.service.d/10-gpu-usbc-off.conf
sudo systemctl daemon-reload
```

Side effect: the cards' USB-C ports do not work while this is in place.

Larger windows on Turing are not new in themselves. [NvStrapsReBar](https://github.com/terminatorul/NvStrapsReBar)
does it by flashing a modified UEFI onto the motherboard, cmp50hx-unlock with kernel and firmware patches, and
geforce-bar1-resize alone already reached 4 GB on a user's GTX 1650. What T2 adds is a way to make room where the
resize fails for lack of it: no flashing, no kernel patches and no boot options, even behind a chipset.

## T3. Reboot and check the windows

```bash
sudo reboot
```

```bash
journalctl -b -u geforce-bar1-resize --no-pager -o cat | grep -E 'gpu-usbc-off|armed and resized'
```

Expected (test rig):

```
gpu-usbc-off: removing 0000:04:00.2 (class 0x0c0330) next to GPU 0000:04:00.0
gpu-usbc-off: removing 0000:04:00.3 (class 0x0c8000) next to GPU 0000:04:00.0
gpu-usbc-off: removing 0000:2b:00.2 (class 0x0c0330) next to GPU 0000:2b:00.0
gpu-usbc-off: removing 0000:2b:00.3 (class 0x0c8000) next to GPU 0000:2b:00.0
... 0000:04:00.0 armed and resized to 8192 MiB
... 0000:2b:00.0 armed and resized to 4096 MiB
```

```bash
for d in $(lspci -D -d 10de::0300 | awk '{print $1}'); do echo "$d $(sudo lspci -vv -s $d | grep -oE 'Region 1:.*')"; done
```

Each `Region 1` must be at least the card's memory (`[size=8G]`, `[size=4G]`). Then the pair check:

```bash
cd ~/geforce-dynbar1-p2p && contrib/p2p-check.sh --pairs
```

Expected: `untested (TU104 + TU116): run p2p-check.sh --integrity on a trial load ...`. Before T1-T3 it said
`not supported (BAR1 of ... is 256 MiB, smaller than its ... MiB of VRAM ...)`.

## T4. Trial load with P2P on

Do [procedure.md step 12](procedure.md#12-prepare-the-live-swap-stop-everything-that-reloads-the-stock-driver) (udev
override, mask `nvidia-persistenced`, stop the desktop), then load the patched build with P2P switched on for both
cards:

```bash
cd ~/geforce-dynbar1-p2p
sudo env P2P_ENABLE=1 contrib/p2p-load.sh --yes
```

The end of its output must show `OK` in both `topo -p2p` tables, and the windows keep their size through the driver
reload.

## T5. CUDA, and the integrity test

```bash
~/how-to-enable-p2p-on-nvidia-gpus/tools/p2p_access
```

Two identical Turing chips should print `YES` here: go straight to the integrity test below, without the shim.

A pair of **different** chips prints `NO` although the driver says `OK`. The kernel driver really does report P2P
as supported: traced on the test rig, its capability answer to CUDA was `0x2043` (reads, writes, PCIe, PCIe-BAR1) for
both directions. But NVIDIA's closed CUDA library, libcuda, refuses peer access between GPUs whose chip
*implementation* differs. It reads that from each GPU through the driver call `MC_GET_ARCH_INFO`: architecture
`0x160` (Turing) for both, implementation `0x4` (TU104) for the RTX 2060 and `0x8` (TU116) for the 1650 SUPER.

[`tools/turing/chipid-shim.c`](../tools/turing/chipid-shim.c) is a small preload library that rewrites that one answer
inside one process, so libcuda sees one chip. It changes nothing in the kernel, the driver or the GPUs, and does
nothing unless you set `P2P_CHIP_IMPL_MAP=ARCH:FROM:TO`. Build it and look up your values:

```bash
cd ~/how-to-enable-p2p-on-nvidia-gpus/tools/turing
gcc -O2 -shared -fPIC chipid-shim.c -o libchipid-shim.so -ldl
P2P_CHIP_IMPL_VERBOSE=1 LD_PRELOAD=$PWD/libchipid-shim.so ../p2p_access
```

```
chipid-shim: GPU object 0x5c000003 architecture 0x160 implementation 0x4
chipid-shim: GPU object 0x5c00000a architecture 0x160 implementation 0x8
GPU 0 -> GPU 1: NO
```

Map one implementation to the other (here TU116 `0x8` to TU104 `0x4`) and check again:

```bash
P2P_CHIP_IMPL_MAP=0x160:0x8:0x4 LD_PRELOAD=$PWD/libchipid-shim.so ../p2p_access
```

Expected: `YES` both ways. Now the real test, with the shim (or without it, for two identical chips):

```bash
cd ~/geforce-dynbar1-p2p
sudo env P2P_CHIP_IMPL_MAP=0x160:0x8:0x4 LD_PRELOAD=$HOME/how-to-enable-p2p-on-nvidia-gpus/tools/turing/libchipid-shim.so contrib/p2p-check.sh --integrity
```

The last line must be `ALL PASS`. Cards with 4 GB skip the 4 GB sizes (`skip malloc 4 GiB: ... MiB free`), so the
run has 240 cases instead of 264. The end of the test rig's run after T6, in the boot setup:

```
skip malloc 4 GiB: 5655 MiB free on A, 3654 MiB on B
skip cumem  4 GiB: 5655 MiB free on A, 3654 MiB on B
bandwidth (cudaMemcpyPeerAsync, 256 MiB x 20, contents checked): A->B 3.28 GB/s, B->A 2.93 GB/s, both ways at once 5.49 GB/s
RESULT PASS 0000:04:00.0 0000:2b:00.0 max-mib 4096 cases 240 transfers 1920 copied 636.0 GiB
PASS integrity 0000:04:00.0 <-> 0000:2b:00.0 (max-mib 4096, 240 cases)
note: a run below --max-mib 4096 (264 cases) does not count for p2p-enable.sh
```

The run on the trial load before it passed the same way (3.28 / 2.95 / 5.52 GB/s). Each took just under 5 minutes,
and the kernel log had no Xid, AER or DMAR errors. The note in the last line is why T6 needs `--skip-integrity`.

**The shim is a workaround, not a fix.** libcuda may use the chip implementation for more than this check. On the
test rig the integrity test, `p2pBandwidthLatencyTest`, `nvbandwidth` and NCCL all ran clean with it, but that is one
pair. Use it only for the programs that need P2P, never in `/etc/ld.so.preload`, and repeat the integrity test after
any driver update.

## T6. Switch P2P on for every boot

```bash
cd ~/geforce-dynbar1-p2p
sudo contrib/p2p-enable.sh --dry-run --skip-integrity 0000:04:00.0 0000:2b:00.0
```

```bash
sudo contrib/p2p-enable.sh --yes --skip-integrity 0000:04:00.0 0000:2b:00.0
```

It adds an entry per card (tied to its PCI ids) with `keys=RMDynBar1P2PEnable=1;` to `/etc/geforce-bar1-resize.conf`
and rewrites the modprobe option file. `--skip-integrity` is needed because its gate wants the full 264-case run,
which a 4 GB card cannot do; the reduced run above passed.

Do the boot install ([procedure.md step 16b](procedure.md#16b-install-and-reboot)) if you have not yet, put udev and
`nvidia-persistenced` back to Ubuntu's defaults
([as in step 16](procedure.md#put-udev-and-nvidia-persistenced-back-to-ubuntus-defaults)), and reboot. Then:

```bash
cd ~/geforce-dynbar1-p2p && sudo contrib/p2p-check.sh
```

Expected: `opted in (on, TU104 + TU116: RMDynBar1P2PEnable=1; ...)` and `ALL PASS`; `nvidia-smi topo -p2p r` shows
`OK`. With the shim, `p2p_access` prints `YES`; without it, `NO` (as it should).

## Using it

Run every CUDA program that should use P2P with the two variables, for example:

```bash
P2P_CHIP_IMPL_MAP=0x160:0x8:0x4 LD_PRELOAD=$HOME/how-to-enable-p2p-on-nvidia-gpus/tools/turing/libchipid-shim.so python3 your_script.py
```

NCCL picks `via P2P/direct` by itself once CUDA offers peer access. Two identical chips need none of this.

**vLLM: pass `--disable-custom-all-reduce`.** On the test rig, vLLM's custom all-reduce reset the whole machine with an
"uncorrected error caused a data fabric sync flood event". The cause is both GPUs' kernels reading each other's memory
at the same time, which [`tools/p2p_patterns.cu`](../tools/p2p_patterns.cu) reproduces without vLLM
([troubleshooting](troubleshooting.md#the-machine-resets-data-fabric-sync-flood)). vLLM with NCCL's P2P ran clean for an
hour. The end-to-end numbers: [benchmarks/vllm](../benchmarks/vllm/README.md).

## Results

Same rig, same two slots as the RTX 3080 pair in [benchmarks.md](benchmarks.md); every CUDA program ran with the shim.
Without P2P (through host RAM) -> **with P2P**:

| | Turing pair | RTX 3080 pair |
|---|---|---|
| Delay for one GPU to send a copy to the other, average of both directions | 29 -> **1.5 µs** | 19 -> 1.4 µs |
| Both GPUs sending to each other at the same time | 3.26 -> **5.49 GB/s** | 3.50 -> 5.46 GB/s |
| NCCL all_reduce (syncing a model split across both GPUs), 512 MiB | 1.89 -> **2.74 GB/s** | 1.85 -> 2.72 GB/s |
| NCCL all_gather (each GPU collects the other's data), 512 MiB | 1.53 -> **2.73 GB/s** | 1.56 -> 2.69 GB/s |
| NCCL send/recv, 512 MiB | 1.86 -> **2.49 GB/s** | 1.79 -> 2.72 GB/s |
| Errors in 1.3 TB of GPU-to-GPU test copies (two integrity runs) | **0** | 0 |

P2P on Turing is as fast as on the RTX 3080s here: GPU0's x4 chipset link sets the pace for both pairs. NCCL's
largest size is 512 MiB because of the 4 GB card. All numbers, how they were measured, and how repeatable they are:
[benchmarks.md](benchmarks.md#turing-pair-rtx-2060--gtx-1650-super).

## What did not work, and why

All of these were tried on the test rig, so you do not have to:

| Tried | What happened |
|---|---|
| The resize tool alone | `resize failed (No space left on device)` for both cards: the GPUs' own USB-C functions sat in the windows. Fixed by T2. |
| The tool's layout module (`--layout dryrun`) | Plan only. It would refuse anyway: the chipset's USB and SATA controllers and the GPUs' USB-C functions already have drivers when it runs, and it never touches the boot VGA card. Upstream has not tested it on real switch hardware, and on this board it would hold the network card. |
| Arm `0x610`, then a warm reboot, hoping the firmware sizes the window | The firmware still assigned 256 MB. |
| Remove the whole chipset root port and rescan, with the BAR1 size set by hand | **Hard freeze**, power cycle needed; not even a software reboot timer fired. Do not. |
| Smaller windows | No help: the patch needs BAR1 at least as large as VRAM, and the firmware's 289 MB windows had no room above 256 MB anyway. |

The cmp50hx-unlock project solved the same window problem on TU102 cards differently, with the kernel options
`pci=realloc,hpmemsize=8G` plus a firmware or kernel ReBAR patch. Not tried here.

For experiments like these, arm the AMD chipset's hardware watchdog first, so a frozen machine resets itself
(runtime only, gone after a reboot):

```bash
sudo modprobe sp5100_tco
sudo mkdir -p /run/systemd/system.conf.d
printf '[Manager]\nRuntimeWatchdogSec=60s\n' | sudo tee /run/systemd/system.conf.d/90-watchdog.conf
sudo systemctl daemon-reexec
```

## Undo

```bash
cd ~/geforce-dynbar1-p2p && sudo contrib/p2p-enable.sh --yes --undo 0000:04:00.0 0000:2b:00.0
sudo rm -r /etc/systemd/system/geforce-bar1-resize.service.d /usr/local/sbin/gpu-usbc-off
cd ~/geforce-bar1-resize && sudo ./uninstall.sh
sudo reboot
```

`uninstall.sh` removes every `RMForceStaticBar1=0` from the modprobe option file. If you swap back to RTX 30 cards,
write that file again ([procedure.md step 11b](procedure.md#11b-driver-dynamic-bar1-for-each-card-rmforcestaticbar10)).

Leaving the patched driver as well? Run [procedure.md step 18](procedure.md#18-back-to-stock) (`p2p-fallback.sh`)
**before** `uninstall.sh`: the fallback refuses to run once that option is gone (`ABORT: restore the
RMForceStaticBar1=0 option first`). On the test rig it had to be put back for a moment to finish the switch to driver
580.
