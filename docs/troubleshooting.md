# Troubleshooting

Every problem below came up in this order, on the test rig unless noted. The `static BAR1` and initramfs entries showed
up only on a fresh Ubuntu 26.04 install, the two `Turing:` entries only with Turing cards ([turing.md](turing.md)), the
`IO_PAGE_FAULT` entry on a machine with two Tesla V100s ([pascal-and-volta.md](pascal-and-volta.md); its Intel form,
`DMAR`, in an outside report with Tesla P40s), and the machine reset with vLLM on the Turing pair, two different chips
([benchmarks/vllm](../benchmarks/vllm/README.md)). The next two entries come from outside reports on Intel boards:
reads returning `0xffffffff`, and NCCL not using P2P (checked on the test rig). The last four come from an outside
report with four RTX 2080 Ti on an Intel X79 board: the board's 64-bit window, leftovers of Ubuntu's prebuilt modules,
settings the driver swap drops, and memory with three or more cards.

## `make modules` fails in the CXL code

**Symptom:** the build of `geforce-dynbar1-p2p` stops with errors in `nv-linux.h` / `nv-pci.h` around `struct nv_cxl`
(`struct cxl_dev_state`, `cxl_memdev`).

**Cause:** Linux 7.0 has `<cxl/cxl.h>` and `<cxl/pci.h>`, so the driver's conftest defines `NV_CXL_CXL_H_PRESENT`.
But the kernel's CXL API is not the one the 615.71.09 sources were written for. The headers are guarded only on
"header present", not on "API usable".

**Fix:** already in the fork as commit [`9504c3e`](https://github.com/Cinderella-Man/geforce-dynbar1-p2p/commit/9504c3e0237a2a46c24cb43664b1654b5222a3b3). It adds
`&& defined(NV_CXL_INIT_SUPPORTED)` to the four guards and is cherry-picked from
[Ubsefor's commit `0e23c574`](https://github.com/Ubsefor/open-gpu-kernel-modules/commit/0e23c574b430c27d7b5ae394bc6ca488aff346cd).
If you see this error, you are building plain upstream. Check that the fix is in your tree:

```bash
grep -n 'NV_CXL_CXL_H_PRESENT.*NV_CXL_INIT_SUPPORTED' kernel-open/common/inc/nv-linux.h kernel-open/common/inc/nv-pci.h
```

## `make modules` fails at the very end (BTF)

**Symptom:** everything compiles, and the final module step fails (`pahole: not found` / BTF generation).

**Cause:** Ubuntu's kernel is built with `CONFIG_DEBUG_INFO_BTF_MODULES=y`. Building an external module then needs
`pahole`, which is not installed by default.

**Fix:**

```bash
sudo apt install -y pahole
```

Then run `make modules` again.

With `pahole` installed, two kinds of lines still show up and are harmless: `nvidia-modeset.o: warning: objtool: ...`
and `Skipping BTF generation for nvidia-uvm.ko due to unavailability of vmlinux`.

## A CUDA program fails: `the provided PTX was compiled with an unsupported toolchain`

**Symptom:** for example `p2pBandwidthLatencyTest` prints the connectivity matrix, then
`Cuda failure ...: 'the provided PTX was compiled with an unsupported toolchain.'` Programs that only copy memory (no
GPU kernels), like this repository's `tools/`, are not affected.

**Cause:** the CUDA toolkit (13.5) is newer than what driver 615.71.09 supports (13.4; `nvidia-smi` shows
`CUDA UMD Version: 13.4`). A plain `nvcc` build of an RTX 3090 program carries PTX that the driver has to JIT-compile,
and the driver refuses PTX from a newer toolkit. It has nothing to do with P2P or the patched modules: the stock
driver does the same.

**Fix:** compile native code for your cards so nothing is JIT-compiled:

```bash
nvcc -O2 -arch=native ...
```

`-arch=sm_86` also works for RTX 30-series cards. Check with `cuobjdump --list-elf <binary>`: it must list an `sm_86`
cubin.

## `p2p-load.sh` refuses: GPUs still held

**Symptom:** `holders:` lists processes, then `ABORT: GPUs still held; nothing unloaded`.

**Cause:** the desktop (gnome-shell, Xwayland) and `nvidia-persistenced` keep `/dev/nvidia*` / `/dev/dri/*` open.

**Fix:** run over SSH, then:

```bash
sudo systemctl stop display-manager
sudo systemctl stop nvidia-persistenced && sudo systemctl mask nvidia-persistenced
sudo fuser -v /dev/nvidia* /dev/dri/* 2>&1
```

You have to *mask* persistenced, not just stop it. udev's `71-nvidia.rules` starts it again on every nvidia driver
`add` event (`ENV{SYSTEMD_WANTS}="nvidia-persistenced.service"`), and so does any unit with
`Wants=nvidia-persistenced.service`.

## The driver keeps reloading every few seconds

This was the big one: "something keeps restarting the kernel service".

**Symptom:** right after `p2p-load.sh` reports success, the kernel log repeats this block every ~4 seconds:

```
[nvidia-drm] [GPU ID 0x00002b00] Unloading driver
nvidia-modeset: Unloading
nvidia-nvlink: Unregistered Nvlink Core, major device number 510
NVRM: loading NVIDIA UNIX Open Kernel Module for x86_64  615.71.09  Release Build  (dvs-builder@...)
nvidia-modeset: Loading NVIDIA UNIX Open Kernel Mode Setting Driver ...
(udev-worker)[...]: nvidia: Process '/sbin/modprobe -r nvidia-modeset' failed with exit code 1.
```

`dvs-builder@...` is NVIDIA's stock build. Your own build shows `(<you>@<host>)`. On the test rig the patched module
was loaded once (18:16:47), and from then on only stock loads followed: about 1000 cycles until 19:26. The first
integrity run then failed because `nvidia` and `nvidia-uvm` were the stock ones:

```
FAIL nvidia loaded: '040E903CBCB6AA48C223679', not the srcversion of .../kernel-open/nvidia.ko
FAIL nvidia-uvm loaded: '4D46B930398C9ECEC64A213', not the srcversion of .../kernel-open/nvidia-uvm.ko
...
RESULT NO_P2P 0000:04:00.0 0000:2b:00.0
```

**Cause:** Ubuntu's `/usr/lib/udev/rules.d/71-nvidia.rules` (from the driver packages) contains:

```
ACTION=="add",    DEVPATH=="/bus/pci/drivers/nvidia", RUN+="/sbin/modprobe nvidia-modeset"
ACTION=="remove", DEVPATH=="/bus/pci/drivers/nvidia", RUN+="/sbin/modprobe -r nvidia-modeset"
(the same pair for nvidia-drm and nvidia-uvm)
```

`p2p-load.sh` pauses udev (`udevadm control --stop-exec-queue`), runs `rmmod` on the stock stack, insmods the patched
build, and resumes udev. On resume udev replays the queued `remove` event. `modprobe -r nvidia-uvm` takes the patched
`nvidia-uvm` out, and with it `nvidia`, which is now unused. The queued or next `add` event then runs
`modprobe nvidia-modeset`, which resolves to the **stock** `.ko.zst` in `updates/dkms/` and loads stock `nvidia` as a
dependency. Each unload emits a new `remove` and each load a new `add`, so the two rules keep each other going. (The
`modprobe -r` "failed with exit code 1" lines show up because the rules run in parallel and race each other. The
unloads still happen.)

**How it was found** (in case it comes back in another form):

```bash
# who unloads modules: the delete_module syscall with the calling process
sudo bpftrace -e 'tracepoint:syscalls:sys_enter_delete_module { printf("DELETE_MODULE pid=%d ppid=%d comm=%s module=%s\n", pid, curtask->real_parent->tgid, comm, str(args->name_user)); }'

# every modprobe that is started, with its arguments
sudo bpftrace -e 'tracepoint:syscalls:sys_enter_execve /comm == "modprobe"/ { printf("MODPROBE ppid=%d %s %s %s\n", curtask->real_parent->tgid, str(args->argv[1]), str(args->argv[2]), str(args->argv[3])); }'

# the parent turns out to be a (udev-worker); its environment names the event
sudo tr '\0' '\n' < /proc/<ppid>/environ | grep -E '^(ACTION|DEVPATH|SUBSYSTEM|DRIVER|SEQNUM)='

# the rules that match
sudo grep -RniE 'modprobe.*(-r|--remove)|rmmod' /etc/udev/rules.d /usr/lib/udev/rules.d
```

**Fix:** override the rules file in `/etc` with the three `remove` lines commented out, then run the load again.

1. If the loop is already running, freeze it first:

   ```bash
   sudo udevadm control --stop-exec-queue
   sudo pkill -f '/sbin/modprobe.*nvidia'
   ```

2. Install the override:

   ```bash
   sudo cp /usr/lib/udev/rules.d/71-nvidia.rules /etc/udev/rules.d/71-nvidia.rules
   sudo sed -i '/^ACTION=="remove".*modprobe -r/ s/^/# /' /etc/udev/rules.d/71-nvidia.rules
   sudo udevadm control --reload-rules
   ```

3. Load again (steps 12-13 of [procedure.md](procedure.md)). `p2p-load.sh` resumes the udev queue itself at its end.

After the override the next `p2p-load.sh` (19:29:30) loaded the patched build, it stayed loaded, and the integrity test
passed.

**Remove the override** once you no longer swap drivers at runtime:

```bash
sudo rm /etc/udev/rules.d/71-nvidia.rules && sudo udevadm control --reload-rules
```

## `p2p-fallback.sh` refuses: "restore the RMForceStaticBar1=0 option first"

**Cause:** the scripts expect the per-card `RMForceStaticBar1=0` line in modprobe.d. The geforce-bar1-resize tool would
normally write it, but a stock-ReBAR RTX 3090 does not need that tool.

**Fix:** write the line by hand. See [procedure step 11b](procedure.md#11b-driver-dynamic-bar1-for-each-card-rmforcestaticbar10).

## `p2p-fallback.sh` fails its verification: modules `MISSING` / `WANT stock`

**Symptom:**

```
modules.dep resolves:
  nvidia MISSING
  nvidia-modeset MISSING
  ...
verification failed
```

**Cause:** `resolution()` in `contrib/common.sh` only matched `/<module>.ko`. Ubuntu builds the stock DKMS modules as
`updates/dkms/nvidia.ko.zst` (`CONFIG_MODULE_COMPRESS_ZSTD=y`).

**Fix:** already in the fork as commit [`e5ad35d`](https://github.com/Cinderella-Man/geforce-dynbar1-p2p/commit/e5ad35de7fe517ffa85ff6d057777be7cb9ff465).

## `nvidia-smi` shows `GNS` again after a reboot

**Cause:** `p2p-load.sh` swaps the running driver only. After a reboot `modprobe` resolves the stock DKMS modules again.
That is how the test rig ended up the next morning:

```bash
for m in nvidia nvidia_uvm; do echo "$m loaded=$(cat /sys/module/$m/srcversion) stock=$(modinfo -F srcversion $m)"; done
```

When `loaded` equals `stock`, the stock driver is running.

**Fix:** the boot install, [procedure step 16](procedure.md#16-boot-install-make-it-survive-a-reboot).

## P2P works but the bandwidth is low (~3 GB/s)

**Cause:** the slowest PCIe link on the path. On the test rig GPU0 sits in a B550 chipset slot (`PCIe x4`, 3.0 through
the chipset) and GPU1 in the CPU's x16 slot:

```bash
for d in $(lspci -D -d 10de::0300 | awk '{print $1}'); do echo "$d $(cat /sys/bus/pci/devices/$d/current_link_speed) x$(cat /sys/bus/pci/devices/$d/current_link_width) (max $(cat /sys/bus/pci/devices/$d/max_link_speed) x$(cat /sys/bus/pci/devices/$d/max_link_width))"; done
```

Read the link while the GPUs are busy. At idle they drop to 2.5 GT/s to save power, so a reading of `2.5 GT/s` at idle
means nothing.

`./tools/pcie_bandwidth` measures each card's host link. P2P cannot be faster than the slower card's link.

## `Driver/library version mismatch` after an update

**Cause:** apt replaced the 615.71.09 userspace with another version, while the patched 615.71.09 kernel module is still
loaded.

**Fix:** hold the 615 packages ([procedure step 3](procedure.md#pin-the-driver-version)). If it already happened,
downgrade to 615.71.09 or fall back to stock (`p2p-fallback.sh --yes`, reboot).

## `p2p-check.sh --pairs` says `untested ... static BAR1 ... (no RMForceStaticBar1=0 ...)`

**Symptom:** right after [procedure step 11b](procedure.md#11b-driver-dynamic-bar1-for-each-card-rmforcestaticbar10),
`contrib/p2p-check.sh --pairs` prints `untested (GA102 + GA102): ... then enable with: p2p-enable.sh ...; static BAR1
on <both cards> (no RMForceStaticBar1=0 ...)`. It looks as if 11b did not work.

**Cause:** the check judges the options of the **running** driver (the first line of its output says so). That is
still the stock driver loaded at boot, before 11b wrote its file:

```bash
grep -E '^RegistryDwordsPerDevice:' /proc/driver/nvidia/params
```

prints `RegistryDwordsPerDevice: ""`.

**Fix:** nothing to fix, and do not run `p2p-enable.sh`. Check that 11b's file is what modprobe reads
(`sudo modprobe -c | grep NVreg_RegistryDwordsPerDevice`) and carry on. After the trial load (step 13) or any reboot,
`--pairs` prints `on by default, untested (GA102 + GA102 ...)`.

## `p2p-install.sh` refuses: `ABORT: the initramfs carries an nvidia module`

**Symptom:** the boot install ([procedure step 16](procedure.md#16-boot-install-make-it-survive-a-reboot)) stops in its
preflight with `ABORT: the initramfs carries an nvidia module; the driver must load from the root file system, after
the BAR1 resize`.

**Cause:** a fresh Ubuntu 26.04 install builds the initramfs with **dracut**. Its `drm` module (in hostonly mode, the
default) adds the KMS driver of every GPU it finds, so the stock `nvidia`, `nvidia-modeset`, `nvidia-drm` and
`nvidia-uvm` from `updates/dkms` end up in the initramfs, with their firmware. The stock driver then loads before the
root file system is mounted, and the patched modules installed there would never be used. A system upgraded from
24.04 keeps initramfs-tools, which leaves them out; that is why the original test rig never hit this. To see it:

```bash
lsinitramfs /boot/initrd.img-$(uname -r) | grep -E '(^|/)nvidia(-drm|-modeset|-uvm|-peermem)?\.ko'
```

```bash
journalctl -b -o short-monotonic | grep -E 'Switching root|loading NVIDIA UNIX'
```

The second one shows `loading NVIDIA UNIX` before `Switching root.`.

**Fix:** [procedure step 16a](procedure.md#16a-keep-the-nvidia-driver-out-of-the-initramfs): an `omit_drivers` line in
`/etc/dracut.conf.d/90-omit-nvidia.conf`, then `sudo update-initramfs -u -k "$(uname -r)"`. The script is right to
refuse; do not work around the check.

## Turing: `geforce-bar1-resize` logs `resize failed (No space left on device)`

**Symptom:** with RTX 20 or GTX 16 cards, `journalctl -b -u geforce-bar1-resize` shows `<card> resize failed (No space
left on device); left armed for the layout module (a PCIe switch above it?)`, and BAR1 stays at 256 MB. On the test
rig this happened in the CPU slot too, not only behind the chipset.

**Cause:** these cards have a USB-C controller (function `.2`) and a UCSI function (`.3`) next to the GPU, and their
64-bit prefetchable address ranges sit in the same bridge window as BAR1. The kernel cannot grow a window that holds
another device's range. To see them:

```bash
lspci -D -d 10de: | grep -iE 'usb|ucsi'
```

**Fix:** remove those two functions before the resize: [turing.md step T2](turing.md#t2-add-the-usb-c-pre-step). Do not
try the remove-and-rescan of the whole root port instead; on the test rig that froze the machine.

## Turing: `nvidia-smi topo -p2p r` shows `OK`, but CUDA has no peer access

**Symptom:** after a trial load with P2P on, `nvidia-smi topo -p2p r` shows `OK`, but `tools/p2p_access` prints `NO`,
and the integrity test stops at once with `RESULT NO_P2P` and `FAIL integrity ...: no peer access although the pair is
opted-in`.

**Cause:** the two cards are **different chips** of the same generation (on the test rig TU104 + TU116). The kernel
driver reports P2P as supported, but NVIDIA's closed CUDA library refuses peer access between GPUs whose chip
implementation differs.

**Fix:** the preload shim in [turing.md step T5](turing.md#t5-cuda-and-the-integrity-test), for the programs that
should use P2P. Two identical chips do not hit this; if they show the same symptom, the cause is something else.
Two different *architectures* (for example an RTX 2060 with an RTX 3060) the shim cannot help: CUDA refuses those
pairs outright. What works there: [mixed-generations.md](mixed-generations.md).

## Peer copies faster than the link allows, or `IO_PAGE_FAULT` from nvidia

**Symptom:** CUDA reports peer access and `nvidia-smi topo -p2p r` shows `OK`, but `tools/p2p_bandwidth` measures one
direction faster than the slower card's PCIe link can carry, or the integrity test fails. On two Tesla V100s with the
stock driver: 6.6 GB/s into a card whose PCIe 2.0 x4 link carries at most 2 GB/s. The kernel log has lines like

```
nvidia 0000:07:00.0: AMD-Vi: Event logged [IO_PAGE_FAULT domain=0x0012 address=0x7000000000 flags=0x0020]
```

On Intel the same block shows up as a `DMAR` fault. From an outside report, four Tesla P40s on a dual-socket X99 board
with the stock driver ([pascal-and-volta](pascal-and-volta.md#outside-report-four-tesla-p40s-on-a-dual-socket-x99)):

```
DMAR: [DMA Write NO_PASID] Request device [xx:00.0] fault addr ... [fault reason 0x05] PTE Write access is not set
```

There every copy, in both directions, came back wrong, and on PCIe 3.0 x16 links the speed looked normal: only the
kernel log and a byte-by-byte check showed it.

**Cause:** the IOMMU translates device addresses (Ubuntu's default wherever the firmware turns the IOMMU on), and it
blocks one GPU's writes into the other GPU's memory. The copy completes without the data, and neither the driver nor
CUDA reports an error: **the data is silently lost.**

**Fix:** IOMMU passthrough, [procedure step 4](procedure.md#4-iommu-in-passthrough-mode). Afterwards every IOMMU group
must be `identity`:

```bash
cat /sys/kernel/iommu_groups/*/type | sort | uniq -c
```

Then repeat the copy test (both directions now at or below the link speed) and the integrity test.

## `cudaGetDeviceCount: invalid device ordinal` with one GPU that failed to start

**Symptom:** every CUDA program fails at once with `cudaGetDeviceCount: invalid device ordinal` (`tools/p2p_access`
from before 2026-11-23 printed endless `GPU 0 -> GPU n: NO` lines instead), although `nvidia-smi` lists the GPUs. The
kernel log has, for one card:

```
NVRM: Xid (PCI:0000:05:00): 79, GPU has fallen off the bus.
NVRM: GPU 0000:05:00.0: RmInitAdapter failed! (0x25:0xf:1636)
```

**Cause:** one GPU failed to start, and CUDA then gives up on all of them; `CUDA_VISIBLE_DEVICES` does not help. On the
test rig it was a display card on a mining riser in a PCIe x1 slot, most likely without power on the riser.

**Fix:** connect the riser's power and reboot. Until the next reboot, removing the failed card from the PCI bus (every
function of it, here `05:00.0` and its audio function `05:00.1`) lets CUDA start on the other GPUs:

```bash
for f in /sys/bus/pci/devices/0000:05:00.*; do echo 1 | sudo tee $f/remove; done
```

## The machine resets: `data fabric sync flood`

**Symptom:** the whole machine freezes and resets while a program uses P2P. The kernel log just stops, and kdump saves
nothing. The next boot logs:

```
x86/amd: Previous system reset reason [0x08000800]: an uncorrected error caused a data fabric sync flood event
mce: [Hardware Error]: CPU 0: Machine Check: 0 Bank 5: bea0000000000108
```

(After a normal reboot the same line reads `software wrote 0x6 to reset control register 0xCF9`, with no machine check.)
On the test rig it first happened in vLLM with its custom all-reduce (on by default whenever CUDA offers P2P), on the
first prompt.

**Cause:** the trigger is the kernels of **both GPUs reading each other's memory at the same time**, with many threads.
Why the machine cannot take it is not known: it was seen only on a pair of **two different chips** with one card
**behind the chipset**, and either may be the reason (see below). [`tools/p2p_patterns.cu`](../tools/p2p_patterns.cu)
reproduces it without vLLM, on the Turing pair:

| Pattern | Result |
|---|---|
| One GPU's kernels read the other's memory, either direction | clean |
| One GPU's kernels read while the other's kernels write | clean |
| Copy engines, both directions at once, reading or writing (`cudaMemcpyPeerAsync`) | clean |
| Both GPUs' kernels read each other's memory at once, 1 to 24 blocks x 512 threads | clean |
| Both GPUs' kernels read each other's memory at once, **36 blocks x 512 threads** | **reset within a second, twice** |

A sync flood is how an AMD CPU stops when an error on its internal interconnect (the data fabric) cannot be recovered;
a read whose answer never comes back is a typical way to get one. vLLM's custom all-reduce has each GPU read the other's
buffer on every call, with 36 x 512 threads. A small message while generating keeps only a few threads busy and got
through; a 1 MiB message for a prompt did not. The integrity test does not catch this: it reads in one direction at a
time. Every result: [p2p-patterns.txt](../benchmarks/vllm/results/2026-11-24-turing-pair/p2p-patterns.txt).

**Tested only on the Turing pair**, and two things about it are unusual, so either may be the real reason:

- **two different chips** (TU104 + TU116). CUDA offers peer access between them only through the shim, which makes it
  treat both cards as the same chip, and CUDA may rely on that chip identity for more than the one check the shim
  changes ([turing.md T5](turing.md#t5-cuda-and-the-integrity-test));
- **GPU0 behind the B550 chipset**, so every read from the other card needs an answer back through the chipset.

A pair of identical cards in the same slots, which needs no shim, would tell the two apart; it has not been tested yet.
Until then, treat this as a risk for any pair, not as proven for any pair but this one.

An outside report, not verified here, points to the chipset. On an Intel i5-10400 with a B460 board, two identical RTX
4060 Ti 16GB cards (no shim needed), one on CPU lanes and one behind the chipset (DMI 3.0), driver 620.70.06 and a
different P2P method: peer writes were byte-exact both ways (5.44 GB/s at once, as here), but **every peer read came
back as `0xffffffff`**, with no error and no hang ([next entry](#peer-reads-return-0xffffffff-writes-work)). So on both
boards, reads through the chipset are the weak spot; they just fail differently. The Intel chipset does not forward them
at all. The B550 forwards them, and they work at light load, but heavy reads in both directions at once reset the
machine.

**Fix:** run vLLM with `--disable-custom-all-reduce`; with NCCL's P2P it ran clean for an hour. Other programs with their
own P2P all-reduce kernels may read the same way. In your own code, have each GPU write into the other's memory, or
use `cudaMemcpyPeerAsync`, rather than letting both GPUs' kernels read across at once.

To test a pair for it, **save your work first: on the test rig's mixed pair this resets the machine.**

```bash
nvcc -O2 -arch=native tools/p2p_patterns.cu -o tools/p2p_patterns
```

```bash
sync
```

```bash
tools/p2p_patterns rbi 30
```

For two different Turing chips, put the shim's two variables in front, as in
[turing.md T5](turing.md#t5-cuda-and-the-integrity-test). `BLOCKS=16 tools/p2p_patterns rbi 30` runs a lighter load.
Running `tools/p2p_patterns` without arguments lists the other modes. Every mode checks the data it moves and ends with
`DATA OK`, or `DATA WRONG` and exit status 1: run `r01` and `r10` first, to see whether reads work at all on your board.

## Peer reads return `0xffffffff`, writes work

From an outside report, not reproduced here: an Intel i5-10400 with a B460 board, one card on CPU lanes and one behind
the chipset (previous entry).

**Symptom:** writes into the other card's memory arrive byte-exact in both directions, but everything read from the
other card comes back as `0xffffffff`, in both directions. No error, no hang, nothing in the kernel log. Expect the
integrity test's read transfers (`load` and `pull`) and `tools/p2p_patterns r01` / `r10` to fail.

**Cause:** the chipset passes peer writes on but not peer reads. A write needs no answer; a read needs one, and the
chipset does not route it between its own port and the CPU's. `0xffffffff` is what a PCIe read gets back when it fails.

**Fix:** none in software. Either put both cards on CPU lanes (a board that splits them x8/x8), or use only software
that writes into the other card. NCCL shows its mode at the end of each `via P2P/...` line with `NCCL_DEBUG=INFO`:
`/read` means it reads (on the test rig it wrote, `via P2P/CUMEM`). Never trust P2P on such a board without the
integrity test: the wrong data arrives silently.

## NCCL goes `via SHM` although CUDA offers peer access

From an outside report on an Intel board, whose NCCL runs set `NCCL_P2P_LEVEL=PHB`
([pascal-and-volta](pascal-and-volta.md#outside-report-four-tesla-p40s-on-a-dual-socket-x99)), and NCCL's source.
NCCL's choice was checked on the test rig, and seen on Intel hardware in a second outside report (below).

**Symptom:** `nvidia-smi topo -p2p r` shows `OK`, CUDA reports peer access and the integrity test passes, but NCCL logs
`via SHM/direct` with `NCCL_DEBUG=INFO` and runs at host RAM speed. So do PyTorch and vLLM, which use NCCL. No error.

**Cause:** NCCL has its own rule on top of CUDA's. By default it uses P2P only between cards behind the same PCIe
switch, not through the CPU, except on an AMD CPU with at most two GPUs. Two cards in separate CPU slots, or one behind
the chipset, are a path through the CPU (`PHB` in `nvidia-smi topo -m`):

| CPU | Two cards through the CPU: NCCL's default |
|---|---|
| AMD, two GPUs (the test rig) | P2P |
| Intel | host RAM |
| AMD, three or more GPUs | host RAM (NCCL 2.26 and later; from the source, not tested) |

The rule is in `ncclTopoCheckP2p` (`src/graph/paths.cc`). The Intel row holds in every version checked, 2.21.5 to
2.32.6.

**Fix:** once the integrity test has passed, allow P2P through the CPU:

```bash
export NCCL_P2P_LEVEL=PHB
```

Then check the `via` line again ([benchmarks](benchmarks.md#4-nccl-and-nccl-tests)). The NCCL inside PyTorch and vLLM
reads the same variable. Upstream's README gives the same advice. (`SYS` would also allow pairs on two different CPU
sockets; the driver offers no P2P between those anyway.)

**Seen on Intel hardware:** with four RTX 2080 Ti on an X79 board
([turing.md](turing.md#outside-report-four-rtx-2080-ti-on-an-x79-board)), NCCL went `via SHM` by default and `via
P2P/CUMEM` with `NCCL_P2P_LEVEL=SYS`. On one socket `PHB` is enough.

**Checked on the test rig** (a Ryzen with an RTX 2060 + RTX 3060 on the stock driver 615, which offers this pair no
P2P, so NCCL always ends at host RAM there): NCCL logs `P2P is disabled between connected GPUs` only when its own rule
allowed the pair and the driver then said no. With NCCL's description of the machine changed to an Intel CPU
(`NCCL_TOPO_FILE`), that line disappeared, in NCCL 2.32.6 and in PyTorch's NCCL 2.30.7 and 2.29.3, and it came back
with `NCCL_P2P_LEVEL=PHB`. That shows NCCL's choice, not what it costs: on the test rig's RTX 30 pairs a 1 GiB
all-reduce ran at 2.72-2.75 GB/s with P2P and 1.85-1.88 GB/s through host RAM
([benchmarks](benchmarks.md#4-nccl-and-nccl-tests)).

## After the BAR1 resize every card fails: `kbusVerifyBar2_GM107: ... returned garbage`

From an outside report, not reproduced here: four RTX 2080 Ti on an Intel X79 board with a Xeon E5-2680 v2
([turing.md](turing.md#outside-report-four-rtx-2080-ti-on-an-x79-board)).

**Symptom:** with Turing cards, after geforce-bar1-resize grew the windows at boot
([turing.md T1-T3](turing.md#t1-install-geforce-bar1-resize-pinned)), the driver fails on every card:

```
kbusVerifyBar2_GM107: MMUTest BAR0 window offset 0x70e000 returned garbage 0x0
RmInitAdapter failed
```

With the stock 256 MB windows the same machine booted fine.

**Cause:** the resize moved the windows to addresses the CPU does not pass on to PCIe. Linux places them anywhere in
the 64-bit range the firmware declares for the PCI root (ACPI `_CRS`; in `dmesg` as `root bus resource [mem ...
window]`). On this board that range starts at `0xa40000000`, but the Xeon E5 v2 forwards 64-bit addresses only from
its MMIOH range, `0x380000000000-0x381fffffffff` (128 GiB, read from the CPU's registers). The firmware had put the 256
MB windows there; after the resize Linux took the lowest free addresses, from `0xc00000000` on, and the driver's first
read through them got nothing back.

To see where your windows are: the firmware's placement at boot comes first in the log, the resize's after it.

```bash
sudo dmesg | grep -E 'root bus resource \[mem'
```

```bash
for d in $(lspci -D -d 10de::0300 | awk '{print $1}'); do sudo dmesg | grep -E "$d: BAR [13] "; done
```

Windows that the resize moved far below where the firmware had them (here from `0x381f...` to `0xc...`) point to this.

**Fix (the reporter's):** reserve the part of the range below MMIOH, so Linux leaves it alone. The kernel option is
`memmap=<size>$<start>`, with `<start>` the start of the 64-bit root range and `<size>` the MMIOH base minus that
start. On this board, `0x380000000000 - 0xa40000000`:

```
memmap=0x37F5C0000000$0xa40000000
```

Your values differ: take `<start>` from the `root bus resource` line above, and the MMIOH base from where the firmware
put the 64-bit windows at boot (or from the CPU's registers). In `/etc/default/grub` the `$` must be written `\\\$`:

```
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash intel_iommu=on iommu=pt memmap=0x37F5C0000000\\\$0xa40000000"
```

After `sudo update-grub` and a reboot, `cat /proc/cmdline` must show `memmap=0x37F5C0000000$0xa40000000`, and
`sudo dmesg | grep 'E820 reservations'` must say `Using E820 reservations for host bridge windows`. Linux does that by
default when the BIOS is older than 2023; with a newer BIOS add `pci=use_e820`. On the reporter's board the four
windows then landed at `0x3800`, `0x3808`, `0x3810` and `0x3818_0000_0000`, and every card came up. That filled MMIOH:
each card's 16 GiB BAR1 took 32 GiB of addresses there, so larger windows do not fit on that board.

**Do not unload the driver to retry.** With every card failed, `modprobe -r nvidia_drm nvidia_modeset nvidia_uvm
nvidia` crashed the kernel there (stock 615.71.09: `BUG: unable to handle page fault ... RIP: gpuGetRpc+0x2e
[nvidia]`, after `Internal device object leak`). After `RmInitAdapter failed`, reboot.

## `p2p-load.sh`: `nvidia.ko.sig not found`, then `neither the stock driver nor the build`

From an outside report (the X79 board above).

**Symptom:** `p2p-load.sh` stops in its preflight with

```
modinfo: ERROR: Module /lib/modules/<kernel>/kernel/nvidia-595/bits/nvidia.ko.sig not found.
ABORT: loaded nvidia (srcversion ...) is neither the stock driver nor the build in .../kernel-open
```

**Cause:** files left by Ubuntu's prebuilt 595 modules. `linux-objects-nvidia-595-<kernel>` and
`linux-signatures-nvidia-<kernel>` put object files and `nvidia.ko.sig` into `kernel/nvidia-595/bits/`, and they stay
installed after the switch to 615. The scripts look for the stock module by its file name, `nvidia.ko*`, and can take
the `.sig` for it. `p2p-rollback.sh` can trip over the same files.

**Fix:** remove those packages
([procedure step 3](procedure.md#leftovers-of-ubuntus-prebuilt-modules)), then run `p2p-load.sh` again. Fixed in
the fork at `4c5a97a`, pinned since 2026-11-25: the scripts take only module files, so a build from it is not stopped by
these files. Removing the packages is still the clean state.

## Cards fall off the bus after `p2p-load.sh` (`Xid 79`)

From an outside report: four RTX 2080 Ti that need a clock lock to run stable.

**Symptom:** the cards ran stable on the stock driver, but after the live swap, under load, they drop off the bus:
`NVRM: Xid (PCI:...): 79, GPU has fallen off the bus`.

**Cause:** the swap unloads the driver, and with it every setting `nvidia-smi` made at run time: power limits,
locked clocks, persistence mode. With `nvidia-persistenced` masked
([procedure step 12b](procedure.md#12b-stop-and-mask-nvidia-persistenced)), a setting made afterwards also lasts only
while a program holds the card.

**Fix:** reboot if a card already fell off the bus. After each live swap, set persistence mode first and then your
settings, or put them in `RESTORE_CMD`
([procedure step 13](procedure.md#13-trial-load-nothing-is-installed-a-reboot-returns-to-stock)).

## Three or more cards: `cudaMalloc` runs out of memory with peer access between all of them

From an outside report (four RTX 2080 Ti, the X79 board above), explained from the patched driver's source; not
reproduced here, the test rig has two slots.

**Changed in the fork at `b70b69c`**, pinned since 2026-11-25: the count described below is now kept per other card,
so each card can map all of every other card's memory (BAR1 is at least as large as VRAM). Checked on three RTX 3090s:
the old build refused any allocation above 16352 MiB per card, `b70b69c` allowed 23552
([three-cards.md](three-cards.md#peer-memory-old-and-new-build)). This entry describes builds up to `25c929e`.

**Symptom:** with peer access between every pair of three or more cards (for example llama.cpp with
`GGML_CUDA_P2P=1`), `cudaMalloc` fails with out of memory long before the cards are full. `nvidia-smi` shows free
memory and free BAR1, and the kernel log has nothing (from `b70b69c` on, a refusal logs `METHOD3: peer window
refused`). On four RTX 2080 Ti (16 GiB BAR1 each) about 5.3 of 11 GB per
card were usable. The integrity test passes, because it tests one pair at a time.

**Cause:** a check in the patched driver. Every allocation another card can reach gets a window in its owner's BAR1,
once, however many cards use it. Before it makes one, the driver adds up the peer memory the *mapping* card already
has open on *all* other cards, and compares that sum with the BAR1 size of *one* card, minus 64 MiB
(`_nvGpuOpsDynBar1Create` in `nv_gpu_ops.c`; upstream calls it a rough early guard). With two cards that is the right
bound. With four, each card can map at most 16 GiB - 64 MiB of the other three cards' memory together:

| Allocated on each of four cards, every card mapping every other | Each card maps | Result (outside report) |
|---|---|---|
| 3000-5000 MiB | up to 15000 MiB | all ok |
| 5500-6400 MiB | 16500 MiB and more | the third card's allocation fails |
| 8192 MiB | 16384 MiB | the second card's allocation fails |

**Fix:** build from the fork at `b70b69c` or later (the pinned commit). With an older build, give peers access to the
buffers they exchange, not to whole cards. llama.cpp built with NCCL, without
`GGML_CUDA_P2P`, does that (NCCL maps only its own buffers), and the outside report then used all of the cards'
memory. Peer access only within pairs also works. A larger BAR1 raises the limit, where the board has room for it.
Static BAR1 is no way around it: the patched driver refuses P2P for a card with static BAR1.
