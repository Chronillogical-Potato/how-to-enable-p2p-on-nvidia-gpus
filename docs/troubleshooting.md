# Troubleshooting

Every problem below happened on the test rig, in the order they came up. The `static BAR1` and initramfs entries showed
up only on a fresh Ubuntu 26.04 install, and the last two only with Turing cards ([turing.md](turing.md)).

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
