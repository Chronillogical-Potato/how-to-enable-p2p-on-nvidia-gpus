# What was done on the test rig (2026-11-22)

This was reconstructed from the shell history, apt/dpkg logs, the systemd journal of that boot, and the reports of the
upstream scripts. Times are CET. It records where [procedure.md](procedure.md) comes from, including the dead ends.

## Hardware

| | |
|---|---|
| GPUs | 2x ASUS GeForce RTX 3090 (`10de:2204`, subsystem `1043:87b5`, vBIOS `94.02.42.00.B2`), 24 GiB, BAR1 32 GiB (Resizable BAR) |
| CPU / board | AMD Ryzen 9 3900X, MSI MPG B550 GAMING PLUS |
| PCIe | GPU0 `04:00.0`: B550 chipset slot, x4. GPU1 `2b:00.0`: CPU slot, x16. `nvidia-smi topo -m`: `PHB` |
| OS | Ubuntu 26.04.1 LTS, kernel `7.0.0-44-generic`, GCC 15.2 |
| Driver | `nvidia-driver-615-open` 615.71.09-0ubuntu0.26.04.2 (DKMS, open kernel modules) |
| CUDA | 13.5.0 (`cuda-toolkit-13-5`, local repo deb) |
| Boot | Secure Boot disabled; `amd_iommu=on iommu=pt` |

## Timeline

### 15:16-16:46: OS and driver

- 15:16: installed from the **Ubuntu 24.04.4** ISO. The installer's third-party driver option pulled
  `nvidia-driver-595-open`.
- 15:59: `apt update && apt upgrade`.
- 16:03-16:23: **release upgrade 24.04 -> 26.04** with the GUI upgrader. It brought kernel `7.0.0-44-generic`.
- 16:29-16:33: remote access tooling (not part of this guide).
- 16:36: `ubuntu-drivers devices`, then `sudo apt install nvidia-driver-615`. That replaced the 595 packages and
  installed `nvidia-dkms-615-open`, `nvidia-firmware-615-615.71.09`, and the rest of the 615 stack. Reboot.
- 16:45: a `gpu-power-limit.service` (oneshot `nvidia-smi -pm 1` and `-pl 250` per card, with
  `Wants=nvidia-persistenced.service`) was created and enabled. It is not needed for P2P, but it later played a part in
  keeping the GPUs busy.

### 16:46-17:58: baseline, CUDA, IOMMU

- `nvidia-smi topo -p2p r` / `-p2p w`: `GNS` between the cards. `nvidia-smi nvlink -s`: nothing (no bridge).
- A small `cudaDeviceCanAccessPeer` program ([`tools/p2p_access.cu`](../tools/p2p_access.cu)) needed `nvcc`, so
  **CUDA 13.5** was installed from NVIDIA's local repo deb (17:01) and `/usr/local/cuda/bin` added to `PATH`. Result:
  `NO` both ways.
- Checked Resizable BAR (`lspci -vv`: BAR1 32 GiB on both cards), Secure Boot (off), and the IOMMU (`dmesg | grep
  AMD-Vi`).
- Added `amd_iommu=on iommu=pt` to `GRUB_CMDLINE_LINUX_DEFAULT`, ran `update-grub`, rebooted (17:58).

### 18:02-18:14: build

- `apt install git build-essential dkms linux-headers-$(uname -r) pciutils`
- `git clone https://github.com/noon-at-cgn/geforce-dynbar1-p2p.git` (HEAD `ab0ee7d4`)
- `make modules`: **failed** in the CXL code. Fixed with Ubsefor's patch:
  `curl -L .../0e23c574...patch -o /tmp/nvidia-cxl-fix.patch && git apply /tmp/nvidia-cxl-fix.patch`
  (now commit [`9504c3e`](https://github.com/Cinderella-Man/geforce-dynbar1-p2p/commit/9504c3e0237a2a46c24cb43664b1654b5222a3b3) in the fork).
- `make modules`: **failed** again, this time because `pahole` was missing. 18:08: `apt install pahole`.
- `make modules`: **OK**. Then `sha256sum nvidia*.ko > SHA256SUMS`.
- Edited `contrib/config.sh`: `GPUS="0000:04:00.0 0000:2b:00.0"`, `OUT=<repo>/kernel-open`.
- `contrib/p2p-check.sh --pairs`, then checked the GSP firmware in `/lib/firmware/nvidia/615.71.09/`.

### 18:14-18:19: first trial load

- `sudo env P2P_ENABLE=1 contrib/p2p-load.sh --dry-run`
- `--yes`: refused, because the desktop held the GPUs. `systemctl stop display-manager`.
- `--yes`: refused, because `nvidia-persistenced` held them. `systemctl stop nvidia-persistenced`.
- 18:16:38 `--yes`: the patched module loaded at 18:16:47. A second later udev unloaded it, and stock `nvidia` came
  back. **The reload loop started** (one unload/reload about every 4 s).
- 18:19 `sudo contrib/p2p-check.sh --integrity`: 9 FAIL lines. `nvidia`/`nvidia-uvm` were the stock modules, there was
  no per-device entry, and the result was `NO_P2P`.

### 18:25-18:37: fallback attempts, two script problems

- `p2p-fallback.sh --yes --live`: refused with *restore the RMForceStaticBar1=0 option first*. The scripts expect
  geforce-bar1-resize's per-card line.
- 18:30: wrote `/etc/modprobe.d/zz-geforce-bar1-resize.conf` by hand
  (`options nvidia NVreg_RegistryDwordsPerDevice="pci=0000:04:00.0;RMForceStaticBar1=0;pci=0000:2b:00.0;RMForceStaticBar1=0;"`).
- `p2p-fallback.sh --yes --live` again: its verification reported every stock module `MISSING`. Cause: Ubuntu's
  `.ko.zst`. 18:37: patched `resolution()` in `contrib/common.sh` (now commit [`e5ad35d`](https://github.com/Cinderella-Man/geforce-dynbar1-p2p/commit/e5ad35de7fe517ffa85ff6d057777be7cb9ff465) in the
  fork).

### 18:37-19:26: hunting the reload loop

The kernel log showed the stock driver loading about every 4 seconds, roughly 150 times per 10 minutes. What was tried:

- `udevadm monitor --kernel --udev --property`: a steady stream of `remove`/`add` events for
  `/bus/pci/drivers/nvidia`.
- `ps --forest`, `/proc/<pid>/stack` of stuck `modprobe` processes, `journalctl -u systemd-udevd`.
- `auditctl` (not installed), then **bpftrace** on `sys_enter_delete_module` and on `execve` of `modprobe`. The
  unloading `modprobe -r` was started by a `(udev-worker)`. Its `/proc/<pid>/environ` named the `remove` event of the
  nvidia driver.
- `cat /usr/lib/udev/rules.d/71-nvidia.rules`: the `ACTION=="remove" ... RUN+="/sbin/modprobe -r ..."` lines.
- Several more `p2p-load.sh --yes` runs with `udevadm control --stop-exec-queue` around them. All were undone as soon as
  the queue resumed.
- 19:19: `systemctl mask nvidia-persistenced`, because udev kept starting it again.

### 19:26-19:40: the fix and a passing test

- `udevadm control --stop-exec-queue` and `pkill` of the hanging `modprobe` processes. The loop stopped at 19:26:14.
- `cp /usr/lib/udev/rules.d/71-nvidia.rules /etc/udev/rules.d/` and commented out the `ACTION=="remove".*modprobe -r`
  lines, then `udevadm control --reload-rules`.
- 19:29:21 `sudo env P2P_ENABLE=1 contrib/p2p-load.sh --yes`: patched build loaded at 19:29:30 and **stayed**.
- 19:30 `sudo contrib/p2p-check.sh --integrity`: **ALL PASS** (264 cases, 1403.6 GiB copied, A->B 3.27 GB/s, B->A
  2.96 GB/s, both 5.53 GB/s). `nvidia-smi topo -p2p r`: `OK`.

### 19:40-22:09: checks, cleanup, shutdown

- `./p2p` (the `cudaDeviceCanAccessPeer` program) ran again. The kernel log for the rest of the boot has no Xid, AER,
  DMAR, `RmInitAdapter` or `DynBar1P2P:` lines.
- `rm /etc/udev/rules.d/71-nvidia.rules` and `udevadm control --reload-rules`. The override was removed again. That is
  harmless while the patched modules are loaded, but it is needed for the next live swap.
- `lspci` link check, plus [`tools/pcie_bandwidth.cu`](../tools/pcie_bandwidth.cu) and
  [`tools/p2p_bandwidth.cu`](../tools/p2p_bandwidth.cu).
- `shutdown`.

## State the next morning (2026-11-23)

After the reboot the rig runs the **stock** modules again (`nvidia` srcversion `040E903C...` = the DKMS
`.ko.zst`), and `nvidia-smi topo -p2p r` shows `GNS`. `p2p-install.sh` (the boot install) was never run. These are left
over from the session:

| What | State |
|---|---|
| `/etc/modprobe.d/zz-geforce-bar1-resize.conf` (`RMForceStaticBar1=0` per card) | present; the stock driver boots fine with it |
| `nvidia-persistenced` | masked |
| `/etc/udev/rules.d/71-nvidia.rules` override | removed |
| `amd_iommu=on iommu=pt` | in GRUB |
| `contrib/common.sh` patch, CXL patch, `config.sh` edits | in the build tree, uncommitted |
| `/var/lib/geforce-dynbar1-p2p/integrity-pass` | one PASS for the pair with this build |

## Re-run from the fork (2026-11-23)

The whole of [procedure.md](procedure.md) was then followed on the same rig, from a new folder, with a clone of the
fork at `e5ad35d` (no hand patching):

| Step | Result |
|---|---|
| 1-7 | checks as documented; the 615 packages put on hold (they were not before) |
| 8-10 | clone and `make modules` OK in 47 s, first try. `nvidia`/`nvidia-uvm` srcversions identical to the hand-patched build of the day before |
| 11 | the generated `zz-geforce-bar1-resize.conf` is byte-identical to the hand-written one |
| 12-13 | with the udev override in place the live swap held on the first try: 2 driver loads this boot, 0 udev `modprobe -r` |
| 14 | integrity test ALL PASS (264 cases, 1403.6 GiB, 10.4 min) |
| 15 | test programs OK; `p2pBandwidthLatencyTest` needed v13.3 and `-arch=native` (see troubleshooting) |
| 16 | boot install, reboot: patched build loaded at boot (1 driver load), `p2p-check.sh` ALL PASS. Then udev override removed and persistenced unmasked, rebooted again: still patched, P2P `OK`, `p2p-check.sh` and the integrity test ALL PASS |

Final state of the rig: boot install for `7.0.0-44-generic` in `/lib/modules/7.0.0-44-generic/updates/p2p`, Ubuntu's
default udev rules, `nvidia-persistenced` running, 615 packages on hold.

## Blind run on a fresh Ubuntu 26.04.1 install (2026-11-23)

The rig was then wiped and installed from the Ubuntu 26.04.1 image (third-party drivers ticked, so it started with the
595 driver). A new session, with no knowledge beyond this repository, followed procedure.md steps 1-16 in order and
logged every deviation without editing the docs.

| Step | Result |
|---|---|
| 1-5 | as documented; step 3 replaced the installer's 595 driver. The kernel logged `AMD-Vi: Unknown option - 'on'`: `amd_iommu=on` is not a valid AMD option (harmless; dropped from step 4 since) |
| 6 | **failed**: `git: command not found` (a fresh install has no git; the doc installed it only in step 7). Fixed: step 6 now installs it |
| 7-10 | as documented (build 48 s; more harmless warnings than were listed) |
| 11c | **differed**: `--pairs` judges the running stock driver, so it reported `static BAR1 ... no RMForceStaticBar1=0`. Expected on a first run; documented since |
| 12-15 | as documented; the integrity test passed (264 cases, 1403.6 GiB, 3.27 / 2.96 / 5.53 GB/s, 10.4 min). The doc's `modprobe -r` counter gave false positives; replaced by the driver-load count |
| 16 | **failed**: `ABORT: the initramfs carries an nvidia module`. Fresh 26.04 builds the initramfs with dracut, which carried the stock modules (they loaded at 6.09 s, before the root switch at 6.92 s). Fixed by step 16a (dracut `omit_drivers`); then 16 ran as documented |
| 16, two reboots | patched driver loaded at boot (after the root switch), 1 driver load, `p2p-check.sh` ALL PASS, integrity test ALL PASS again with udev and `nvidia-persistenced` back to defaults |

Not run in the blind run: the NVIDIA benchmarks of benchmarks.md, and steps 17-18.
