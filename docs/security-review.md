# Security review of the fork

Before loading a third-party kernel driver as root, it is worth knowing what is in it. This is a review of
[Cinderella-Man/geforce-dynbar1-p2p](https://github.com/Cinderella-Man/geforce-dynbar1-p2p) at
`e5ad35de7fe517ffa85ff6d057777be7cb9ff465`, done on 2026-11-23, and extended on 2026-11-24 to
`21f0c27555dfb12bbbdd21430084440541369e23`, the commit [procedure.md](procedure.md#8-get-the-patched-driver-source)
now pins: one more commit, a change to one script and its tests ([below](#verdict-no-malicious-code-found)).

## Scope: only what is not NVIDIA's code

The fork's history contains NVIDIA's official
[open-gpu-kernel-modules](https://github.com/NVIDIA/open-gpu-kernel-modules) tag `615.71.09` (commit
`61dcc93722ecb418bb5f2e00923f05b4b8051dd1`, checked against NVIDIA's repository). So everything that is not NVIDIA's
code is exactly this diff:

```bash
git diff 61dcc93722ecb418bb5f2e00923f05b4b8051dd1 21f0c27555dfb12bbbdd21430084440541369e23
```

That is 26 commits by noon-at-cgn (10), dingiv (7), aikitoria (4), Duanyll (2), the fork owner (2) and Ubsefor (1). It
touches 133 files, about 6,200 lines without docs: roughly 3,000 in the kernel driver, 2,000 in shell scripts, and the
rest tests and packaging. There are **no binary files** in the diff.

## Verdict: no malicious code found

The whole diff was read hunk by hunk (kernel driver, contrib scripts, integrity test, packaging, tests), and grepped for
red flags:

- **Kernel driver:** no network or file I/O from the kernel (`sock_*`, `filp_open`, `kernel_read/write`, `vfs_*`), no
  process spawning (`call_usermodehelper`), no credential or capability changes, no new ioctls, no encoded blobs. The
  changes are dynamic BAR1 windows, forced P2P capabilities, the UVM BAR1 gate, and hugepage pinning: all P2P, BAR1 or
  UVM work. Several existing bounds checks on mmap paths were made stricter, not weaker.
- **Scripts:** no network access (a report URL is only printed). Writes go only to the documented paths: `updates/p2p`,
  `/etc/depmod.d/p2p.conf`, `/etc/modprobe.d/zz-*.conf`, `/run/modprobe.d/`, `/var/backups/...`,
  `/var/lib/geforce-dynbar1-p2p`. Every `rm -rf` targets a fixed path or a temp directory.
- **Integrity test** (`contrib/integrity/p2p-integrity.cu`): only CUDA allocation, copies, kernels and compares. No
  file, network or exec I/O. Its report (CPU, board, PCIe paths, no UUIDs) stays local.
- **Build and packaging:** Kbuild/Makefile only add an include path and an optional debug define. DKMS packaging builds
  offline.
- **Hidden tricks:** no zero-width or bidirectional Unicode in the diff. `.github/` holds only issue templates and a
  funding file, no CI workflows.
- **The fork's own three commits** (`9504c3e`, `e5ad35d`, `21f0c27`) are a 4-line `#if` change, a 14-line `awk`
  change, and a change to `contrib/p2p-check.sh` (9 lines: it counts the pairs the integrity test ran, and a run that
  tests none is a FAIL) with its stand-in tests. The last one adds no commands, paths, writes or network access.

What was **not** covered: the correctness of the new locking and refcounting (bugs there would mean instability, not
malice), and the external `geforce-bar1-resize` tool, which only the Turing steps use; [turing.md](turing.md#t1-install-geforce-bar1-resize-pinned)
notes what was checked in it before it ran on the test rig.

## Risky but legitimate: know these

1. **P2P safety checks are bypassed by design.** On GA102 the driver forces P2P read/write over BAR1 and skips NVIDIA's
   chipset and topology checks. On an untested platform that can mean silent data corruption or hangs. That is why the
   integrity test (step 14) is not optional.
2. **IOMMU in passthrough mode** (`iommu=pt`) lets the GPUs DMA into any host memory. It is standard for P2P setups,
   but it is a host-security trade-off.
3. **Do not use the top-level `install.sh`.** It runs `sudo make modules_install` over the stock modules with no backup
   and no checksum pinning. Use the `contrib/` flow from [procedure.md](procedure.md), which backs up, pins and can
   fall back.
4. **The scripts run your config as root.** `contrib/config.sh` is sourced, and `DRAIN_CMD`/`RESTORE_CMD` run via
   `sh -c`. Keep the checkout owned by you and not writable by others.
5. **Known low-severity issues upstream documents itself:** a BAR1 range can leak on a lock failure, and memory-safety
   assumptions in the hugepage pin path hold only while `os_lock_user_pages` is the only producer. Neither is
   reachable by unprivileged users.

## Re-checking a newer version

If the fork moves to a newer driver, repeat the same scoping: find NVIDIA's official tag commit, confirm it is an
ancestor of the fork's HEAD (`git merge-base --is-ancestor <tag-commit> HEAD`), and review only that diff.
