// SPDX-License-Identifier: MIT
//
// chipid-shim: EXPERIMENTAL LD_PRELOAD shim that lets CUDA use P2P between two *different* chips of the same GPU
// architecture (for example an RTX 2060 / TU104 and a GTX 1650 SUPER / TU116).
//
// Why: with the patched driver, the kernel reports P2P as supported for such a pair (nvidia-smi topo -p2p shows OK),
// but libcuda refuses peer access between GPUs whose chip "implementation" differs: cudaDeviceCanAccessPeer stays 0.
// libcuda reads each GPU's architecture and implementation through the RM control NV2080_CTRL_CMD_MC_GET_ARCH_INFO
// (0x20801701). This shim rewrites the implementation in that answer, inside the calling process only, so libcuda
// sees one chip.
//
// What it does NOT do: it changes nothing in the kernel, the driver or the GPUs, and nothing for other processes.
// It does nothing at all unless P2P_CHIP_IMPL_MAP is set, and it only remaps GPUs of the architecture named there.
//
// Risk: libcuda may use the chip implementation for other decisions. On the test rig the integrity test (240 cases,
// 636 GiB byte-exact) and NCCL ran clean with it, but that is one pair and one driver version. Run the integrity test
// on your pair with the shim before trusting it. Do not put it in /etc/ld.so.preload.
//
// Build:  gcc -O2 -shared -fPIC chipid-shim.c -o libchipid-shim.so -ldl
// Use:    P2P_CHIP_IMPL_MAP=0x160:0x8:0x4 LD_PRELOAD=/path/to/libchipid-shim.so <your CUDA program>
//         (ARCH:FROM:TO: Turing 0x160, TU116 0x8 reported as TU104 0x4; find your values with
//         P2P_CHIP_IMPL_VERBOSE=1, see docs/turing.md)
// Debug:  P2P_CHIP_IMPL_VERBOSE=1 prints every chip id libcuda reads, and every remap.

#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/ioctl.h>

#define NV_IOCTL_MAGIC 'F'
#define NV_ESC_RM_CONTROL 0x2a
#define NV2080_CTRL_CMD_MC_GET_ARCH_INFO 0x20801701u

typedef struct {               // NVOS54_PARAMETERS (RM control ioctl)
    uint32_t hClient, hObject, cmd, flags;
    uint64_t params;
    uint32_t paramsSize, status;
} nvos54_t;

typedef struct {               // NV2080_CTRL_MC_GET_ARCH_INFO_PARAMS (head)
    uint32_t architecture, implementation, revision;
} arch_info_t;

static int (*real_ioctl)(int, unsigned long, ...);
static int configured = -1;    // -1 not parsed yet, 0 off, 1 on
static unsigned long map_arch, impl_from, impl_to;
static int verbose;

static void parse_config(void)
{
    const char *m = getenv("P2P_CHIP_IMPL_MAP");
    char *end;
    verbose = getenv("P2P_CHIP_IMPL_VERBOSE") != NULL;
    configured = 0;
    if (!m || !*m) return;
    map_arch = strtoul(m, &end, 0);
    if (*end != ':') goto bad;
    impl_from = strtoul(end + 1, &end, 0);
    if (*end != ':') goto bad;
    impl_to = strtoul(end + 1, &end, 0);
    if (*end) goto bad;
    configured = 1;
    return;
bad:
    fprintf(stderr, "chipid-shim: P2P_CHIP_IMPL_MAP must be ARCH:FROM:TO, e.g. 0x160:0x8:0x4; doing nothing\n");
}

int ioctl(int fd, unsigned long req, ...)
{
    va_list ap;
    va_start(ap, req);
    void *arg = va_arg(ap, void *);
    va_end(ap);

    if (!real_ioctl) real_ioctl = (int (*)(int, unsigned long, ...))dlsym(RTLD_NEXT, "ioctl");
    int rc = real_ioctl(fd, req, arg);

    if (configured < 0) parse_config();
    if ((configured || verbose) && arg && ((req >> 8) & 0xff) == NV_IOCTL_MAGIC && (req & 0xff) == NV_ESC_RM_CONTROL) {
        nvos54_t *p = arg;
        if (p->cmd == NV2080_CTRL_CMD_MC_GET_ARCH_INFO && p->status == 0 && p->params &&
            p->paramsSize >= sizeof(arch_info_t)) {
            arch_info_t *a = (arch_info_t *)(uintptr_t)p->params;
            if (verbose)
                fprintf(stderr, "chipid-shim: GPU object 0x%08x architecture 0x%x implementation 0x%x\n",
                        p->hObject, a->architecture, a->implementation);
            if (configured && a->architecture == map_arch && a->implementation == impl_from) {
                a->implementation = (uint32_t)impl_to;
                if (verbose)
                    fprintf(stderr, "chipid-shim: reported implementation 0x%lx as 0x%lx\n", impl_from, impl_to);
            }
        }
    }
    return rc;
}
