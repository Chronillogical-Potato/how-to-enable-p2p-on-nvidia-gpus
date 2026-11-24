"""Split each GPU's time in the profiled prefill.py runs into model compute, NCCL and idle.

usage: python3 profile_kernels.py <p2p-off-dir> <p2p-on-dir>

Each directory holds the PyTorch profiler traces that vLLM's workers write (one per GPU, prefill.py with
TORCH_PROFILE_DIR, as profile.sh runs it). The window is the profiled part: from the first GPU kernel to the last.

- Time inside an NCCL kernel includes waiting for the other GPU, so on the faster card it is mostly waiting.
- "Idle" is the part of the window in which the GPU ran no kernel at all: CPU-side work in vLLM between steps, or
  the short pauses between the timed runs.
"""
import json
import pathlib
import sys


def kernels(trace_dir):
    by_device = {}
    for f in sorted(pathlib.Path(trace_dir).rglob("*.json")):
        events = json.loads(f.read_text()).get("traceEvents", [])
        for ev in events:
            if ev.get("ph") == "X" and ev.get("cat") == "kernel":
                dev = ev.get("args", {}).get("device", ev.get("pid"))
                by_device.setdefault(dev, []).append((ev["ts"], ev["ts"] + ev["dur"], ev["name"]))
    if not by_device:
        sys.exit(f"{trace_dir}: no GPU kernels in any trace; run profile.sh")
    return by_device


def busy(intervals):
    total, cur_s, cur_e = 0.0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    return total + (cur_e - cur_s if cur_e is not None else 0.0)


def summarize(trace_dir, label):
    print(f"== {label}")
    for dev, ks in sorted(kernels(trace_dir).items()):
        window = max(k[1] for k in ks) - min(k[0] for k in ks)
        nccl = sum(k[1] - k[0] for k in ks if "nccl" in k[2].lower())
        compute = sum(k[1] - k[0] for k in ks if "nccl" not in k[2].lower())
        idle = window - busy([(k[0], k[1]) for k in ks])
        print(f"  GPU{dev}: window {window / 1e3:6.0f} ms | model compute {compute / 1e3:6.0f} ms | "
              f"NCCL kernels {nccl / 1e3:6.0f} ms | idle {idle / 1e3:6.0f} ms ({idle / window * 100:4.1f} %)")


if len(sys.argv) != 3:
    sys.exit(__doc__)
summarize(sys.argv[1], "P2P off (NCCL_P2P_DISABLE=1)")
summarize(sys.argv[2], "P2P on (NCCL_P2P_DISABLE=0)")
