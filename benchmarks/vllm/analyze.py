"""Summarize `vllm bench serve` results: P2P off (base) vs NCCL P2P (nccl), mean over the rounds.

usage: python3 analyze.py <results-dir>

<results-dir> holds <model>-<dtype>/<round>/<base|nccl>/<test>.json, as bench.sh writes them. Each cell is
"P2P off -> P2P on, gain". Gain is positive when P2P is faster: higher throughput, or lower TTFT / TPOT.
"""
import json
import pathlib
import sys
from statistics import mean

TESTS = {
    "p1k": "1 user, 1k prompt",
    "p4k": "1 user, 4k prompt",
    "p8k": "1 user, 8k prompt",
    "d1": "1 user, generating",
    "m4": "4 users, 2k prompts",
    "m8": "8 users, 1k prompts",
    "d16": "16 users, short prompts",
    "d32": "32 users, short prompts",
}
FIELDS = [  # key, column label, higher is better
    ("mean_ttft_ms", "TTFT ms", False),
    ("mean_tpot_ms", "TPOT ms", False),
    ("output_throughput", "output tok/s", True),
    ("total_token_throughput", "total tok/s", True),
]

if len(sys.argv) != 2:
    sys.exit(__doc__)
root = pathlib.Path(sys.argv[1])

for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
    rows = {}
    for f in model_dir.glob("*/*/*.json"):
        rnd, cfg, test = f.parts[-3], f.parts[-2], f.stem
        if test in TESTS:
            rows.setdefault(test, {}).setdefault(cfg, {})[rnd] = json.loads(f.read_text())
    if not rows:
        continue
    print(f"\n## {model_dir.name}")
    print(f"{'test':24s}" + "".join(f"{label:>28s}" for _, label, _ in FIELDS))
    for test, title in TESTS.items():
        if test not in rows or not {"base", "nccl"} <= rows[test].keys():
            continue
        cells = []
        for key, _, higher_better in FIELDS:
            b = mean(r[key] for r in rows[test]["base"].values())
            n = mean(r[key] for r in rows[test]["nccl"].values())
            gain = (n / b - 1) * 100 if higher_better else (b / n - 1) * 100
            cells.append(f"{b:8.1f} -> {n:8.1f} {gain:+5.1f}%")
        print(f"{title:24s}" + "".join(f"{c:>28s}" for c in cells))
    spreads = []  # same configuration, round 1 vs round 2: a noise estimate
    for cfgs in rows.values():
        for rounds in cfgs.values():
            if len(rounds) == 2:
                a, b = (r["output_throughput"] for r in rounds.values())
                spreads.append(abs(a / b - 1) * 100)
    if spreads:
        spreads.sort()
        print(f"round-to-round difference in output tok/s, same configuration: "
              f"median {spreads[len(spreads) // 2]:.1f} %, max {spreads[-1]:.1f} %")
