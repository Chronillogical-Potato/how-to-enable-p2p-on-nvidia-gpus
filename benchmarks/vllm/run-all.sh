#!/bin/bash
# The full run from README.md: both models, P2P off and on, two rounds in alternating order, then the offline
# prompt test and NCCL's all-reduce on its own. About 45 minutes on the test rig.
#
# usage: ./run-all.sh        (results in results/local, or RESULTS=...)
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"
export RESULTS=${RESULTS:-$HERE/results/local}
VENV=${VENV:-$HOME/vllm-venv}
mkdir -p "$RESULTS"
date

for round in run1 run2; do
  if [ $round = run1 ]; then order="base nccl"; else order="nccl base"; fi
  for cfg in $order; do ./bench.sh "$cfg" $round; done
  for cfg in $order; do
    MODEL=Qwen/Qwen3-1.7B DTYPE=half MAXLEN=2048 KV_BYTES=600000000 SUITE=decode ./bench.sh "$cfg" $round
  done
done

for cfg in base nccl; do
  if [ $cfg = base ]; then p2p_disable=1; else p2p_disable=0; fi
  NCCL_P2P_DISABLE=$p2p_disable NO_CAR=1 "$VENV/bin/python" prefill.py > "$RESULTS/prefill-$cfg.log" 2>&1
  echo "prefill.py $cfg:"
  grep -E '^PREFILL' "$RESULTS/prefill-$cfg.log"
  NCCL_P2P_DISABLE=$p2p_disable "$VENV/bin/python" allreduce.py > "$RESULTS/allreduce-$cfg.log" 2>&1
  echo "allreduce.py $cfg:"
  grep -E '^ALLREDUCE' "$RESULTS/allreduce-$cfg.log"
done

date
echo ALLDONE
