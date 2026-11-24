#!/bin/bash
# Profile prompt processing with P2P off and on, then split each GPU's time into model compute, NCCL and idle.
# vLLM's workers record the timed runs with the PyTorch profiler; profile_kernels.py reads the traces.
#
# usage: [SHAPES=8x1024] ./profile.sh        (results in results/local, or RESULTS=...)
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"
RESULTS=${RESULTS:-$HERE/results/local}
VENV=${VENV:-$HOME/vllm-venv}
SHAPES=${SHAPES:-8x1024}
mkdir -p "$RESULTS"

for setting in off on; do
  if [ $setting = off ]; then p2p_disable=1; else p2p_disable=0; fi
  dir=$RESULTS/profile-p2p$setting
  rm -rf "$dir"
  NCCL_P2P_DISABLE=$p2p_disable NO_CAR=1 SHAPES=$SHAPES TORCH_PROFILE_DIR=$dir \
    "$VENV/bin/python" prefill.py > "$dir.log" 2>&1
  echo "P2P $setting: $(grep -E '^PREFILL' "$dir.log")"
done

python3 profile_kernels.py "$RESULTS/profile-p2poff" "$RESULTS/profile-p2pon"
