#!/bin/bash
# One configuration of the vLLM benchmark: start a tensor-parallel-2 server, send the request suite, stop it.
#
# usage: [MODEL=..] [DTYPE=..] [SUITE=full|decode] ./bench.sh <base|nccl> <round>
#
#   base  P2P off: NCCL_P2P_DISABLE=1 and --disable-custom-all-reduce. All-reduce goes through host RAM
#         (NCCL "via SHM").
#   nccl  P2P on for NCCL ("via P2P/..."). vLLM's custom all-reduce stays off.
#   car   NCCL P2P plus vLLM's custom all-reduce. Refused unless ALLOW_CAR=1: on the test rig's Turing pair (two
#         different chips, one behind the chipset; either may be the reason) it reset the machine (README.md,
#         "vLLM's custom all-reduce reset the machine").
#
# Environment, defaults sized for an RTX 2060 6 GB + GTX 1650 SUPER 4 GB pair (README.md, "Memory settings"):
#   MODEL=Qwen/Qwen3-0.6B  DTYPE=float32  MAXLEN=8448  KV_BYTES=1100000000  SUITE=full  PORT=8000
#   VENV=~/vllm-venv  RESULTS=<this folder>/results/local
# Two different Turing chips: export LD_PRELOAD and P2P_CHIP_IMPL_MAP for the CUDA shim first (README.md).
set -u
CFG=${1:?usage: bench.sh <base|nccl> <round>}
ROUND=${2:?usage: bench.sh <base|nccl> <round>}
HERE=$(cd "$(dirname "$0")" && pwd)
MODEL=${MODEL:-Qwen/Qwen3-0.6B}
DTYPE=${DTYPE:-float32}
MAXLEN=${MAXLEN:-8448}
KV_BYTES=${KV_BYTES:-1100000000}
SUITE=${SUITE:-full}
PORT=${PORT:-8000}
VENV=${VENV:-$HOME/vllm-venv}
RESULTS=${RESULTS:-$HERE/results/local}
OUT=$RESULTS/$(basename "$MODEL")-$DTYPE/$ROUND/$CFG
mkdir -p "$OUT"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# NCCL's log goes to one file per process: its stdout is buffered and lost when vLLM stops the workers.
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,P2P,SHM
export NCCL_DEBUG_FILE=$OUT/nccl.%p.log

EXTRA=()
case $CFG in
  base) export NCCL_P2P_DISABLE=1; EXTRA=(--disable-custom-all-reduce) ;;
  nccl) export NCCL_P2P_DISABLE=0; EXTRA=(--disable-custom-all-reduce) ;;
  car)
    if [ "${ALLOW_CAR:-}" != 1 ]; then
      echo "car refused: vLLM's custom all-reduce reset the test rig's mixed Turing pair (README.md). ALLOW_CAR=1 overrides."
      exit 2
    fi
    export NCCL_P2P_DISABLE=0 ;;
  *) echo "unknown configuration '$CFG' (base, nccl or car)"; exit 2 ;;
esac

vllm serve "$MODEL" --dtype "$DTYPE" --tensor-parallel-size 2 --max-model-len "$MAXLEN" \
  --kv-cache-memory-bytes "$KV_BYTES" --max-num-seqs 32 --no-enable-prefix-caching \
  --compilation-config '{"max_cudagraph_capture_size": 64}' \
  --port "$PORT" "${EXTRA[@]}" > "$OUT/server.log" 2>&1 &
PID=$!

for i in $(seq 900); do
  curl -sf "localhost:$PORT/health" > /dev/null && break
  if ! kill -0 $PID 2> /dev/null; then echo "server died, see $OUT/server.log"; exit 1; fi
  sleep 1
done
echo "$(date +%T) $(basename "$MODEL")-$DTYPE $CFG $ROUND: server up after ${i}s"

run() {
  local name=$1
  shift
  vllm bench serve --model "$MODEL" --port "$PORT" --dataset-name random --ignore-eos --seed 0 \
    --save-result --result-dir "$OUT" --result-filename "$name.json" "$@" > "$OUT/$name.txt" 2>&1
  echo "  $name: $(grep -E 'Output token throughput|Mean TTFT|Mean TPOT' "$OUT/$name.txt" | tr -s ' ' | tr '\n' ';')"
}

# Random token prompts with fixed lengths and --ignore-eos, so every configuration gets the same work.
run warmup --random-input-len 1024 --random-output-len 32  --num-prompts 4  --max-concurrency 1
if [ "$SUITE" = full ]; then
  run p1k  --random-input-len 1024 --random-output-len 64  --num-prompts 8  --max-concurrency 1
  run p4k  --random-input-len 4096 --random-output-len 64  --num-prompts 6  --max-concurrency 1
  run p8k  --random-input-len 8192 --random-output-len 64  --num-prompts 4  --max-concurrency 1
  run d1   --random-input-len 256  --random-output-len 512 --num-prompts 4  --max-concurrency 1
  run m4   --random-input-len 2048 --random-output-len 256 --num-prompts 16 --max-concurrency 4
  run m8   --random-input-len 1024 --random-output-len 256 --num-prompts 24 --max-concurrency 8
  run d16  --random-input-len 256  --random-output-len 256 --num-prompts 64 --max-concurrency 16
  run d32  --random-input-len 256  --random-output-len 256 --num-prompts 96 --max-concurrency 32
else
  run d1   --random-input-len 256  --random-output-len 512 --num-prompts 6  --max-concurrency 1
fi

kill $PID
wait $PID 2> /dev/null

# The transport NCCL picked, without the host:pid prefix of each log line.
cat "$OUT"/nccl.*.log 2> /dev/null | grep -E 'via (P2P|SHM)' | sed 's/^[^ ]* //' | sort | uniq -c > "$OUT/nccl-transport.txt"
transport=$(grep -oE 'via [A-Za-z0-9/]+' "$OUT/nccl-transport.txt" | sort -u | tr '\n' ' ')
echo "  transport: $transport"
if [ "$CFG" != base ] && ! grep -q 'via P2P' "$OUT/nccl-transport.txt"; then
  echo "  WARNING: P2P was allowed but NCCL did not use it. Check nvidia-smi topo -p2p r, and the CUDA shim for two different Turing chips."
fi
