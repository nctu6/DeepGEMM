#!/usr/bin/env bash
# End-to-end A/B: serve a block-FP8 model with TokenSpeed on GPU 7, with stock kernels ("base") or with this
# DeepGEMM build plus the patched tokenspeed_kernel overlay ("dg"), then run `sm120_e2e_client.py`.
#   sm120_e2e.sh base|dg <model_path> <port> <out_json>
set -euo pipefail
mode=$1 model=$2 port=$3 out=$4
DG=/workspace/nctu6/infra/DeepGEMM
TS=/workspace/nctu6/infra/tokenspeed/.venv/bin
OVERLAY=/workspace/nctu6/infra/tmp/ts_sm120_overlay/pkg
export CUDA_VISIBLE_DEVICES=7 CUDA_HOME=/usr/local/cuda PATH=$TS:$PATH
if [ "$mode" = dg ]; then
    export PYTHONPATH=$(ls -d $DG/build312/lib.*):$OVERLAY DG_JIT_CACHE_DIR=/tmp/dg_cache_nctu6_312
else
    unset PYTHONPATH
fi
log=/tmp/nctu6_e2e_${mode}_${port}.log
# `E2E_KERNEL_VERBOSE=1` logs every tokenspeed_kernel selection (e.g. to confirm the DeepGEMM leaf is used)
[ -n "${E2E_KERNEL_VERBOSE:-}" ] && export TOKENSPEED_KERNEL_VERBOSE=1
cd /tmp
$TS/tokenspeed serve "$model" --served-model-name m --host 127.0.0.1 --port "$port" --tensor-parallel-size 1 \
    --trust-remote-code --max-model-len 16384 --max-num-seqs 64 > "$log" 2>&1 &
server=$!
trap 'kill $server 2>/dev/null; wait $server 2>/dev/null || true' EXIT
for _ in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:$port/v1/models" > /dev/null; then break; fi
    if ! kill -0 $server 2>/dev/null; then echo "server died"; tail -40 "$log"; exit 1; fi
    sleep 5
done
curl -sf "http://127.0.0.1:$port/v1/models" > /dev/null || { echo "server not ready"; tail -40 "$log"; exit 1; }
echo "== $mode"
$TS/python $DG/scripts/sm120_e2e_client.py --port "$port" --model m --out "$out"
