#!/usr/bin/env bash
# Development helper for the SM120 port, run on the test machine (aws-raichu). GPU via DG_GPU (default 7); 6 is also free
#   sm120_remote.sh build            rebuild both the DeepGEMM venv (py3.15) and the TokenSpeed-venv build (py3.12)
#   sm120_remote.sh dg  <args...>    run python in the DeepGEMM venv (repo root on PYTHONPATH)
#   sm120_remote.sh ts  <args...>    run python in the TokenSpeed venv with this DeepGEMM build first on PYTHONPATH
#   sm120_remote.sh idle [secs] [max] wait until GPU 7 has been idle for `secs` seconds (it is shared)
set -euo pipefail
DG=/workspace/nctu6/infra/DeepGEMM
TS_PY=/workspace/nctu6/infra/tokenspeed/.venv/bin/python
# GPU is configurable (DG_GPU, default 7); GPU 6 is now also available for builds/tests while 7 runs benchmarks
GPU=${DG_GPU:-7}
export CUDA_HOME=/usr/local/cuda CUDA_VISIBLE_DEVICES=$GPU
cmd=$1; shift
case $cmd in
    build)
        cd $DG
        (source .venv/bin/activate && bash develop.sh > /tmp/nctu6_build315.log 2>&1) || { tail -30 /tmp/nctu6_build315.log; exit 1; }
        # NOTES: `develop.sh` can exit 0 after a failed compile, so check that the extension was actually produced
        ls deep_gemm/_C.cpython-315-*.so > /dev/null 2>&1 && [ deep_gemm/_C.cpython-315-*.so -nt csrc/apis/gemm.hpp ] \
            || { grep -B2 -A8 "error" /tmp/nctu6_build315.log | head -40; echo "build FAILED (py3.15)"; exit 1; }
        rm -rf build312
        $TS_PY setup.py build --build-base build312 > /tmp/nctu6_build312.log 2>&1 || { grep -B2 -A8 "error" /tmp/nctu6_build312.log | head -40; exit 1; }
        ls build312/lib.*/deep_gemm/_C.*.so > /dev/null 2>&1 || { echo "build FAILED (py3.12)"; exit 1; }
        echo "build OK"
        ;;
    dg)
        cd $DG
        DG_JIT_CACHE_DIR=/tmp/dg_cache_nctu6 PYTHONPATH=$DG:$DG/scripts:$DG/tests $DG/.venv/bin/python "$@"
        ;;
    ts)
        # NOTES: run outside the repo so the source `deep_gemm` directory does not shadow the build
        cd /tmp
        PATH=$(dirname $TS_PY):$PATH DG_JIT_CACHE_DIR=/tmp/dg_cache_nctu6_312 \
            PYTHONPATH=$(ls -d $DG/build312/lib.*):$DG/scripts $TS_PY "$@"
        ;;
    idle)
        # Wait until GPU 7 has been idle (0% utilization) for `$1` consecutive seconds (default 20), at most `$2` seconds
        # (default 1800): the GPU is shared with other jobs, and benchmarks taken while they run are not comparable
        need=${1:-20}; budget=${2:-1800}; quiet=0; waited=0
        while [ $quiet -lt $need ]; do
            util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $GPU | tr -d ' ')
            if [ "$util" = "0" ]; then quiet=$((quiet + 1)); else quiet=0; fi
            sleep 1; waited=$((waited + 1))
            if [ $waited -ge $budget ]; then echo "GPU $GPU not idle after ${budget}s" >&2; exit 2; fi
        done
        echo "GPU $GPU idle for ${need}s (waited ${waited}s)"
        ;;
    *)
        echo "unknown command: $cmd" >&2; exit 1
        ;;
esac
