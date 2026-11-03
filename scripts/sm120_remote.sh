#!/usr/bin/env bash
# Development helper for the SM120 port, run on the test machine (aws-raichu, GPU 7 only)
#   sm120_remote.sh build            rebuild both the DeepGEMM venv (py3.15) and the TokenSpeed-venv build (py3.12)
#   sm120_remote.sh dg  <args...>    run python in the DeepGEMM venv (repo root on PYTHONPATH)
#   sm120_remote.sh ts  <args...>    run python in the TokenSpeed venv with this DeepGEMM build first on PYTHONPATH
set -euo pipefail
DG=/workspace/nctu6/infra/DeepGEMM
TS_PY=/workspace/nctu6/infra/tokenspeed/.venv/bin/python
export CUDA_HOME=/usr/local/cuda CUDA_VISIBLE_DEVICES=7
cmd=$1; shift
case $cmd in
    build)
        cd $DG
        (source .venv/bin/activate && bash develop.sh > /tmp/nctu6_build315.log 2>&1) || { tail -30 /tmp/nctu6_build315.log; exit 1; }
        rm -rf build312
        $TS_PY setup.py build --build-base build312 > /tmp/nctu6_build312.log 2>&1 || { tail -30 /tmp/nctu6_build312.log; exit 1; }
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
    *)
        echo "unknown command: $cmd" >&2; exit 1
        ;;
esac
