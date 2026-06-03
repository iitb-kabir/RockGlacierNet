#!/bin/bash
# Launch helper: sets CUDA library paths for the brats TF 2.13 env and pins GPU.
ENV=/data1/nasiruddink/miniconda3/envs/brats
SP=$ENV/lib/python3.10/site-packages
NVLIBS=$(find $SP/nvidia -name "*.so*" -printf "%h\n" 2>/dev/null | sort -u | tr '\n' ':')
export LD_LIBRARY_PATH="$ENV/lib:$NVLIBS$LD_LIBRARY_PATH"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
exec conda run -n brats --no-capture-output python "$@"
