#!/bin/bash
source /vol/cuda/11.3.1-cudnn8.2.1/setup.sh
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/vol/cuda/11.3.1-cudnn8.2.1
/usr/bin/nvidia-smi

echo $(which python3)

echo "Using config: $1."
echo "Running task: $2."
echo "Start training at $(date +%Y%m%d-%H%M%S)."

python /vol/bitbucket/jk3417/explainable-mbhrl/embodied/agents/director/train.py --logdir /vol/bitbucket/jk3417/explainable-mbhrl/logdir/$(date +%Y%m%d-%H%M%S) --configs $1 --task $2 $3

# Param 1: config (e.g.  dmlab, atari, crafter, dmc_vision, dmc_proprio, pinpad, loconav)
# Param 2: task (e.g. dmc_walker_walk)

