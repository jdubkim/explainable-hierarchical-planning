#!/bin/bash
export PATH=/vol/bitbucket/jk3417/xmbhrl/bin/:$PATH
source activate
source /vol/cuda/11.4.120-cudnn8.2.4/setup.sh
/usr/bin/nvidia-smi

echo "Using config: $1."
echo "Running task: $2."
echo "Start training at $(date +%Y%m%d-%H%M%S)."
python3 /vol/bitbucket/jk3417/explainable-mbhrl/embodied/agents/director/train.py --logdir /vol/bitbucket/jk3417/explainable-mbhrl/logdir/$(date +%Y%m%d-%H%M%S) --configs $1 --task $2

# Param 1: config (e.g.  dmlab, atari, crafter, dmc_vision, dmc_proprio, pinpad, loconav)
# Param 2: task (e.g. dmc_walker_walk)

