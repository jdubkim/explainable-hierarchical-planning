#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END
#SBATCH --mail-user=jk3417
#SBATCH --output=/vol/bitbucket/jk3417/explainable-mbhrl/slurm_outputs/director_result_%j.out
export PATH=/vol/bitbucket/jk3417/xmbhrl/bin/:$PATH
source activate

#use cuda 11.2
if [[ $(getconf LONG_BIT) == "32" ]]; then
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/vol/cuda/11.2.1-cudnn8.1.0.77/lib
    else
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/vol/cuda/11.2.1-cudnn8.1.0.77/lib64:/vol/cuda/11.2.1-cudnn8.1.0.77/lib
fi

if [ -f /vol/cuda/11.2.1-cudnn8.1.0.77/setup.sh ]
    then
        . /vol/cuda/11.2.1-cudnn8.1.0.77/setup.sh
    else
        echo "ERROR: /vol/cuda/11.2.1-cudnn8.1.0.77/setup.sh not found"
fi

echo "CUDA version $(nvcc --version | grep 'release' | awk '{print $5}') is installed"
dirname $(dirname $(which nvcc))

export XLA_FLAGS=--xla_gpu_cuda_data_dir=/vol/cuda/11.2.1-cudnn8.1.0.77

TERM=vt100 # or TERM=xterm
/usr/bin/nvidia-smi
uptime

echo "Using config: $1."
echo "Running task: $2."
cd /vol/bitbucket/${USER}/explainable-mbhrl
srun python embodied/agents/director/train.py --logdir ./logdir/$(date +%Y%m%d-%H%M%S) \\
    --configs $1 --task $2

# Param 1: config (e.g.  dmlab, atari, crafter, dmc_vision, dmc_proprio, pinpad, loconav)
# Param 2: task (e.g. dmc_walker_walk)
