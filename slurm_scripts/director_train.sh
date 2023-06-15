#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END
#SBATCH --mail-user=jk3417
#SBATCH --output=/vol/bitbucket/jk3417/explainable-mbhrl/slurm_outputs/director_result_%j.out
#SBATCH --partition=gpgpuB,gpgpuC
export PATH=/vol/bitbucket/jk3417/xmbhrl/bin/:$PATH
source activate

. /vol/cuda/11.3.1-cudnn8.2.1/setup.sh
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/vol/cuda/11.3.1-cudnn8.2.1
TERM=vt100 # or TERM=xterm
/usr/bin/nvidia-smi
uptime

logdir=$(date +%Y%m%d-%H%M%S)
mkdir /vol/bitbucket/jk3417/explainable-mbhrl/logdir/$logdir/
cp /vol/bitbucket/jk3417/explainable-mbhrl/embodied/agents/director/configs.yaml /vol/bitbucket/jk3417/explainable-mbhrl/logdir/$logdir/

echo "Using config: $1."
echo "Running task: $2."
echo "Using logdir: $logdir."
srun python /vol/bitbucket/jk3417/explainable-mbhrl/embodied/agents/director/train.py --logdir /vol/bitbucket/jk3417/explainable-mbhrl/logdir/$logdir --configs $1 --task $2 $3

# Param 1: config (e.g.  dmlab, atari, crafter, dmc_vision, dmc_proprio, pinpad, loconav)
# Param 2: task (e.g. dmc_walker_walk)
# Param 3: Rest of commands (e.g. additional configs: --goal_regularised True)
