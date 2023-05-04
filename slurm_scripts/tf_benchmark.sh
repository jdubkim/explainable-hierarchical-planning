#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END
#SBATCH --mail-user=jk3417
#SBATCH --output=/vol/bitbucket/jk3417/explainable-mbhrl/slurm_outputs/tf_benchmark_%j.out
#SBATCH --partition gpgpu

export PATH=/vol/bitbucket/jk3417/xmbhrl/bin/:$PATH
source activate

. /vol/cuda/11.3.1-cudnn8.2.1/setup.sh
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/vol/cuda/11.3.1-cudnn8.2.1
TERM=vt100 # or TERM=xterm
/usr/bin/nvidia-smi
ptime

srun python /vol/bitbucket/jk3417/explainable-mbhrl/debug/tf_benchmark.py

