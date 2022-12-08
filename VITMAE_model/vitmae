#!/bin/bash
#SBATCH -A als
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 10:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --output=vitmae_xrd.out
#SBATCH --error=vitmae_xrd.err
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=hasitha@berkeley.edu


module load python
conda activate dev

export SLURM_CPU_BIND="cores"
export HDF5_USE_FILE_LOCKING=FALSE

export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500



srun -l -u python /global/homes/h/hasitha/latent_xrd/vitmae.py -d nccl --rank-gpu --ranks-per-node=${SLURM_NTASKS_PER_NODE} $@