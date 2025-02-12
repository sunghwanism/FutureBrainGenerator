#!/usr/bin/bash

#SBATCH -J LDM
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_grad
#SBATCH -w ariel-k2
#SBATCH -t 4-00:00:00
#SBATCH -o /data/alice6114/logs/slurm-%A_LDM.out


pwd
which python
hostname
torchrun --nproc_per_node=4 --nnodes=1 --master_port=25155 script/LDMTrainer_ddp.py --use_transform --num_workers 8

exit 0
