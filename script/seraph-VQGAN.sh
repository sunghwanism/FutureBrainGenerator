#!/usr/bin/bash

#SBATCH -J VQGAN
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_grad
#SBATCH -w ariel-v2
#SBATCH -t 3-00:00:00
#SBATCH -o /data/msh2044/logs/slurm-%A_VQGAN.out


pwd
which python
hostname
torchrun --nproc_per_node=4 --nnodes=1 --master_port=25134 script/VQTrainer_ddp.py --use_transform --num_workers 7 --batch_size 3

exit 0
