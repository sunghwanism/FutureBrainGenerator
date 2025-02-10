#!/usr/bin/bash

#SBATCH -J VQ-GAN
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=5
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_grad
#SBATCH -w ariel-v2
#SBATCH -t 3-00:00:00
#SBATCH -o /data/msh2044/logs/slurm-%A_VQ-GAN.out


pwd
which python
hostname
torchrun --nproc_per_node=4 --nnodes=1 script/VQTrainer_ddp.py --use_transform --num_workers 5 --latent_channels 16 --batch_size 4

exit 0
