#!/usr/bin/bash

#SBATCH -J AdaLDM
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_grad
#SBATCH -w ariel-k1
#SBATCH -t 4-00:00:00
#SBATCH -o /data/msh2044/logs/slurm-%A_AdaLDM.out


pwd
which python
hostname
torchrun --nproc_per_node=4 --nnodes=1 --master_port=25175 script/MedTrainer.py --use_transform --num_workers 8 --train_model AdaLDM --use_AdaIN

exit 0
