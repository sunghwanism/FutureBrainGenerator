# FutureBrainGenerator


## Dependency
For installing the package for all process, you run below code:

    conda create -n FutureBrainGen python=3.10 -y
    conda activate FutureBrainGen

Install other library as bellow:

    pip install -r requirements.txt


Install `torch=2.2.1` version with `CUDA 11.8` as below:

    pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu118


## Training (Only multi-GPU)

Caution: If you don't have wandb, add `--nowandb` as args in commend line

(1) Train VQGAN

    torchrun --nproc_per_node=4 --nnodes=1 script/VQTrainer_ddp.py --batch_size 4 --use_transform

(2) Train Encoder for LDM

    torchrun --nproc_per_node=2 --nnodes=1 script/LDMTrainer_ddp.py --batch_size 2 --use_transform