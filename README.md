# FutureBrainGenerator


## Dependency
For installing the package for all process, you run below code:

    conda create -n FutureBrainGen python=3.9 -y
    conda activate FutureBrainGen

Install `torch=2.3.1` version with `CUDA 11.8` as below:

    pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118

Install other library as bellow:

    pip install -r requirements.txt


## Training

Caution: If you don't have wandb, add `--nowandb` as args in commend line

(1) Train Encoder for LDM