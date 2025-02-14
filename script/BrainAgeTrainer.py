import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../MONAI')))

import numpy as np
import json

import torch
import torch.nn as nn

import torchio as tio
from monai.utils import set_determinism, optional_import

import wandb

from script.utils import *
from script.configure.VQGANconfig import get_run_parser

from model.BrainPredModel import BrainAgePrediction
from data.CrossDataset import CrossMRIDataset

import warnings
warnings.filterwarnings("ignore")

tqdm, has_tqdm = optional_import("tqdm")


def main(config):

    assert config.device_id is not None, 'Please specify the device id'
    assert config.train_model == 'BrainAge', 'Please specify the model to train brain age prediction'

    # Set the random seed
    torch.cuda_set_deivce(config.device_id)
    device = torch.device('cuda', config.device_id)

    set_determinism(config.seed)
    torch.backends.cudnn.benchmark = True

    # Load VQ-VAE model
    VQVAEPATH = os.path.join(config.base_path, config.enc_model)

    EDmodel = load_VQVAE(VQVAEPATH, wrap_ddp=False, local_rank=config.device_id)
    EDmodel.eval()

    if not config.nowandb:
        init_wandb(config, )
        wandb_save_path = os.path.join(config.save_path, f'{wandb.run.name}')
        wandb_img_path = os.path.join(config.save_img_path, f'{wandb.run.name}')

        if not os.path.exists(wandb_save_path):
            os.makedirs(wandb_save_path)
        if not os.path.exists(wandb_img_path):
            os.makedirs(wandb_img_path)

        print("******"*20)
        print(config)
        print(f"Using device {device} on rank {config.device_id}")
        print("******"*20)

    #######################################################################################
    if config.use_transform:
        train_transform = tio.Compose([
            tio.RandomAffine(scales=(0.90, 1.1), p=0.5),
            tio.RandomAffine(degrees=(0, 10), p=0.5),
            tio.RandomAffine(translation=(5, 5, 5), p=0.8),
            tio.RescaleIntensity(out_min_max=(0, 1)),
            ])
        val_transform = tio.Compose([
            tio.RescaleIntensity(out_min_max=(0, 1)),])
    else:
        train_transform = None
        val_transform = None
    #######################################################################################
    TrainDataset = CrossMRIDataset(config, _type='train', Transform=train_transform, 
                                   train_model=config.train_model)
    ValDataset = CrossMRIDataset(config, _type='val', Transform=val_transform, 
                                 train_model=config.train_model)

    train_loader = torch.utils.data.DataLoader(TrainDataset, 
                                               batch_size=config.batch_size, 
                                               shuffle=True, num_workers=config.num_workers)
    
    val_loader = torch.utils.data.DataLoader(ValDataset,
                                             batch_size=config.batch_size, 
                                             shuffle=False, num_workers=config.num_workers)
    #######################################################################################

    model = BrainAgePrediction(config, input_dim=config.latent_channels, lateten_shape=(24, 28, 24)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9555, last_epoch=-1)
    criterion = nn.MSELoss()

    ###############################################################
    # Train the model

    for epoch in range(config.epochs):
        epoch_train_loss = 0
        epoch_val_loss = 0

        model.train()

        for step, batch in enumerate(train_loader):
            
            optimizer.zero_grad()

            images, age = batch[0].to(device), batch[1].to(device)

            images = EDmodel.encode_stage_2_inputs(images)
            pred, _ = model(images)

            loss = criterion(pred, age)

            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        if not config.nowandb:
            wandb.log({'Train Loss': epoch_train_loss})

        if (epoch+1) % config.save_interval == 0:
            model.eval()

            for step, batch in enumerate(val_loader):
                images, age = batch[0].to(device), batch[1].to(device)
                images = EDmodel.encode_stage_2_inputs(images)
                pred, _ = model(images)

                loss = criterion(pred, age)
                epoch_val_loss += loss.item()

            if not config.nowandb:
                wandb.log({'Val Loss': epoch_val_loss})

            torch.save(model.state_dict(),
                       os.path.join(wandb_save_path, f'BrainAgePrediction_{epoch}_dim{config.latent_channels}.pth'))
            
            print(f"Epoch {epoch+1} Train Loss: {epoch_train_loss} Val Loss: {epoch_val_loss}")

        scheduler.step()


if __name__ == '__main__':
    parser = get_run_parser()
    config = parser.parse_args()
    main(config)