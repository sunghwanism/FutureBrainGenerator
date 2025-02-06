import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../MONAI')))

import json
# from model.classifier import HalfUNetClassifier
# from MONAI.generative.networks.nets import DiffusionModelClfUNet, DiffusionModelUNet

import torch
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
import gc

import wandb

from data.LongDataset import LongitudinalDataset
from MONAI.generative.networks.nets import VQVAE
from monai.utils import first
from MONAI.generative.inferers import LatentCfDiffusionInferer, LatentDiffusionInferer
from MONAI.generative.networks.schedulers import DDPMScheduler, DDIMScheduler


def init_wandb(config):
    if not config.nowandb:
        assert wandb is not None, "Wandb not installed, please install it or run without wandb"
        wandb.init(project=config.wandb_project, entity=config.wandb_entity, config=config)
        config.wandb_url = wandb.run.get_url()
        
        
def load_confg(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def load_VQVAE(config, device, modelpath, wrap_ddp=False):

    encoder = torch.load(modelpath, map_location=device)['encoder']
    encoder_config = torch.load(modelpath, map_location=device)['config']
    
    EDmodel = VQVAE(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        num_channels=encoder_config.num_channels,
        num_res_channels=encoder_config.num_res_channels,
        num_res_layers=encoder_config.num_res_blocks,
        commitment_cost = encoder_config.commitment_cost,
        downsample_parameters=encoder_config.downsample_param, 
        upsample_parameters=encoder_config.upsample_param,
        num_embeddings=encoder_config.num_embeddings,
        embedding_dim=encoder_config.latent_channels,
        ddp_sync=True,
    )
    
    EDmodel.to(device)

    EDmodel = torch.nn.SyncBatchNorm.convert_sync_batchnorm(EDmodel)
    EDmodel.load_state_dict(encoder, strict=False)
    
    if wrap_ddp: # To-do
        if local_rank is None:
            raise ValueError("Need rank for fine-tuning ")
        EDmodel = torch.nn.parallel.DistributedDataParallel(
            EDmodel, device_ids=[local_rank], output_device=local_rank
        )

    else:
        # Freeze all the parameters of the VQ-VAE model
        for param in EDmodel.parameters():
            param.requires_grad = False
            
    del encoder, encoder_config
    gc.collect()
    torch.cuda.empty_cache()

    return EDmodel


def longitudinal_load_dataloader(config, world_size, rank):

    TrainDataset = LongitudinalDataset(config, _type='train', Transform=train_transform)    
    ValDataset = LongitudinalDataset(config, _type='val', Transform=train_transform)
    
    train_sampler = DistributedSampler(TrainDataset, num_replicas=world_size, rank=rank)

    train_loader = DataLoader(
        TrainDataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
    )

    first_batch = first(train_loader)
    
    val_loader = DataLoader(
        ValDataset,
        batch_size=config.batch_size//2,
        num_workers=config.num_workers,
        pin_memory=True,
        shuffle=False,
        drop_last=True,
    )
    
    return train_loader, val_loader, train_sampler, first_batch
    

def merge_loss_all_rank(loss_list, device, world_size, batch_len):
    total_loss_tensor = torch.tensor(loss_list, device=device)
    dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
    total_loss_tensor /= world_size

    epoch_loss = total_loss_tensor[0].item() / batch_len
    clf_loss = total_loss_tensor[1].item() / batch_len
    total_losses = total_loss_tensor[2].item() / batch_len

    return epoch_loss, clf_loss, total_losses


def generate_unet(config, device, cond_size, local_rank):

    unet = DiffusionModelUNet(
        spatial_dims=3,
        in_channels=config.latent_channels,
        out_channels=config.latent_channels,
        num_res_blocks=config.diff_num_res_blocks,
        num_channels=config.diff_num_channels,
        attention_levels=config.diff_attention_levels,
        cross_attention_dim=cond_size,
        with_conditioning=True,
        num_head_channels=config.diff_num_head_channels,
    ).to(device)

    unet = torch.nn.parallel.DistributedDataParallel(unet, device_ids=[local_rank], 
                                                     output_device=local_rank,) 
    
    
    return unet

def generate_Inferer(scheduler, scale_factor, config):
    if config.use_clf or config.use_baseimg:
        inferer = LatentCfDiffusionInferer(scheduler, scale_factor=scale_factor,
                                           use_baseimg=config.use_baseimg)
    else:
        inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor,)

    return inferer


# def generate_scheduler(config):
#     if config.scheduler == 'ddpm':
#         if config.schedule_type == 'cosine':
#             scheduler = DDPMScheduler(num_train_timesteps=config.timestep, 
#                                         schedule=config.schedule_type)
#         else:
#             scheduler = DDPMScheduler(num_train_timesteps=config.timestep, 
#                                 beta_start=config.beta_start, beta_end=config.beta_end,
#                                 schedule=config.schedule_type) # linear_beta scaled_linear_beta
#     elif config.scheduler == 'ddim':
#         if config.schedule_type == 'cosine':
#             scheduler = DDIMScheduler(num_train_timesteps=config.timestep,
#                                 schedule =config.schedule_type,
#                                 steps_offset=0,
#                                 clip_sample=True,
#                                 set_alpha_to_one=False,
#                                 prediction_type="epsilon")
#         else:
#             scheduler = DDIMScheduler(num_train_timesteps=config.timestep,
#                                     schedule =config.schedule_type,
#                                     steps_offset=0,
#                                     beta_start=config.beta_start,
#                                     beta_end=config.beta_end,
#                                     clip_sample=True,
#                                     set_alpha_to_one=True,
#                                     prediction_type="epsilon") # or "sample"
            
#     return scheduler