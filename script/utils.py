import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../MONAI')))

import json
from model.classifier import HalfUNetClassifier
from MONAI.generative.networks.nets import DiffusionModelClfUNet, DiffusionModelUNet

import torch
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist

import wandb

from dataloader.diffusionDataset import LdMCfDataset
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



def load_VQVAE(config, device, modelpath):
    EDmodel = VQVAE(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        num_channels=config.num_channels,
        num_res_channels=config.num_channels,
        num_res_layers=config.num_res_blocks,
        commitment_cost = config.commitment_cost,
        downsample_parameters=config.downsample_param, 
        upsample_parameters=config.upsample_param,
        num_embeddings=config.num_embeddings,
        embedding_dim=config.latent_channels,
        ddp_sync=True,
    )
    EDmodel.to(device)

    EDmodel = torch.nn.SyncBatchNorm.convert_sync_batchnorm(EDmodel)
    encoder = torch.load(modelpath, map_location=device)['encoder']
    EDmodel.load_state_dict(encoder, strict=False)

    # Freeze all the parameters of the VQ-VAE model
    for param in EDmodel.parameters():
        param.requires_grad = False

    return EDmodel

def generateClassifier(config, device, cond_size, local_rank):
    classifier = HalfUNetClassifier(spatial_dims=3,
                                in_channels=config.latent_channels,
                                out_channels=config.num_classes,
                                num_res_blocks=config.clf_num_res_blocks,
                                num_channels=config.clf_num_channels,
                                with_conditioning=True,
                                cross_attention_dim = cond_size - 1, # condition except interval
                                attention_levels=config.clf_attention_levels,
                                num_head_channels=config.clf_num_head_channels,).to(device)
    
    classifier = torch.nn.SyncBatchNorm.convert_sync_batchnorm(classifier)
    classifier = torch.nn.parallel.DistributedDataParallel(classifier, device_ids=[local_rank], output_device=local_rank,
                                                           find_unused_parameters=True if config.use_clf else False)

    return classifier


def load_dataloader(config, world_size, rank, use_val=False):

    TrainDataset = LdMCfDataset(config.data_path, config, _type='static') # To-do
    train_sampler = DistributedSampler(TrainDataset, num_replicas=world_size, rank=rank)
    
    ValDataset = LdMCfDataset(config.data_path, config, _type='converted') # To-do

    train_loader = DataLoader(
        TrainDataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
    )

    first_batch = first(train_loader)
    
    if use_val:
        val_loader = DataLoader(
            ValDataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=True,
            shuffle=False,
            drop_last=True,
        )
        return train_loader, val_loader, train_sampler, first_batch
    
    else:
        return train_loader, train_sampler, first_batch
    

def merge_loss_all_rank(loss_list, device, world_size, batch_len):
    total_loss_tensor = torch.tensor(loss_list, device=device)
    dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
    total_loss_tensor /= world_size

    epoch_loss = total_loss_tensor[0].item() / batch_len
    clf_loss = total_loss_tensor[1].item() / batch_len
    total_losses = total_loss_tensor[2].item() / batch_len

    return epoch_loss, clf_loss, total_losses


def generate_unet(config, device, cond_size, local_rank, use_baseimg=False, use_clf=False):
    if use_clf:
        classifier = generateClassifier(config, device, cond_size, local_rank)
    else:
        classifier = None
        

    if use_baseimg:
       unet = DiffusionModelClfUNet(spatial_dims=3,
                                    in_channels=config.latent_channels,
                                    out_channels=config.latent_channels,
                                    classifier=classifier,
                                    # classifier_scale=config.guidance_scale,
                                    num_res_blocks=config.diff_num_res_blocks,
                                    num_channels=config.diff_num_channels,
                                    attention_levels=config.diff_attention_levels,
                                    cross_attention_dim=cond_size+config.num_classes if use_clf else cond_size,
                                    with_conditioning=True,
                                    num_head_channels=config.diff_num_head_channels).to(device)
       
    elif not use_clf and not use_baseimg:
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
                                                     output_device=local_rank, 
                                                     find_unused_parameters=True if config.use_clf else False) 
    
    return unet, classifier

def generate_Inferer(scheduler, scale_factor, config):
    if config.use_clf or config.use_baseimg:
        inferer = LatentCfDiffusionInferer(scheduler, scale_factor=scale_factor,
                                           use_baseimg=config.use_baseimg)
    else:
        inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor,)

    return inferer


def generate_scheduler(config):
    if config.scheduler == 'ddpm':
        if config.schedule_type == 'cosine':
            scheduler = DDPMScheduler(num_train_timesteps=config.timestep, 
                                        schedule=config.schedule_type)
        else:
            scheduler = DDPMScheduler(num_train_timesteps=config.timestep, 
                                beta_start=config.beta_start, beta_end=config.beta_end,
                                schedule=config.schedule_type) # linear_beta scaled_linear_beta
    elif config.scheduler == 'ddim':
        if config.schedule_type == 'cosine':
            scheduler = DDIMScheduler(num_train_timesteps=config.timestep,
                                schedule =config.schedule_type,
                                steps_offset=0,
                                clip_sample=True,
                                set_alpha_to_one=False,
                                prediction_type="epsilon")
        else:
            scheduler = DDIMScheduler(num_train_timesteps=config.timestep,
                                    schedule =config.schedule_type,
                                    steps_offset=0,
                                    beta_start=config.beta_start,
                                    beta_end=config.beta_end,
                                    clip_sample=True,
                                    set_alpha_to_one=True,
                                    prediction_type="epsilon") # or "sample"
            
    return scheduler