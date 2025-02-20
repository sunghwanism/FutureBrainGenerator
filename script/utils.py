import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../MONAI')))

import json
import torch
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
import gc

import wandb

from data.LongDataset import LongitudinalDataset
from MONAI.generative.networks.nets import VQVAE
from monai.utils import first
from MONAI.generative.networks.schedulers import DDPMScheduler, DDIMScheduler

from model.MedUNet import LongLDMmodel, LongBrainmodel
from model.inferer import LongLDMInferer


def init_wandb(config, key=None):
    if not config.nowandb:
        assert wandb is not None, "Wandb not installed, please install it or run without wandb"
        if key is not None:
            wandb.login(key=key)
        wandb.init(project=config.wandb_project, entity=config.wandb_entity, config=config)
        config.wandb_url = wandb.run.get_url()
    
        
def load_confg(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def load_VQVAE(modelpath, wrap_ddp=False, local_rank=None):


    if local_rank is not None:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assert os.path.exists(modelpath), f"Model path {modelpath} does not exist"

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
        assert local_rank is not None, "Need rank for fine-tuning"
        if local_rank is None:
            raise ValueError("Need rank for fine-tuning ")
        EDmodel = torch.nn.parallel.DistributedDataParallel(
            EDmodel, device_ids=[local_rank], output_device=local_rank
        )

    # Freeze all the parameters of the VQ-VAE model
    for param in EDmodel.parameters():
        param.requires_grad = False
            
    del encoder, encoder_config

    gc.collect()
    torch.cuda.empty_cache()

    return EDmodel


def longitudinal_load_dataloader(config, world_size, rank, train_transform, val_transform):

    TrainDataset = LongitudinalDataset(config, _type='train', Transform=train_transform)    
    ValDataset = LongitudinalDataset(config, _type='val', Transform=val_transform)
    
    train_sampler = DistributedSampler(TrainDataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(ValDataset, num_replicas=world_size, rank=rank)

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
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
        sampler=val_sampler,
        drop_last=True,
    )
    
    return train_loader, val_loader, train_sampler, first_batch
    

def merge_loss_all_rank(loss_list, device, world_size, batch_len):
    total_loss_tensor = torch.tensor(loss_list, device=device)
    dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
    total_loss_tensor /= world_size

    epoch_loss = total_loss_tensor[0].item() / batch_len

    return epoch_loss


def generate_unet(config, device, cond_size, latent_dim, local_rank=None):

    if config.train_model == 'LDM':
        unet = LongLDMmodel(
            spatial_dims=3,
            in_channels=latent_dim,
            out_channels=latent_dim,
            num_res_blocks=config.diff_num_res_blocks,
            num_channels=config.diff_num_channels,
            attention_levels=config.diff_attention_levels,
            cross_attention_dim=cond_size,
            with_conditioning=True,
            clinical_condition=config.condition,
            num_head_channels=config.diff_num_head_channels,
            transformer_num_layers=config.transformer_num_layer,
        ).to(device)

    elif config.train_model == 'AdaLDM':
        assert config.use_AdaIN, "AdaIN must be used for AdaLDM" and config.train_model == 'AdaLDM'

        unet = LongBrainmodel(
            spatial_dims=3,
            in_channels=latent_dim,
            out_channels=latent_dim,
            num_res_blocks=config.diff_num_res_blocks,
            num_channels=config.diff_num_channels,
            attention_levels=config.diff_attention_levels,
            cross_attention_dim=cond_size,
            with_conditioning=True,
            clinical_condition=config.condition,
            num_head_channels=config.diff_num_head_channels,
            transformer_num_layers=config.transformer_num_layer,
            use_AdaIN=True,
        ).to(device)


    unet = torch.nn.parallel.DistributedDataParallel(unet, device_ids=[local_rank], 
                                                     output_device=local_rank,)
    
    return unet

def generate_Inferer(scheduler, scale_factor, config):
    if config.train_model == 'LDM' or config.train_model == 'AdaLDM':
        inferer = LongLDMInferer(scheduler, scale_factor=scale_factor,)
    else:
        raise ValueError(f"Model {config.model} not implemented")

    return inferer


def generate_scheduler(config):
    if config.scheduler == 'ddpm':
        if config.schedule_type == 'cosine':
            scheduler = DDPMScheduler(num_train_timesteps=config.timestep, 
                                        schedule=config.schedule_type)
            
        if config.schedule_type == 'sigmoid':
            scheduler = DDPMScheduler(num_train_timesteps=config.timestep, 
                                      beta_start=config.beta_start, beta_end=config.beta_end,
                                      schedule='sigmoid_beta', )
        else:
            scheduler = DDPMScheduler(num_train_timesteps=config.timestep, 
                                      beta_start=config.beta_start, beta_end=config.beta_end,
                                      schedule=config.schedule_type, sig_range=config.sig_range)
            
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


def get_state_dict(model):
    return model.module.state_dict() if hasattr(model, 'module') else model.state_dict()