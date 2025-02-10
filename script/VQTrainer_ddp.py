import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../MONAI')))

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
import torchio as tio

from monai.config import print_config
from monai.networks.layers import Act
from monai.utils import set_determinism

from tqdm import tqdm
import wandb
import gc

from MONAI.generative.losses import PatchAdversarialLoss, PerceptualLoss
from MONAI.generative.networks.nets import VQVAE, PatchDiscriminator

from script.utils import init_wandb
from script.configure.VQGANconfig import get_run_parser

from data.CrossDataset import CrossMRIDataset
import warnings

warnings.filterwarnings("ignore")


def main(config):

    dist.init_process_group(backend='nccl', init_method='env://')
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])

    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    print(f"Using device {device} on rank {rank}")

    set_determinism(seed=config.seed)
    torch.backends.cudnn.benchmark = True
    
    if rank == 0 and not config.nowandb:
        init_wandb(config,)
        wandb_save_path = os.path.join(config.save_path, f'{wandb.run.name}')
        wandb_img_path = os.path.join(config.save_img_path, f'{wandb.run.name}')

        if not os.exists(wandb_save_path):
            os.makedirs(wandb_save_path)

        if not os.exists(wandb_img_path):
            os.makedirs(wandb_img_path)

    if local_rank == 0:
        print("******"*20)
        print(config)
        print(f"Using device {device} on rank {rank}")
        print("******"*20)

    #######################################################################################
    if config.use_transform:
        train_transform = tio.Compose([
            tio.RandomElasticDeformation(num_control_points=5, max_displacement=5, p=0.3),
            tio.RandomAffine(scales=(0.90, 1.1), p=0.5),
            tio.RandomAffine(degrees=(0, 10), p=0.5),
            tio.RandomAffine(translation=(5, 5, 5), p=0.5),
            tio.RescaleIntensity(out_min_max=(0, 1)),
            ])
        val_transform = tio.Compose([
            tio.RescaleIntensity(out_min_max=(0, 1)),])
    else:
        train_transform = None
        val_transform = None
    #######################################################################################
    
    TrainDataset = CrossMRIDataset(config, _type='train', Transform=train_transform)
    ValDataset = CrossMRIDataset(config, _type='val', Transform=val_transform)

    train_sampler = DistributedSampler(TrainDataset, num_replicas=world_size, rank=rank)

    train_loader = DataLoader(
        TrainDataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
        sampler=train_sampler
    )
    val_loader = DataLoader(
        ValDataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    #######################################################################################

    model = VQVAE(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        num_channels=config.num_channels,
        num_res_channels=config.num_res_channels,
        num_res_layers=config.num_res_blocks,
        commitment_cost = config.commitment_cost,
        downsample_parameters=config.downsample_param, 
        upsample_parameters=config.upsample_param,
        num_embeddings=config.num_embeddings,
        embedding_dim=config.latent_channels,
        ddp_sync=True,
    )
    model.to(device)

    discriminator = PatchDiscriminator(
        in_channels=1,
        spatial_dims=3,
        num_layers_d=3,
        num_channels=32,
        kernel_size=4,
        activation=(Act.LEAKYRELU, {"negative_slope": 0.2, "inplace":False}),
        norm="BATCH",
        bias=False,
    ).to(device)
    
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, )
    
    discriminator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(discriminator)
    discriminator = torch.nn.parallel.DistributedDataParallel(discriminator, device_ids=[local_rank], output_device=local_rank,)
    
    perceptual_loss = PerceptualLoss(spatial_dims=3, 
                                     network_type=config.perceptual_model, 
                                     is_fake_3d=False).to(device)
    
    adv_loss = PatchAdversarialLoss(criterion="least_squares")

    optimizer_g = torch.optim.Adam(model.parameters(), lr=config.gen_lr)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=config.disc_lr)

    scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_g, T_max=30, eta_min=0)
    scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_d, T_max=30, eta_min=0)
    
    for epoch in range(config.epochs):
        
        train_sampler.set_epoch(epoch)

        model.train()
        discriminator.train()

        epoch_loss = 0
        gen_epoch_loss = 0
        disc_epoch_loss = 0
        quat_epoch_loss = 0
        percept_epoch_loss = 0

        if rank == 0:
            progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=160)
        else:
            progress_bar = enumerate(train_loader)

        for step, batch in progress_bar:
            # Geneartor part
            images = batch.to(device, non_blocking=True)
            optimizer_g.zero_grad(set_to_none=True)

            reconstruction, quantization_loss = model(images)
            logits_fake = discriminator(reconstruction.contiguous().float())[-1]

            recon_loss = F.l1_loss(reconstruction.float(), images.float())
            p_loss = perceptual_loss(reconstruction.float(), images.float())
            generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)

            loss_g = recon_loss + (config.perceptual_weight * p_loss) + quantization_loss + (config.adv_weight * generator_loss)

            gen_epoch_loss += generator_loss.item()    
            epoch_loss += recon_loss.item()
            quat_epoch_loss += quantization_loss.item()
            percept_epoch_loss += p_loss.item()

            loss_g.backward()
            optimizer_g.step()

            # Discriminator part
            if epoch+1 > config.autoencoder_warm_up_n_epochs:
                optimizer_d.zero_grad(set_to_none=True)

                logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                
                logits_real = discriminator(images.contiguous().detach())[-1]
                loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                
                discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

                loss_d = config.adv_weight * discriminator_loss
                disc_epoch_loss += discriminator_loss.item()

                loss_d.backward()
                optimizer_d.step()
        
            if rank == 0:
                if epoch+1 > config.autoencoder_warm_up_n_epochs:
                    progress_bar.set_postfix(
                        {
                            "epoch": epoch+1,
                            "recon_loss": round(epoch_loss / (step + 1),4),
                            "p_loss": round(percept_epoch_loss / (step + 1),4),
                            "q_loss": round(quat_epoch_loss / (step + 1),6),
                            "g_loss": round(gen_epoch_loss / (step + 1), 4),
                            "d_loss": round(disc_epoch_loss / (step + 1), 4),
                        }
                    )
                else:
                    progress_bar.set_postfix(
                        {
                            "epoch": epoch+1,
                            "recon_loss": round(epoch_loss / (step + 1), 4),
                            "p_loss": round(percept_epoch_loss / (step + 1), 4),
                            "q_loss": round(quat_epoch_loss / (step + 1), 6),
                            "g_loss": round(gen_epoch_loss / (step + 1), 4),
                        }
                    )

        scheduler_g.step()

        if epoch+1 > config.autoencoder_warm_up_n_epochs:
            scheduler_d.step()
            total_loss_tensor = torch.tensor([epoch_loss, percept_epoch_loss, quat_epoch_loss, gen_epoch_loss, disc_epoch_loss], device=device)
            
        else:
            total_loss_tensor = torch.tensor([epoch_loss, percept_epoch_loss, quat_epoch_loss, gen_epoch_loss], device=device)

        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        total_loss_tensor /= world_size

        epoch_loss = total_loss_tensor[0].item() / len(train_loader)
        percept_epoch_loss = total_loss_tensor[1].item() / len(train_loader)
        quat_epoch_loss = total_loss_tensor[2].item() / len(train_loader)
        gen_epoch_loss = total_loss_tensor[3].item() / len(train_loader)

        if epoch+1 > config.autoencoder_warm_up_n_epochs:
            disc_epoch_loss = total_loss_tensor[4].item() / len(train_loader)

        if rank == 0 and not config.nowandb:
            if epoch+1 > config.autoencoder_warm_up_n_epochs:
                wandb.log({
                    "epoch": epoch+1,
                    "Train-reconLoss": epoch_loss,
                    "Train-perceptLoss": percept_epoch_loss,
                    "Train-quantLoss": quat_epoch_loss,
                    "Train-genLoss": gen_epoch_loss,
                    "Train-discLoss": disc_epoch_loss,
                    'lr_g': optimizer_g.param_groups[0]['lr'],
                    'lr_d': optimizer_d.param_groups[0]['lr']
                })
            else:
                wandb.log({
                    "epoch": epoch+1,
                    "Train-reconLoss": epoch_loss,
                    "Train-perceptLoss": percept_epoch_loss,
                    "Train-quantLoss": quat_epoch_loss,
                    "Train-genLoss": gen_epoch_loss,
                    "Train-discLoss": 0,
                    'lr_g': optimizer_g.param_groups[0]['lr'],
                    'lr_d': optimizer_d.param_groups[0]['lr']
                })

        if (epoch + 1) % config.save_interval == 0 or epoch == 0:
            model.eval()
            val_loss = 0
            val_quant_loss = 0

            with torch.no_grad():
                for val_step, batch in enumerate(val_loader, start=1):
                    images = batch.to(device, non_blocking=True)

                    reconstruction, quantization_loss = model(images)
                    recons_loss = F.l1_loss(reconstruction.float(), images.float())
                    val_loss += recons_loss.item()
                    val_quant_loss += quantization_loss.item()

                    if (epoch + 1) % config.save_img_interval == 0 and val_step == 1 and rank == 0:
                        recon_sample_imgs = reconstruction[:config.n_example_images]

                        for i in range(config.n_example_images):
                            orig_mri = images[i].detach().cpu().numpy()
                            recon_mri = recon_sample_imgs[i].detach().cpu().numpy()

                            np.savez(os.path.join(wandb_img_path, f"recon_ep{epoch+1}_{i}_dim{config.latent_channels}.npz"),
                                     origin=orig_mri, recon=recon_mri)

                        del orig_mri, recon_mri, recon_sample_imgs
            
            val_loss_tensor = torch.tensor([val_loss, val_quant_loss], device=device)

            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
            val_loss = val_loss_tensor[0].item() / (val_step * world_size)
            val_quant_loss = val_loss_tensor[1].item() / (val_step * world_size)

            if rank == 0 and not config.nowandb:
                wandb.log({
                    "epoch": epoch+1,
                    "Val-reconLoss": val_loss,
                    "Val-quantLoss": val_quant_loss,
                })

            if rank == 0:
                cpu_state_dict = {key: value.cpu() for key, value in model.module.state_dict().items()}

                torch.save(
                    {
                        'encoder': cpu_state_dict,
                        'config': config
                    },
                    os.path.join(wandb_save_path, f"best_{config.train_model}_model_dim{config.latent_channels}_reconloss{round(val_loss,3)}_ep{epoch+1}.pth"),
                )
            
            gc.collect()
            torch.cuda.empty_cache()

    dist.destroy_process_group()


if __name__ == '__main__':
    parser = get_run_parser()
    config = parser.parse_args()
    main(config)