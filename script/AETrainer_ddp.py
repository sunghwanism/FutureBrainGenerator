import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../MONAI')))

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import autocast
import torch.distributed as dist

from monai.config import print_config
from monai.networks.layers import Act
from monai.utils import set_determinism

from tqdm import tqdm
import wandb

from MONAI.generative.losses import PatchAdversarialLoss, PerceptualLoss
from MONAI.generative.networks.nets import AutoencoderKL, PatchDiscriminator

from script.utils import init_wandb
from script.configure.config import get_run_parser

from dataloader.AEDataset import MRIDataset
import warnings

warnings.filterwarnings("ignore")
torch.autograd.set_detect_anomaly(True)

def main(config):
    
    print_config()

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
        init_wandb(config)

    #######################################################################################

    TrainDataset = MRIDataset(config.data_path, config, _type='train')
    ValDataset = MRIDataset(config.data_path, config, _type='val')

    train_sampler = DistributedSampler(TrainDataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(ValDataset, num_replicas=world_size, rank=rank)

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
        sampler=val_sampler
    )

    #######################################################################################

    model = AutoencoderKL(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        num_channels=config.num_channels,
        latent_channels=config.latent_channels,
        num_res_blocks=config.num_res_blocks,
        norm_num_groups=config.norm_num_groups,
        attention_levels=(False, False, True),
    ).to(device)

    discriminator = PatchDiscriminator(
        spatial_dims=3,
        num_layers_d=3,
        num_channels=32,
        in_channels=1,
        out_channels=1,
        kernel_size=5,
        activation=(Act.LEAKYRELU, {"negative_slope": 0.2, "inplace":False}),
        norm="BATCH",
        bias=False,
        padding=1,
    ).to(device)
    
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    discriminator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(discriminator)
    
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    discriminator = torch.nn.parallel.DistributedDataParallel(discriminator, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    perceptual_loss = PerceptualLoss(spatial_dims=3, network_type="squeeze", fake_3d_ratio=0.25).to(device)
    adv_loss = PatchAdversarialLoss(criterion="least_squares")

    optimizer_g = torch.optim.Adam(model.parameters(), lr=config.gen_lr)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=config.disc_lr)

    scaler_g = torch.cuda.amp.GradScaler()
    scaler_d = torch.cuda.amp.GradScaler()

    best_validation_loss = np.inf

    for epoch in range(config.epochs):
        
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)

        model.train()
        discriminator.train()

        epoch_loss = 0
        gen_epoch_loss = 0
        disc_epoch_loss = 0
        kl_losses = 0

        if rank == 0:
            progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=120)
        else:
            progress_bar = enumerate(train_loader)

        for step, batch in progress_bar:
            images = batch.to(device, non_blocking=True)
            optimizer_g.zero_grad(set_to_none=True)

            # Generator part
            with autocast(enabled=False):
                reconstruction, z_mu, z_sigma = model(images)
                logits_fake = discriminator(reconstruction.contiguous().float())[-1]

                recons_loss = F.l1_loss(reconstruction.float(), images.float())
                p_loss = perceptual_loss(reconstruction.float(), images.float())
                generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)

                kl_loss = 0.5 * torch.sum(
                    z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1,
                    dim=[1, 2, 3, 4]
                )
                kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

                loss_g = recons_loss + (config.kl_weight * kl_loss) + \
                         (config.perceptual_weight * p_loss) + (config.adv_weight * generator_loss)
                kl_losses += kl_loss.item()
            
            scaler_g.scale(loss_g).backward()
            scaler_g.step(optimizer_g)
            scaler_g.update()

            # Discriminator part
            optimizer_d.zero_grad(set_to_none=True)

            with autocast(enabled=False):
                logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                
                logits_real = discriminator(images.contiguous().detach())[-1]
                loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                
                discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

                loss_d = config.adv_weight * discriminator_loss

            scaler_d.scale(loss_d).backward()
            scaler_d.step(optimizer_d)
            scaler_d.update()

            epoch_loss += recons_loss.item()
            gen_epoch_loss += generator_loss.item()
            disc_epoch_loss += discriminator_loss.item()

            if rank == 0:
                progress_bar.set_postfix(
                    {
                        "recons_loss": epoch_loss / (step + 1),
                        "gen_loss": gen_epoch_loss / (step + 1),
                        "disc_loss": disc_epoch_loss / (step + 1),
                    }
                )
            
            
        total_loss_tensor = torch.tensor([epoch_loss, gen_epoch_loss, disc_epoch_loss, kl_losses], device=device)
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        total_loss_tensor /= world_size

        epoch_loss = total_loss_tensor[0].item() / len(train_loader)
        gen_epoch_loss = total_loss_tensor[1].item() / len(train_loader)
        disc_epoch_loss = total_loss_tensor[2].item() / len(train_loader)
        kl_losses = total_loss_tensor[3].item() / len(train_loader)

        if rank == 0 and not config.nowandb:
            wandb.log({
                "epoch": epoch,
                "Train-reconLoss": epoch_loss,
                "Train-klLoss": kl_losses,
                "Train-genLoss": gen_epoch_loss,
                "Train-discLoss": disc_epoch_loss,
            })

        if (epoch + 1) % config.save_interval == 0:
            model.eval()
            val_loss = 0

            with torch.no_grad():
                for val_step, batch in enumerate(val_loader, start=1):
                    images = batch.to(device, non_blocking=True)

                    reconstruction, z_mu, z_sigma = model(images)
                    recons_loss = F.l1_loss(reconstruction.float(), images.float())
                    val_loss += recons_loss.item()

                    if (epoch + 1) % config.save_img_interval == 0 and val_step == 1 and rank == 0:
                        recon_sample_imgs = reconstruction[:config.n_example_images]

                        for i in range(config.n_example_images):
                            orig_mri = images[i].detach().cpu().numpy()
                            recon_mri = recon_sample_imgs[i].detach().cpu().numpy()

                            np.savez(os.path.join(config.save_img_path, f"recon_ep{epoch}_{i}_ddp.npz"),
                                     origin=orig_mri, recon=recon_mri)

                        del orig_mri, recon_mri, recon_sample_imgs

            
            val_loss_tensor = torch.tensor(val_loss, device=device)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
            val_loss = val_loss_tensor.item() / (val_step * world_size)

            if rank == 0 and not config.nowandb:
                wandb.log({
                    "epoch": epoch,
                    "Val-reconLoss": val_loss,
                })

            if rank == 0 and best_validation_loss > val_loss:
                best_validation_loss = val_loss
                torch.save(
                    {
                        'autoencoder': model.module.state_dict(),
                        'discriminator': discriminator.module.state_dict(),
                    },
                    os.path.join(config.save_path, f"best_{config.train_model}_model_ddp.pth"),
                )
                print(f"Best model saved at epoch {epoch + 1} with loss {best_validation_loss}")

    dist.destroy_process_group()


if __name__ == '__main__':
    parser = get_run_parser()
    config = parser.parse_args()
    print(config)
    main(config)
