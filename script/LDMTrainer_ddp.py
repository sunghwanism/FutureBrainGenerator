import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../MONAI')))

import numpy as np
import json

import torch
import torch.nn.functional as F

import torch.distributed as dist
import torchio as tio
import gc

from monai.config import print_config
from monai.utils import set_determinism

from tqdm import tqdm
import wandb

from script.utils import *
from script.configure.LDMconfig import get_run_parser

import warnings
warnings.filterwarnings("ignore")


def main(config):

    dist.init_process_group(backend='nccl', init_method='env://')
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])

    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)

    set_determinism(seed=config.seed)
    torch.backends.cudnn.benchmark = True
    
    if rank == 0 and not config.nowandb:
        init_wandb(config)
        wandb_save_path = os.path.join(config.save_path, f'{wandb.run.name}')
        wandb_img_path = os.path.join(config.save_img_path, f'{wandb.run.name}')

        if not os.path.exists(wandb_save_path):
            os.makedirs(wandb_save_path)
        if not os.path.exists(wandb_img_path):
            os.makedirs(wandb_img_path)
    
    if local_rank == 0:
        print("******"*20)
        print(config)
        print(f"Using device {device} on rank {rank}")
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

    # Load DataLoader
    (train_loader, val_loader, 
     train_sampler, first_batch) = longitudinal_load_dataloader(config, world_size, rank,
                                                                train_transform, val_transform)

    # Load VQ-VAE model
    VQVAEPATH = os.path.join(config.base_path, config.enc_model)

    EDmodel = load_VQVAE(device, VQVAEPATH, wrap_ddp=False, local_rank=None)
    EDmodel.eval()
    
    # Calculate the scale factor of latent space
    with torch.no_grad():
        z = EDmodel.encode_stage_2_inputs(first_batch['base_img'].to(device))
    scale_factor = 1 # / torch.std(z)
    
    base_img_size = z[0].numel()
    latent_dim = z.shape[1]
    
    unet = generate_unet(config, device, base_img_size, latent_dim, local_rank)
    scheduler = generate_scheduler(config)

    inferer = generate_Inferer(scheduler, scale_factor, config)
    
    optimizer_diff = torch.optim.Adam(params=unet.parameters(), lr=config.unet_lr)
    unet_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_diff, T_max=50, eta_min=0)
    
    config_dict = vars(config)

    if rank == 0 and not config.nowandb:
        configPath = os.path.join(wandb_save_path, f'config_{wandb.run.name}.json')
        with open(configPath, 'w') as f:
            json.dump(config_dict, f, indent=4)

    del first_batch
    gc.collect()
    torch.cuda.empty_cache()
    
    ############################################# Training Process #############################################    
    for epoch in range(config.epochs):
        
        unet.train()
        EDmodel.eval()

        epoch_loss = 0
        train_sampler.set_epoch(epoch)

        if rank == 0:
            progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=150)
        else:
            progress_bar = enumerate(train_loader)

        for step, batch in progress_bar:
            
            base_img = batch['base_img'].to(device)
            follow_img = batch['follow_img'].to(device)
            condition = batch["condition"].to(device)
            
            base_img_z = EDmodel.encode_stage_2_inputs(base_img).flatten(1).unsqueeze(1)
            base_img_z = base_img_z * scale_factor
            base_img_z = base_img_z.to(device) + batch['interval'].to(device)
            
            optimizer_diff.zero_grad(set_to_none=True)

            noise = torch.randn_like(z).to(device)
            timesteps = torch.randint(0,
                                      inferer.scheduler.num_train_timesteps, 
                                      (base_img.shape[0],), 
                                      device=base_img_z.device).long()

            noise_pred = inferer(inputs=follow_img, autoencoder_model=EDmodel,
                                 diffusion_model=unet, noise=noise, timesteps=timesteps,
                                 condition=base_img_z,
                                 clinical_cond=condition,
                                 mode='crossattn', quantized=True)
                
            diff_loss = F.mse_loss(noise_pred.float(), noise.float())
            diff_loss.backward()

            optimizer_diff.step()

            epoch_loss += diff_loss.item()

            if rank == 0:
                progress_bar.set_postfix(
                    {
                        "epoch": epoch+1,
                        "noise_loss": round(epoch_loss / (step + 1), 5),
                    }
                )
        
        epoch_loss= merge_loss_all_rank([epoch_loss], device, world_size, len(train_loader))

        # Log to wandb
        if rank == 0 and not config.nowandb:
            wandb.log({
                "epoch": epoch+1,
                "noise_loss": epoch_loss,
                'lr_diff': unet_lr_scheduler.optimizer.param_groups[0]['lr'],
            })

        # Model Save
        if rank == 0 and ((epoch+1) % config.save_interval == 0):

            save_dict = {
                "epoch": epoch+1,
                "unet_state_dict": get_state_dict(unet),
                "optimizer_diff": optimizer_diff.state_dict(),
                "scheduler": unet_lr_scheduler.state_dict(),
            }

            torch.save(save_dict, os.path.join(wandb_save_path, f"{config.train_model}_ep{epoch+1}_dim{latent_dim}_{wandb.run.name}.pth"))
            print(f"Model saved at epoch {epoch+1} with noise loss {epoch_loss}")
            del save_dict
        
        unet_lr_scheduler.step()

        del base_img, follow_img, base_img_z, noise, condition
        gc.collect()
        torch.cuda.empty_cache()

        if rank == 0 and (((epoch+1) % config.save_img_interval == 0) or (epoch+1) == 1):
            print("Generating Synthetic Images using Validation Dataset...")

            unet.eval()
            EDmodel.eval()
            val_loss = 0

            with torch.no_grad():
                for idx, batch in enumerate(val_loader):
                    base_img = batch['base_img'].to(device)
                    follow_img = batch['follow_img'].to(device)
                    condition = batch["condition"].to(device)
                    
                    base_img_z = EDmodel.encode_stage_2_inputs(base_img).flatten(1).unsqueeze(1)
                    base_img_z = base_img_z * scale_factor
                    base_img_z = base_img_z + batch['interval'].to(device)

                    noise = torch.randn_like(z).to(device)
                    scheduler.set_timesteps(num_inference_steps=config.timestep)

                    synthetic_images, intermediate_img = inferer.sample(input_noise=noise, 
                                                                        autoencoder_model=EDmodel,
                                                                        diffusion_model=unet,
                                                                        conditioning=base_img_z,
                                                                        clinical_cond=condition,
                                                                        mode='crossattn',
                                                                        save_intermediates=True if idx == 0 else False,
                                                                        intermediate_steps=200,
                                                                        verbose=True,
                                                                        scheduler=scheduler)

                    recons_loss = F.l1_loss(synthetic_images.float(), follow_img.float())
                    val_loss += recons_loss.item()

                    if idx == 0:
                        intermediate_img = [img.detach().cpu().numpy() for img in intermediate_img]
                        intermediate_img = np.array(intermediate_img)

                        save_img_dict = {
                            "epoch": epoch+1,
                            'follow_img': follow_img[:config.n_example_images],
                            "base_img": base_img[:config.n_example_images],
                            "synthetic_images": synthetic_images[:config.n_example_images],
                            'intermediate_img': intermediate_img[:, :config.n_example_images],
                            'condition': condition[:config.n_example_images],
                            'Sex': batch['Sex'][:config.n_example_images],
                            'Age_F': batch['Age_F'][:config.n_example_images],
                            'Age_B': batch['Age_B'][:config.n_example_images],
                        }

                    torch.save(save_img_dict, os.path.join(wandb_img_path, f"{config.train_model}_ep{epoch+1}_dim{latent_dim}_{wandb.run.name}.pth"))
                
                epoch_val_loss = merge_loss_all_rank([val_loss], device, world_size, len(val_loader))
                
                # Log to wandb
                if rank == 0 and not config.nowandb:
                    wandb.log({
                        "epoch": epoch+1,
                        "val_recon_loss": epoch_val_loss,
                    })

            del save_img_dict, intermediate_img, synthetic_images, base_img, follow_img, base_img_z, noise, condition
            gc.collect()
            torch.cuda.empty_cache()
            
if __name__ == "__main__":
    parser = get_run_parser()
    config = parser.parse_args()
    main(config)

