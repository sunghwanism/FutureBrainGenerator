import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../MONAI')))

import numpy as np

import torch
import torch.nn.functional as F

import torch.distributed as dist
from torch.nn import L1Loss
import torchio as tio

from monai.config import print_config
from monai.networks.layers import Act
from monai.utils import set_determinism, first

from tqdm import tqdm
import wandb

from script.utils import longitudinal_load_dataloader
from script.configure.LDMconfig import get_run_parser

import warnings
warnings.filterwarnings("ignore")


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
     train_sampler, first_batch) = longitudinal_load_dataloader(config, world_size, rank)

    # Load VQ-VAE model
    VQVAEPATH = os.path.join(config.base_path, config.enc_model)

    EDmodel = load_VQVAE(config, device, VQVAEPATH)
    EDmodel.eval()
    
    # Calculate the scale factor of latent space
    with torch.no_grad():
        z = EDmodel.encode_stage_2_inputs(first_batch['base_img'].to(device))
    scale_factor = 1 / torch.std(z)
    
    cond_size = len(config.condition) + config.latent_channels + 1 # 2 (Age, Sex) + baseimg latent space + interval 
    
    unet = generate_unet(config, device, cond_size, local_rank,)
    scheduler = generate_scheduler(config)

    inferer = generate_Inferer(scheduler, scale_factor, config)
    
    optimizer_diff = torch.optim.AdamW(params=unet.parameters(), lr=config.unet_lr)
    unet_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_diff, T_max=50, eta_min=0)
    
    config_dict = vars(config)

    if rank == 0 and not config.nowandb:
        configPath = os.path.join(config.save_path, f'config_{wandb.run.name}.json')
        with open(configPath, 'w') as f:
            json.dump(config_dict, f, indent=4)

    ############################################# Training Process #############################################    
    for epoch in range(config.epochs):
        
        unet.train()
        if config.use_clf:
            classifier.train()
            clf_loss = 0

        epoch_loss = 0
        total_losses = 0
        train_sampler.set_epoch(epoch)

        if rank == 0:
            progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=150)
        else:
            progress_bar = enumerate(train_loader)

        for step, batch in progress_bar:
            
            base_img = batch['base_img'].to(device)
            follow_img = batch['follow_img'].to(device)
            true_y = batch["control_B"].to(device)
            condition = batch["condition"].to(device)
            
            if config.use_baseimg:
                base_img_z = EDmodel.encode_stage_2_inputs(base_img).flatten(1).unsqueeze(1)
                base_img_z = base_img_z * scale_factor
                condition = torch.cat([base_img_z, condition], dim=2)
                

            optimizer_diff.zero_grad(set_to_none=True)

            noise = torch.randn_like(z).to(device)
            timesteps = torch.randint(0,
                                      inferer.scheduler.num_train_timesteps, 
                                      (base_img.shape[0],), 
                                      device=base_img.device).long()

            if config.use_baseimg:
                noise_pred, logit_pred = inferer(inputs=follow_img, autoencoder_model=EDmodel,
                                                diffusion_model=unet, noise=noise, timesteps=timesteps,
                                                condition=condition,
                                                mode='crossattn', quantized=True,)
            else:
                noise_pred = inferer(inputs=follow_img, autoencoder_model=EDmodel,
                                                diffusion_model=unet, noise=noise, timesteps=timesteps,
                                                condition=condition,
                                                mode='crossattn', quantized=True)
            if not config.use_clf:
                logit_pred = 0
                
            diff_loss = F.mse_loss(noise_pred.float(), noise.float())
            
            if config.use_clf:
                pred_loss = F.cross_entropy(logit_pred, true_y)
                total_loss = diff_loss * config.diff_weight + pred_loss * config.clf_weight
                clf_loss += pred_loss.item()

            else:
                total_loss = diff_loss
                clf_loss = 0
            
            total_loss.backward()
            if (epoch+1) > 100:
                torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=0.5)

            optimizer_diff.step()

            epoch_loss += diff_loss.item()
            total_losses += total_loss.item()

            if rank == 0:
                progress_bar.set_postfix(
                    {
                        "epoch": epoch+1,
                        "total_loss": round(total_losses / (step + 1), 4),
                        "recon_loss": round(epoch_loss / (step + 1), 4),
                        "clf_loss": round(clf_loss / (step + 1), 4) if config.use_clf else 0,
                    }
                )
        
        epoch_loss, clf_loss, total_losses = merge_loss_all_rank([epoch_loss, clf_loss, total_losses],
                                                                 device, world_size, len(train_loader))
        # Log in wandb
        if rank == 0 and not config.nowandb:
            wandb.log({
                "epoch": epoch+1,
                "noise_loss": epoch_loss,
                'clf_loss': clf_loss if config.use_clf else 0,
                'total_loss': total_losses,
                'lr_diff': unet_lr_scheduler.optimizer.param_groups[0]['lr'],
            })

        # Model Save
        if rank == 0 and ((epoch+1) % config.save_interval == 0):

            save_dict = {
                "epoch": epoch+1,
                "unet_state_dict": unet.state_dict(),
                'classifier_state_dict': classifier.state_dict() if config.use_clf else None,
                'encoder_state_dict': EDmodel.state_dict(),
                "optimizer_diff": optimizer_diff.state_dict(),
                "scheduler": unet_lr_scheduler.state_dict(),
            }

            if config.use_clf:
                torch.save(save_dict, os.path.join(config.save_path, f"diffusion_ddp_ep{epoch+1}_dim{config.latent_channels}_{wandb.run.name}.pth"))
            else:
                torch.save(save_dict, os.path.join(config.save_path, f"diffusion_ddp_noclf_ep{epoch+1}_dim{config.latent_channels}_{wandb.run.name}.pth"))
                
            print(f"Model saved at epoch {epoch+1} with noise loss {epoch_loss}")
        
        unet_lr_scheduler.step()

        if rank == 0 and (((epoch+1) % config.save_img_interval == 0) or (epoch+1) == 1):
            unet.eval()
            EDmodel.eval()

            if config.use_clf:
                classifier.eval()

            noise = torch.randn_like(z).to(device)
            scheduler.set_timesteps(num_inference_steps=config.timestep)

            synthetic_images, intermediate_img = inferer.sample(input_noise=noise, 
                                              autoencoder_model=EDmodel, 
                                              diffusion_model=unet,
                                              conditioning=condition,
                                              mode='crossattn',
                                              save_intermediates=True,
                                              verbose=True,
                                              scheduler=scheduler)
            
            intermediate_img = [img.detach().cpu().numpy() for img in intermediate_img]
            intermediate_img = np.array(intermediate_img)

            save_img_dict = {
                "epoch": epoch+1,
                'follow_img': follow_img[:config.n_example_images],
                "base_img": base_img[:config.n_example_images],
                "synthetic_images": synthetic_images[:config.n_example_images],
                'intermediate_img': intermediate_img[:, :config.n_example_images],
                'condition': condition[:config.n_example_images],
                'control_B': true_y[:config.n_example_images],
                'control_F': batch['control_F'][:config.n_example_images],
                'Age_F': batch['Age_F'][:config.n_example_images],
                'Age_B': batch['Age_B'][:config.n_example_images],
            }

            if config.use_clf:
                torch.save(save_img_dict, os.path.join(config.save_img_path, f"diffusion_image_ddp_clf_ep{epoch+1}_dim{config.latent_channels}_{wandb.run.name}.pth"))
            else:
                torch.save(save_img_dict, os.path.join(config.save_img_path, f"diffusion_image_ddp_noclf_ep{epoch+1}_dim{config.latent_channels}_{wandb.run.name}.pth"))

            del save_img_dict, intermediate_img, synthetic_images
            
if __name__ == "__main__":
    parser = LDMCf_get_run_parser()
    config = parser.parse_args()
    print(config)
    main(config)

