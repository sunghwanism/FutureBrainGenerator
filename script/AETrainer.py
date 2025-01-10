import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../MONAI')))

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

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


def main(config):
    print_config() # Print MONAI configuration
    set_determinism(config.seed) # Set Seed

    #######################################################################################
    
    TrainDataset = MRIDataset(config.data_path, config, _type='train')
    Valdataset = MRIDataset(config.data_path, config, _type='val')
    
    train_loader = DataLoader(TrainDataset, batch_size=config.batch_size, 
                              num_workers=config.num_workers, shuffle=True)
    val_loader = DataLoader(Valdataset, batch_size=config.batch_size,
                            num_workers=config.num_workers, shuffle=False)
    
    #######################################################################################
    
    device = torch.device(f"cuda:{config.device_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    
    model = AutoencoderKL(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        num_channels=config.num_channels,
        latent_channels=config.latent_channels,
        num_res_blocks=config.num_res_blocks,
        norm_num_groups=config.norm_num_groups,
        attention_levels=(False, False, True),
    )
    model.to(device)
    
    discriminator = PatchDiscriminator(
    spatial_dims=3,
    num_layers_d=3,
    num_channels=32,
    in_channels=1,
    out_channels=1,
    kernel_size=5,
    activation=(Act.LEAKYRELU, {"negative_slope": 0.2}),
    norm="BATCH",
    bias=False,
    padding=1,
    )
    discriminator.to(device)
    
    perceptual_loss = PerceptualLoss(spatial_dims=3, network_type="squeeze", fake_3d_ratio=0.25)
    perceptual_loss.to(device)

    adv_loss = PatchAdversarialLoss(criterion="least_squares")

    optimizer_g = torch.optim.Adam(model.parameters(), lr=config.gen_lr)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=config.disc_lr)
    
    scaler_g = torch.cuda.amp.GradScaler()
    scaler_d = torch.cuda.amp.GradScaler()
    
    best_validation_loss = np.inf
    
    
    for epoch in range(config.epochs):
        model.train()
        discriminator.train()
        epoch_loss = 0
        gen_epoch_loss = 0
        disc_epoch_loss = 0
        kl_losses = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=70)
        progress_bar.set_description(f"Epoch {epoch}")
        
        for step, batch in progress_bar:
            images = batch.to(device)
            optimizer_g.zero_grad(set_to_none=True)
            # Generator part
            with autocast(enabled=True):
                reconstruction, z_mu, z_sigma = model(images)
                logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                
                recons_loss = F.l1_loss(reconstruction.float(), images.float())
                p_loss = perceptual_loss(reconstruction.float(), images.float())
                generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)

                kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3, 4])
                kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

                loss_g = recons_loss + (config.kl_weight * kl_loss) + (config.perceptual_weight * p_loss) + (config.adv_weight * generator_loss)
                kl_losses += kl_loss.item()
            
            
            scaler_g.scale(loss_g).backward()
            scaler_g.step(optimizer_g)
            scaler_g.update()
            
            # Discriminator part
            optimizer_d.zero_grad(set_to_none=True)

            with autocast(enabled=True):
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

            progress_bar.set_postfix(
                {
                    "recons_loss": epoch_loss / (step + 1),
                    "gen_loss": gen_epoch_loss / (step + 1),
                    "disc_loss": disc_epoch_loss / (step + 1),
                }
            )
            
        if not config.nowandb:
            wandb.log({"epoch": epoch,
                        "Train-reconLoss": epoch_loss / (step + 1),
                        "Train-klLoss": kl_losses / (step + 1),
                        "Train-genLoss": gen_epoch_loss / (step + 1),
                        "Train-discLoss": disc_epoch_loss / (step + 1),
                        })
        
        if (epoch + 1) % config.save_interval == 0:
            model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for val_step, batch in enumerate(val_loader, start=1):
                    images = batch.to(device)
                    optimizer_g.zero_grad(set_to_none=True)

                    reconstruction, z_mu, z_sigma = model(images)
                    recons_loss = F.l1_loss(reconstruction.float(), images.float())
                    val_loss += recons_loss.item()
                    
                    if (epoch + 1) % config.save_img_interval == 0 and val_step == 1:
                        if val_step == 1:               
                            recon_sample_imgs = reconstruction[:config.n_example_images]
                            
                            for i in range(config.n_example_images):
                                orig_mri = images[i].detach().cpu().numpy()
                                recon_mri = recon_sample_imgs[i].detach().cpu().numpy()
                                
                                np.savez(os.path.join(config.save_img_path, f"recon_ep{epoch}_{i}.npz"),
                                        origin=orig_mri, recon=recon_mri)
                                
                            del orig_mri, recon_mri, recon_sample_imgs

            val_loss /= val_step
            
            if not config.nowandb:
                wandb.log({"epoch": epoch,
                            "Val-reconLoss": val_loss,
                            })
            
            if best_validation_loss > val_loss:
                best_validation_loss = val_loss
                torch.save(
                        {'autoencoder': model.state_dict(),
                         'discriminator': discriminator.state_dict(),},
                        os.path.join(config.save_path, f"best_{config.train_model}_model.pth"),
                )
                print(f"Best model saved at epoch {epoch + 1} with loss {best_validation_loss}")

    progress_bar.close()


if __name__ == '__main__':
    parser = get_run_parser()
    config = parser.parse_args()
    print(config)
    
    init_wandb(config)
    main(config)