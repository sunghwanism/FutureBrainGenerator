
import argparse


################### LDMCf Configuration ###################

def get_run_parser():
    parser = argparse.ArgumentParser()

    ####################### BASE Configuration #######################
    # Seed
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Model
    parser.add_argument('--enc_model', type=str,
                        default='ckpt/VQGAN/dim8/best_vqvae_model_dim8_reconloss0.01_ep100.pth',
                        help='Encoder File name')
    parser.add_argument('--train_model', type=str, default='LDM',
                        help='Which model to run')
    
    # BASE
    parser.add_argument('--base_path', type=str, 
                        default=f'/NFS/FutureBrainGen/',
                        )
    
    
    # Data
    parser.add_argument('--data_path', type=str,
                        default=f'/NFS/FutureBrainGen/data/long',
                        # default='/local_datasets/msh2044/long',
                        help='Path to data')
    parser.add_argument('--crop_size',type=int, nargs='+', default=(96, 112, 96),)
    parser.add_argument('--use_transform', action='store_true',)
    
    # Device Arguments
    parser.add_argument('--device_id', type=str,
                        help='Which GPU to use')
    parser.add_argument('--num_workers', type=int, default=5,
                        help='Number of workers for dataloader')

    ####################### LDM Configuration #######################

    # Train Arguments
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of epochs')    
    parser.add_argument('--unet_lr', type=float, default=2e-4,
                        help='Generator Learning rate')
    parser.add_argument('--condition', nargs='+', default=['Age', 'Sex'],
                        help='Condition for classifier')
    
    # Diffusion Scheduler Arguments
    parser.add_argument('--scheduler', default='ddpm', # or ddim
                        help='Scheduler')
    parser.add_argument('--schedule_type', type=str, default='linear_beta',)
    parser.add_argument('--timestep', type=int, default=1000,)
    parser.add_argument('--beta_start', type=float, default=0.0015,) # 0.0015
    parser.add_argument('--beta_end', type=float, default=0.0205,) # 0.0195 

    # Hyperparameters
    parser.add_argument('--diff_num_channels', type=int, nargs='+', default=(64, 128, 256),
                        help="List of channel sizes")
    parser.add_argument('--diff_num_res_blocks', type=int, default=(1, 1, 1))
    parser.add_argument('--diff_num_head_channels', type=int, nargs='+', default=(0, 64, 128),)
    parser.add_argument('--diff_attention_levels', type=int, nargs='+', default=(0, 1, 1),)

    # Save and Log Arguments
    parser.add_argument('--save_path', type=str,
                        default=f'/NFS/FutureBrainGen/ckpt/LDM',
                        # default='/data/msh2044/results/FutureBrainGen/ckpt/LDM',
                        help='Where to save the model')
    parser.add_argument('--save_img_path', type=str,
                        default=f'/NFS/FutureBrainGen/results/LDM/img',
                        # default='/data/msh2044/results/FutureBrainGen/img/LDM',
                        help='Where to save the images')
    parser.add_argument('--save_interval', type=int, default=1,
                        help='How often to save')
    
    parser.add_argument('--save_img_interval', type=int, default=1,
                        help='How often to save images')
    parser.add_argument('--n_example_images', type=float, default=2,
                        help='Validation images')
    
    # Wandb Arguments
    parser.add_argument('--wandb_project', type=str, default='FutureBrain_LDM',
                        help='Wandb project')
    parser.add_argument('--wandb_entity', type=str, default='msh2044',
                        help='Wandb entity')
    parser.add_argument('--nowandb', action='store_true',
                        help='Don\'t use wandb')
    
    return parser

########################################################################################################

