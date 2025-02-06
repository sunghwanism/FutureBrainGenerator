
import argparse


################### LDMCf Configuration ###################

def LDMCf_get_run_parser():
    parser = argparse.ArgumentParser()

    ####################### BASE Configuration #######################
    # Seed
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Model
    parser.add_argument('--train_model', type=str, default='LDM',
                        help='Which model to run')
    
    # Data
    parser.add_argument('--data_path', type=str, default=f'/NFS/FutureBrainGen/data/long',
                        help='Path to data')
    parser.add_argument('--crop_size',type=int, nargs='+', default=(84, 104, 84),)
    parser.add_argument('--use_transform', action='store_true',)
    
    # Device Arguments
    parser.add_argument('--device_id', type=str,
                        help='Which GPU to use')
    parser.add_argument('--num_workers', type=int, default=5,
                        help='Number of workers for dataloader')

    ####################### LDM Configuration #######################

    # Finetuning Arguments
    parser.add_argument('--encoder_all_freeze', action='store_true',
                        help='Freeze all encoder layers')
    
    # Train Arguments
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=10000,
                        help='Number of epochs')    
    parser.add_argument('--unet_lr', type=float, default=1e-4,
                        help='Generator Learning rate')
    parser.add_argument('--condition', nargs='+', default=['Age', 'Sex', 'interval'],
                        help='Condition for classifier')
    
    # Diffusion Scheduler Arguments
    parser.add_argument('--scheduler', default='ddpm', # or ddim
                        help='Scheduler')
    parser.add_argument('--schedule_type', type=str, default='scaled_linear_beta',)
    parser.add_argument('--timestep', type=int, default=1000,)
    parser.add_argument('--beta_start', type=float, default=0.0015,) # 0.0015
    parser.add_argument('--beta_end', type=float, default=0.0195,) # 0.0195 

    # Hyperparameters
    parser.add_argument('--diff_num_channels', type=int, nargs='+', default=(128, 256, 512), help="List of channel sizes")
    parser.add_argument('--diff_num_res_blocks', type=int, default=1)
    parser.add_argument('--diff_num_head_channels', type=int, nargs='+', default=(0, 128, 256),)
    parser.add_argument('--diff_attention_levels', type=int, nargs='+', default=(0, 1, 1),)


    # Save and Log Arguments
    parser.add_argument('--save_path', type=str, default=f'/NFS/FutureBrainGen/ckpt/LDM',
                        help='Where to save the model')
    parser.add_argument('--save_img_path', type=str, default=f'/NFS/FutureBrainGen/results/LDM/img',
                        help='Where to save the images')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='How often to save')
    
    parser.add_argument('--save_img_interval', type=int, default=50,
                        help='How often to save images')
    parser.add_argument('--n_example_images', type=float, default=2,
                        help='Validation images')
    
    # Wandb Arguments
    parser.add_argument('--wandb_project', type=str, default='FutureGen_LDM',
                        help='Wandb project')
    parser.add_argument('--wandb_entity', type=str, default='msh2044',
                        help='Wandb entity')
    parser.add_argument('--nowandb', action='store_true',
                        help='Don\'t use wandb')
    
    return parser

########################################################################################################

