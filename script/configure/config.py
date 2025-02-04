import argparse

################### VQVAE Configuration ###################

def get_run_parser():
    parser = argparse.ArgumentParser()
    model = 'vqvae'

    # Seed
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Model
    parser.add_argument('--train_model', type=str, default='vqvae',
                        help='Which model to run')
    
    # Data
    parser.add_argument('--data_path', type=str, default=f'/NFS/FutureBrainGen/data/cross/',
                        help='Path to data')
    parser.add_argument('--crop_size',type=int, nargs='+', default=(86, 106, 86),)
    
    # Device Arguments
    parser.add_argument('--device_id', type=str,
                        help='Which GPU to use')
    parser.add_argument('--num_workers', type=int, default=5,
                        help='Number of workers for dataloader')

    # Train Arguments
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=150,
                        help='Number of epochs')    
    parser.add_argument('--gen_lr', type=float, default=1e-4,
                        help='Generator Learning rate')
    parser.add_argument('--disc_lr', type=float, default=5e-4,
                        help='Discriminator Learning rate')

    # Hyperparameters
    parser.add_argument('--num_embeddings', type=int, default=1024) # code book size
    parser.add_argument('--latent_channels', type=int, default=8)
    parser.add_argument('--num_channels', type=int, nargs='+', default=(64, 128, 256), help="List of channel sizes")
    parser.add_argument('--num_res_blocks', type=int, default=2)
    parser.add_argument('--norm_num_groups', type=int, default=32)
    parser.add_argument('--autoencoder_warm_up_n_epochs', type=int, default=1)
    parser.add_argument('--downsample_param', type=int, nargs='+', 
                        default=((1, 3, 1, 1), (2, 4, 1, 1), (2, 4, 1, 1)),
                        help='list of (stride, kernel, dilation, padding)')
    parser.add_argument('--upsample_param', type=int, nargs='+', 
                        default=((1, 3, 1, 1, 0), (2, 4, 1, 1, 0), (2, 4, 1, 1, 0),))
    
    # Loss Arguments
    parser.add_argument('--adv_weight', type=float, default=0.01)
    parser.add_argument('--perceptual_weight', type=float, default=0.0001)
    parser.add_argument('--commitment_cost', type=float, default=0.8) # base = 0.25
    
    # Save and Log Arguments
    parser.add_argument('--save_path', type=str, default=f'/NFS/ADDiff/ckpt/{model}',
                        help='Where to save the model')
    parser.add_argument('--save_img_path', type=str, default=f'/NFS/ADDiff/img/{model}',
                        help='Where to save the images')
    parser.add_argument('--save_interval', type=int, default=3,
                        help='How often to save')
    
    parser.add_argument('--save_img_interval', type=int, default=3,
                        help='How often to save images')
    parser.add_argument('--n_example_images', type=float, default=1,
                        help='Validation images')
    
    # Wandb Arguments
    parser.add_argument('--wandb_project', type=str, default='AD_diff_VQVAE_QC',
                        help='Wandb project')
    parser.add_argument('--wandb_entity', type=str, default='msh2044',
                        help='Wandb entity')
    parser.add_argument('--nowandb', action='store_true',
                        help='Don\'t use wandb')
    
    return parser

########################################################################################################