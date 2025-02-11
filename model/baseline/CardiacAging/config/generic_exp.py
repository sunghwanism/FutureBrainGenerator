'''Configuration template for a generic experiment'''
LATENT_DIM = 130

EXPERIMENT_PARAMS = {
    # 'augment': True,  # Use data augmentation
    'batch_size': 2,
    'beta1': 0.9, # first moment decay
    'beta2': 0.999, # second moment decay
    'cycle_cons_weight': 1, # Cycle consistency weight in paper
    'epochs': 500,
    'gradient_penalty_weight': 10,
    'initial_filters': 32,
    'input_shape': (96,112,96), # Modified by HG (original : (256,256))
    'kernel_size': (3, 3, 3),
    'latent_space_dim': LATENT_DIM,
    'learning_rate': 1e-4,  # Learning rate
    'module_name': 'GAN',
    'ncritic': 5, # Number of training critic (Discriminator updates ncritic*10 times per Generator update)
    'num_classes': 1, # used in deprecated TransformerXia
    'n_channels': 1,
    'optD': 'adamw', # Optimizer for Discriminator
    'optG': 'adamw', # Optimizer for Generator
    'project_name': 'agesynthesis',
    'radial_prior_weight': 0, # not used
    'reduced': True, # use GeneratorXiaReduced and DiscriminatorXiaReduced
    'reconstruction_weight': 10, # not used
    'regularization_weight': 1, # weight of |Real - Synthetic images|
    'regressor_weight': 0, # not used
    'subcat': '',
    'task_type': 'generative',
    'use_tanh': False, # when generating images : not used in paper
    'use_wclip': False, # not used
    'view': '',
    'nonlinearity': 'relu', # new added by HK
    'warming_epochs': 20,
    'weight_decay': 1e-4,  # Weight decay rate
    'discr_params': {
        'activation': 'relu',
        'latent_space': LATENT_DIM,
        'depth': 5,
        'encoding': 'both',
        'input_shape': (96,112,96),
        'model': 'DiscriminatorXia',
        'name': 'discriminator',
        'norm': 'batchnorm',
        'initial_filters': 32,
    },
    'gen_params': {
        'activation': 'relu',
        'latent_space': LATENT_DIM,
        'depth': 4,
        'encoding': 'both',
        'input_shape': (96,112,96),
        'model': 'GeneratorXia',
        'name': 'generator',
        'norm': 'layernorm',
        'initial_filters': 32,
        'G_activation': 'linear'
    }
}


def get():
    'Return experiment settings.'
    return EXPERIMENT_PARAMS
