import os

import torch
from torch.utils.data import Dataset

import nibabel as nib
import pandas as pd


class CrossMRIDataset(Dataset):
    def __init__(self, config, _type='train', Transform=None): # train, val
        self.imgpath = os.path.join(config.data_path, 'down_img_1.7mm')
        self.config = config
        df = pd.read_csv(os.path.join(config.data_path, 'cross_old_subj_phenotype_splited_v3.csv'))
        self.subj_info = df[df['mode']==_type].copy()
        self.subj_files = self.subj_info['File name'].to_list()
        self.Transform = Transform
        
        del df
    
    def __len__(self):
        return len(self.subj_info)
    
    def __getitem__(self, idx):
        # Load the NIfTI file
        nifti_img = nib.load(os.path.join(self.imgpath, self.subj_files[idx]))
        img_data = nifti_img.get_fdata()
        
        # Convert the numpy array to a PyTorch tensor
        img_data = torch.from_numpy(img_data).float()
        img_data = img_data.unsqueeze(0)
        
        # Get the original dimensions (assumed to be 3D data)
        d, h, w = img_data.shape[1:]  # Shape without the channel

        # Define the target crop size
        target_d, target_h, target_w = self.config.crop_size

        # Calculate the start and end indices for cropping (crop from the center)
        start_d = (d - target_d) // 2
        start_h = (h - target_h) // 2
        start_w = (w - target_w) // 2

        # Perform the cropping
        img_data = img_data[:, start_d:start_d + target_d, start_h:start_h + target_h, start_w:start_w + target_w]
        
        if self.Transform is not None:
            img_data = self.Transform(img_data)

        return img_data