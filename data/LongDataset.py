import os

import torch
from torch.utils.data import Dataset

import pandas as pd
import nibabel as nib


def crop_img(img, crop_size):
    
    # Get the original dimensions (assumed to be 3D data)
    d, h, w = img.shape[1:]  # Shape without the channel

    # Define the target crop size
    target_d, target_h, target_w = crop_size

    # Calculate the start and end indices for cropping (crop from the center)
    start_d = (d - target_d) // 2
    start_h = (h - target_h) // 2
    start_w = (w - target_w) // 2
    
    return img[:, start_d:start_d + target_d, start_h:start_h + target_h, start_w:start_w + target_w]


class LongitudinalDataset(Dataset):
    def __init__(self, config, _type='train', Transform=None):
        self.imgpath = os.path.join(config.data_path, 'down_img_1.7mm')
        self.config = config
        df = pd.read_csv(os.path.join(config.data_path, 'long_old_HC_subj_phenotype_splited.csv'))
        self.subj_info = df[df['mode']==_type].copy()
        self.subj_files_B = self.subj_info['File_name_B'].to_list()
        self.subj_files_F = self.subj_info['File_name_F'].to_list()
        self.conditions = self.subj_info.loc[:, ['Age_B', 'Sex']].values
        
        self.interval = self.subj_info.loc[:, 'interval'].to_list()
        
        self.Transform = Transform
        
    def get_subj_info(self, idx):
        return self.subj_info.iloc[idx]
        
    def __len__(self):
        return len(self.subj_info)
    
    def __getitem__(self, idx):
        
        # Load the NIfTI file
        base_img = nib.load(self.subj_files_B[idx])
        base_img = base_img.get_fdata()
        follow_img = nib.load(self.subj_files_F[idx])
        follow_img = follow_img.get_fdata()

        condition = self.conditions[idx]
        interval = self.interval[idx]
        
        # Convert the numpy array to a PyTorch tensor
        base_img = torch.from_numpy(base_img).unsqueeze(0).float()
        follow_img = torch.from_numpy(follow_img).unsqueeze(0).float()
        
        # Processing the image (Crop & Transform)
        base_img = crop_img(base_img, self.config.crop_size)
        follow_img = crop_img(follow_img, self.config.crop_size)
        
        if self.Transform is not None:
            follow_img = self.Transfrom(follow_img)
            base_img = self.Transform(base_img)
        
        # Phenotype
        condition = torch.tensor(condition).float()

        result = {'base_img': base_img,
                  'follow_img': follow_img, 
                  'condition': condition,
                  'interval': interval}

        return result
    