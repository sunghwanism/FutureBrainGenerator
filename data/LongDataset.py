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
        self.Transform = Transform

        del df
        
    def get_subj_info(self, idx):
        return self.subj_info.iloc[idx]
        
    def __len__(self):
        return len(self.subj_info)
    
    def __getitem__(self, idx):
        
        # Load the NIfTI file
        base_img = nib.load(os.path.join(self.imgpath, self.subj_info['File_name_B'].iloc[idx]+".gz"))
        base_img = base_img.get_fdata()
        follow_img = nib.load(os.path.join(self.imgpath, self.subj_info['File_name_F'].iloc[idx]+".gz"))
        follow_img = follow_img.get_fdata()

        condition = self.subj_info.loc[idx, ['Age_B', 'Sex']].values.astype(float)
        interval = self.subj_info['Interval'].iloc[idx]
        
        # Convert the numpy array to a PyTorch tensor
        base_img = torch.from_numpy(base_img).unsqueeze(0).float()
        follow_img = torch.from_numpy(follow_img).unsqueeze(0).float()
        
        # Processing the image (Crop & Transform)
        base_img = crop_img(base_img, self.config.crop_size)
        follow_img = crop_img(follow_img, self.config.crop_size)
        
        if self.Transform is not None:
            follow_img = self.Transform(follow_img)
            base_img = self.Transform(base_img)
        
        # Phenotype
        condition = torch.from_numpy(condition)

        result = {'base_img': base_img,
                  'follow_img': follow_img, 
                  'condition': condition,
                  'interval': interval,
                  'Age_F': self.subj_info['Age_F'].iloc[idx],
                  'Age_B': self.subj_info['Age_B'].iloc[idx],
                  'Sex': self.subj_info['Sex'].iloc[idx]}

        return result
    