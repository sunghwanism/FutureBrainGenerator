import os

import torch
from torch.utils.data import Dataset

import pandas as pd
import nibabel as nib


# ['Dataset', 'SubID' , 'Age_B', 'MMSE_B', 'CDR_B','Control_B', 'impath_B', 
# 'Age_F','MMSE_F','CDR_F','Control_F','impath_F', 'Sex', 'Education', 'Interval']

class LongitudinalDataset(Dataset):
    def __init__(self, config, _type='train', Transform=None):
        self.imgpath = os.path.join(config.data_path, 'down_img_1.7mm')
        self.config = config
        df = pd.read_csv(os.path.join(config.data_path, 'long_old_HC_subj_phenotype_splited.csv'))
        self.subj_info = df[df['mode']==_type].copy()
        self.subj_files_B = self.subj_info['File_name_B'].to_list()
        self.subj_files_F = self.subj_info['File_name_F'].to_list()
        self.subj_condition = self.subj_info.loc[:,config.condition] # Sex, Age, Interval
        
        self.Transform = Transform
        
    def __len__(self):
        return len(self.subj_info)
    
    def __getitem__(self, idx):
        # Load the NIfTI file
        use_subj = self.phenotpye.iloc[idx]
        base_img = nib.load(use_subj['impath_B'])
        follow_img = nib.load(use_subj['impath_F'])
        base_data = base_img.get_fdata()
        follow_data = follow_img.get_fdata()

        condition = use_subj[self.condition].values
        tg_age = use_subj[self.tg_age].values

        # Convert the numpy array to a PyTorch tensor
        base_data = torch.from_numpy(base_data).float()
        base_data = base_data.unsqueeze(0)

        follow_data = torch.from_numpy(follow_data).float()
        follow_data = follow_data.unsqueeze(0)
        
        # Get the original dimensions (assumed to be 3D data)
        d, h, w = base_data.shape[1:]  # Shape without the channel

        # Define the target crop size
        target_d, target_h, target_w = self.config.crop_size

        # Calculate the start and end indices for cropping (crop from the center)
        start_d = (d - target_d) // 2
        start_h = (h - target_h) // 2
        start_w = (w - target_w) // 2

        # Perform the cropping
        base_data = base_data[:, start_d:start_d + target_d, start_h:start_h + target_h, start_w:start_w + target_w]
        follow_data = follow_data[:, start_d:start_d + target_d, start_h:start_h + target_h, start_w:start_w + target_w]


        # Phenotype
        condition = torch.tensor(condition).float()
        condition = condition.unsqueeze(0)

        result = {'base_img': base_data, 'follow_img': follow_data, 
                  'condition': condition}

        return result