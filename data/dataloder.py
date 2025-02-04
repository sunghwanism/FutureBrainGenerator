import os

import torch
from torch.utils.data import Dataset

import nibabel as nib


class MRIDataset(Dataset):
    def __init__(self, imgpath, config, _type='train'): # train, val
        self.imgpath = os.path.join(imgpath, 'img_split', _type)
        self.file_paths = os.listdir(self.imgpath)
        self.config = config
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        # Load the NIfTI file
        nifti_img = nib.load(os.path.join(self.imgpath, self.file_paths[idx]))
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

        return img_data