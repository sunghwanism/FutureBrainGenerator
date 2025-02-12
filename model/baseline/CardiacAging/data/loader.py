'''
Utilities to load datasets.
'''
import os
import numpy as np

import torch
import pandas as pd

from monai.data import ImageDataset, ZipDataset, Dataset
from monai.transforms import AddChannel, Compose, Resize, \
    ScaleIntensity, EnsureType, RandBiasField, RandAdjustContrast, \
    RandHistogramShift, Transpose
from monai.transforms import LoadImaged, AddChanneld, Resized, RandRotated, \
    ScaleIntensityd, EnsureTyped, RandBiasFieldd, RandAdjustContrastd, \
    RandHistogramShiftd, AsDiscreted, ToTensord, Rand2DElasticd, Transform

from data import ukbb_loader
from data.paired_data import AlignedPairedData 
from data.transforms import LoadPngImaged, LoadNumpyImaged, Slice2DImage, SliceImageMid, \
    SliceImaged, SliceImage, SliceImageZ, Slice2DImaged

from utils.tensorbacked_array import TensorBackedImmutableStringArray, TensorBackedImmutableNestedArray

import sys
import warnings

# MONAI is surprisingly annoying
warnings.simplefilter(action='ignore', category=UserWarning)

class CropTransform:
    """
    3D MRI 이미지에서 중심을 기준으로 Crop 후 Channel=1을 추가하는 Transform
    """
    def __init__(self, crop_size=(96,112,96)):  # (Depth, Height, Width)
        self.crop_size = crop_size

    def __call__(self, image):
        """
        이미지에 Crop 적용 후 Channel 차원 추가
        """
        if isinstance(image, np.ndarray):
            image = torch.tensor(image, dtype=torch.float32)

        d, h, w = image.shape

        crop_d, crop_h, crop_w = self.crop_size
        start_d = max((d - crop_d) // 2, 0)
        start_h = max((h - crop_h) // 2, 0)
        start_w = max((w - crop_w) // 2, 0)

        image = image[start_d:start_d + crop_d, start_h:start_h + crop_h, start_w:start_w + crop_w]

        image = image.unsqueeze(0)  # Channel=1 추가

        return image

class RowExtractor(Transform):
    def __init__(self, data):
        self.data = data

    def __call__(self, index):
        return self.data.iloc[index].to_dict()

def load_dataset(args, settings, split_data=True):
    '''
    Prepare dataset for pytorch model
    with corresponding transformation functions.
    '''

    # --------------------------------------

    use_cols = ["File_name_B", "File_name_F", "Age_B", "Age_F", "Sex", "mode"]
    df = pd.read_csv("/NFS/FutureBrainGen/data/long/long_old_HC_subj_phenotype_splited.csv", usecols=use_cols)

    df['File_name_B'] = df['File_name_B'] + '.gz'
    df['File_name_F'] = df['File_name_F'] + '.gz'
    train_df = df[df['mode']=='train']
    val_df = df[df['mode']=='val']

    def get_file_paths(df, column):
        return [os.path.join(args.dataf, f) for f in df[column]]

    train_files = get_file_paths(train_df, 'File_name_B')
    train_files2 = get_file_paths(train_df, 'File_name_F')
    val_files = get_file_paths(val_df, 'File_name_B')
    val_files2 = get_file_paths(val_df, 'File_name_F')

    train_labels = train_df['Age_B'].to_numpy()
    train_labels2 = train_df['Age_F'].to_numpy()
    val_labels = val_df['Age_B'].to_numpy()
    val_labels2 = val_df['Age_F'].to_numpy()

    def get_label_dataset(df):
        return Dataset(data=list(range(len(df))), transform=RowExtractor(df[['Sex']].reset_index(drop=True)))

    train_data_dset = get_label_dataset(train_df)
    train_data_dset2 = get_label_dataset(train_df)
    val_data_dset = get_label_dataset(val_df)
    val_data_dset2 = get_label_dataset(val_df)

    transform = CropTransform(crop_size=args.crop_size)

    train_dset = ZipDataset([ImageDataset(train_files, transform=transform, labels=train_labels, reader='NibabelReader'), train_data_dset])

    train_dset2 = ZipDataset([ImageDataset(train_files2, transform=transform, labels=train_labels2, reader='NibabelReader'), train_data_dset2])
    val_dset = ZipDataset([ImageDataset(val_files, transform=transform, labels=val_labels, reader='NibabelReader'), val_data_dset])
    val_dset2 = ZipDataset([ImageDataset(val_files2, transform=transform, labels=val_labels2, reader='NibabelReader'), val_data_dset2])
    
    def get_dataloader(dataset):
        return torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True
        )

    train_loader = get_dataloader(train_dset)
    train_loader2 = get_dataloader(train_dset2)
    val_loader = get_dataloader(val_dset)
    val_loader2 = get_dataloader(val_dset2)

    train_loader = AlignedPairedData(train_loader, train_loader2, False)
    val_loader = AlignedPairedData(val_loader, val_loader2, False)

    return train_loader, val_loader
