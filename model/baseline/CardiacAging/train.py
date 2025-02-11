'''
Train script for pytorch model
'''
## monai.utilis.misc.py -> line 52 : "MAX_SEED = np.iinfo(np.uint32).max +1" to "MAX_SEED = np.iinfo(np.uint32).max - 1"
import os
import glob
import logging
import importlib
import wandb

import numpy as np
import pandas as pd

import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, Callback

from monai.data import ImageDataset, ZipDataset, Dataset
from data.paired_data import AlignedPairedData
from data.loader import load_dataset, load_segmentation_dset
from data.cardiacDataset import cardiacDataset
from config.options import parse_options
from models.gan import GAN

from monai.transforms import Transform, Transpose
# from models.regressor import Regressor
# from models.classifier import Classifier
# from models.segmentor import Segmentor

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
log = logging.getLogger("proposed_executor")
wd = os.path.dirname(os.path.realpath(__file__))


class InitCallback(Callback):
    def on_fit_start(self, trainer, pl_module):
        print("Sending model weights to GPU!")
        pl_module.regressor = [rgr.to(pl_module.device) for rgr in pl_module.regressor]

class RowExtractor(Transform):
    def __init__(self, data):
        self.data = data

    def __call__(self, index):
        return self.data.iloc[index].to_dict()

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import nibabel as nib
import numpy as np

class MRI3DTransform:
    """3D MRI 이미지 크기 조정 및 axial slice 변환"""
    def __init__(self, target_size=(113, 113), depth=4):
        self.target_size = target_size  # (H, W) 크기 조정
        self.depth = depth  # axial slice depth 유지

    def __call__(self, img):
        """
        img: (D, H, W) 형태의 3D MRI 이미지
        반환: (1, depth, target_size[0], target_size[1]) 크기로 변환된 이미지
        """
        if isinstance(img, nib.Nifti1Image):  # NiFTI 포맷 처리
            img = img.get_fdata()  # numpy 배열로 변환
        
        img = np.array(img, dtype=np.float32)  # NumPy 배열로 변환

        # 🔹 원본 이미지 shape (D, H, W) 출력
        print(f"[INFO] Original Image Shape: {img.shape}")

        D, H, W = img.shape

        # 중앙에서 depth=4 만큼 axial slice 추출
        mid_slice = D // 2
        start_slice = max(mid_slice - self.depth // 2, 0)
        end_slice = min(mid_slice + self.depth // 2, D)

        img = img[start_slice:end_slice, :, :]  # (depth, H, W)

        # 🔹 변환 후 shape 출력
        print(f"[INFO] After Axial Slicing Shape: {img.shape}")

        # 크기 조정 (depth 유지, HxW 변경)
        img = torch.tensor(img).unsqueeze(0)  # (1, depth, H, W)로 변환

        # 🔹 Tensor 변환 확인
        print(f"[INFO] After ToTensor Shape: {img.shape}")

        img = F.interpolate(img, size=self.target_size, mode='bilinear', align_corners=False)  # 크기 조정

        # 🔹 최종 변환된 이미지 shape 확인
        print(f"[INFO] Final Resized Image Shape: {img.shape}")

        return img  # (1, depth, target_size[0], target_size[1])

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



def main():
    '''Main function.'''
    torch.cuda.empty_cache()
    args, settings = parse_options()
    settings['working_dir'] = wd
    if settings['task_type'] != 'segmentation':
        # data_loader = load_dataset 
        data_loader = load_dataset
    else:
        data_loader = load_segmentation_dset

    # GPUs to use
    gpus = [0]
    if args.gpu is not None:
        gpus = np.asarray(args.gpu.split(',')).astype(int)
    # Folder for model checkpoints
    ckpt_folder = os.path.join(args.outf, args.name)
    os.makedirs(ckpt_folder, exist_ok=True)

    df = pd.read_csv("/NFS/FutureBrainGen/data/long/long_old_HC_subj_phenotype_splited.csv")
    df['File_name_B'] = df['File_name_B'] + '.gz'
    df['File_name_F'] = df['File_name_F'] + '.gz'
    train_df = df[df['mode']=='train']
    val_df = df[df['mode']=='val']
    test_df = df[df['mode']=='test']
    
    train_files = [os.path.join(args.dataf, f) for f in train_df['File_name_B']]
    train_labels = train_df['Age_B'].to_numpy()
    train_data = train_df[['Sex']]
    train_data.reset_index(inplace=True, drop=True)

    train_files2 = [os.path.join(args.dataf, f) for f in train_df['File_name_F']]
    train_labels2 = train_df['Age_F'].to_numpy()
    train_data2 = train_df[['Sex']]
    train_data2.reset_index(inplace=True, drop=True)

    val_files = [os.path.join(args.dataf, f) for f in val_df['File_name_B']]
    val_labels = val_df['Age_B'].to_numpy()
    val_data = val_df[['Sex']]
    val_data.reset_index(inplace=True, drop=True)

    val_files2 = [os.path.join(args.dataf, f) for f in val_df['File_name_F']]
    val_labels2 = val_df['Age_F'].to_numpy()
    val_data2 = val_df[['Sex']]
    val_data2.reset_index(inplace=True, drop=True)

    # transform = MRI3DTransform(target_size=(137, 113), depth=4)
    transform = CropTransform(crop_size=args.crop_size)

    train_dset = ImageDataset(train_files, transform=transform, labels=train_labels, reader='NibabelReader')
    train_data_dset = Dataset(data=list(range(len(train_data))), transform=RowExtractor(train_data))
    train_dset = ZipDataset([train_dset, train_data_dset])

    train_dset2 = ImageDataset(train_files2, transform=transform, labels=train_labels2, reader='NibabelReader')
    train_data_dset2 = Dataset(data=list(range(len(train_data2))), transform=RowExtractor(train_data2))
    train_dset2 = ZipDataset([train_dset2, train_data_dset2])

    val_dset = ImageDataset(val_files, transform=transform, labels=val_labels, reader='NibabelReader')
    val_data_dset = Dataset(data=list(range(len(val_data))), transform=RowExtractor(val_data))
    val_dset = ZipDataset([val_dset, val_data_dset])

    val_dset2 = ImageDataset(val_files2, transform=transform, labels=val_labels2, reader='NibabelReader')
    val_data_dset2 = Dataset(data=list(range(len(val_data2))), transform=RowExtractor(val_data2))
    val_dset2 = ZipDataset([val_dset2, val_data_dset2])

    train_loader = torch.utils.data.DataLoader(train_dset,
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers)

    for batch in train_loader:
            print(type(batch))  # Likely a tuple
            print(batch[0].shape)  # Image tensor shape
            print(batch[1])  # Age label
            print(batch[2])  # Tabular data
            break

    train_loader2 = torch.utils.data.DataLoader(train_dset2,
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers)

    val_loader = torch.utils.data.DataLoader(val_dset,
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers)

    val_loader2 = torch.utils.data.DataLoader(val_dset2,
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers)

    if args.train:
        print('-> Training model', args.name)
        # Data loading code
        # train_loaders, val_loaders = data_loader(args, settings)
        
        train_loader = AlignedPairedData(train_loader, train_loader2, False)
        val_loader = AlignedPairedData(val_loader, val_loader2, False)

        checkpoint_callback = ModelCheckpoint(
            monitor='train_loss',
            every_n_epochs=50, # Save ckpt every 20 epochs
            dirpath=ckpt_folder,
            filename='model-{epoch:03d}-{train_loss:.2f}',
            save_top_k=10,
            save_last=True,
            mode='min',
        )
        callbacks = [checkpoint_callback]

        # Initialize model: Load model and call init() function
        settings['num_training_samples'] = len(train_loader)
        settings['num_validation_samples'] = len(val_loader)
        settings['dataset_file'] = args.dataset
        settings['data_type'] = args.data_type
        init_module = importlib.import_module(
            'models.{}'.format(settings['module_name'].lower()))
        model, clbk = init_module.init(args, settings)
        if clbk:
            callbacks.append(clbk)

        # Train with Pytorch Lightning
        pl.seed_everything(42, workers=True)

        tb_logger = pl_loggers.TensorBoardLogger(os.path.join(wd, 'logs'), name=args.name)

        # wandb logger
        if args.wandb:
            wandb_logger = WandbLogger(name=args.name, project='CardiacAging')
            wandb_logger.log_hyperparams(settings)
            wandb_logger.watch(model, log='all')
            loggers = [wandb_logger]
        else:
            loggers = []

        # Drop comet logger when testing
        loggers.append(tb_logger)
        if args.iters is not None:
            if args.iters <= 10:
                loggers.append(tb_logger)

        if args.precision == 16:
            precision = 16
        else:
            precision = 32

        trainer = pl.Trainer(
            precision=precision, # 16-bit precision
            deterministic=True,
            default_root_dir=args.outf,
            max_epochs=settings['epochs'],
            gpus=[*gpus], # GPU id to use (can be [1,3] [ids 1 and 3] or [-1] [all])
            max_steps=args.iters, # default is None (not limited)
            logger=loggers,
            accumulate_grad_batches=settings['accumulated_grad_batches'],
            callbacks=callbacks)

        # Provide confirmation message of settings defined for debugging
        print('Training model for {} epochs with dataset {}.'.format(
            settings['epochs'], args.experiment))

        trainer.fit(model, train_loader, val_loader)

    if args.test:
        print('Testing model {} with data from {}'.format(
            args.name, args.dataf))

        # Find checkpoint and hparams file
        if os.path.exists(os.path.join(ckpt_folder, 'last.ckpt')):
            ckpt_model = os.path.join(ckpt_folder, 'last.ckpt')
        else:
            ckpt_name = '*-epoch*.ckpt'
            if args.epochs > 0:
                ckpt_name = '*-epoch={0:03d}*.ckpt'.format(args.epochs)
            print('Looking for model in "{}" with name "{}"'.format(ckpt_folder, ckpt_name))
            ckpt_model = list(sorted(glob.iglob(
                os.path.join(ckpt_folder, ckpt_name)
            )))[-1]
        print('Found model "{}"'.format(ckpt_model))
        hpms_file = list(glob.iglob(
            os.path.join(wd, 'logs', args.name, '*', 'hparams.yaml')))[-1]
            # os.path.join(wd, 'logs', 'default', '*', 'hparams.yaml')))[-1]
        print('hparams file "{}"'.format(hpms_file))

        if settings['task_type'] == 'regression':
            model = Regressor.load_from_checkpoint(
                checkpoint_path=ckpt_model,
                hparams_file=hpms_file,
                map_location=None,
                results_folder=settings['results_folder'])
        elif settings['task_type'] == 'classification':
            model = Classifier.load_from_checkpoint(
                checkpoint_path=ckpt_model,
                hparams_file=hpms_file,
                map_location=None)
        elif settings['task_type'] == 'segmentation':
            model = Segmentor.load_from_checkpoint(
                checkpoint_path=ckpt_model,
                hparams_file=hpms_file,
                map_location=None,
                results_folder=settings['results_folder'])
        elif settings['task_type'] == 'generative':
            model = GAN.load_from_checkpoint(
                checkpoint_path=ckpt_model,
                hparams_file=hpms_file,
                map_location=None,
                results_folder=settings['results_folder'])

        # Data loading code
        test_files = [os.path.join(args.dataf, f) for f in test_df['File_name_B']]
        test_labels = test_df['Age_B'].to_numpy()
        test_data = test_df[['Sex']]
        test_data.reset_index(inplace=True, drop=True)

        test_files2 = [os.path.join(args.dataf, f) for f in test_df['File_name_F']]
        test_labels2 = test_df['Age_F'].to_numpy()
        test_data2 = test_df[['Sex']]
        test_data2.reset_index(inplace=True, drop=True)

        transform = CropTransform(crop_size=args.crop_size)

        test_dset = ImageDataset(test_files, transform=transform, labels=test_labels, reader='NibabelReader')
        test_data_dset = Dataset(data=list(range(len(test_data))), transform=RowExtractor(test_data))
        test_dset = ZipDataset([test_dset, test_data_dset])

        test_dset2 = ImageDataset(test_files2, transform=transform, labels=test_labels2, reader='NibabelReader')
        test_data_dset2 = Dataset(data=list(range(len(test_data2))), transform=RowExtractor(test_data2))
        test_dset2 = ZipDataset([test_dset2, test_data_dset2])

        test_loader = torch.utils.data.DataLoader(test_dset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers)
        test_loader2 = torch.utils.data.DataLoader(test_dset2, batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers)
        
        test_loaders = AlignedPairedData(test_loader, test_loader2, False)

        # test_loaders, _ = data_loader(args, settings, split_data=False)

        # Test model
        os.makedirs(
            os.path.join(settings['results_folder'], settings['experiment_name']),
            exist_ok=True)
        trainer = pl.Trainer(gpus=[*gpus])
        trainer.test(model, test_loaders)



if __name__ == '__main__':
    main()
