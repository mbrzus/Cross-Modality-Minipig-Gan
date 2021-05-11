import glob
import os
import shutil
import time
import itk

from pathlib import Path
import matplotlib.pyplot as plt
import pytorch_lightning as pl
# from pytorch_lightning import LightningModule 
import torch
from joblib import Parallel, delayed
import json
import numpy as np
from itertools import product as cartesian_product
from monai.data import CacheDataset, NiftiSaver
from monai.inferers import sliding_window_inference, SimpleInferer, SlidingWindowInferer
from monai.losses import DiceLoss
from monai.metrics import compute_meandice, DiceMetric, compute_hausdorff_distance, HausdorffDistanceMetric
from monai.networks.layers import Norm
from monai.networks.nets import UNet
from monai.networks.utils import one_hot
from monai.transforms import (
    Activations,
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    SpatialCropd,
    Orientationd,
    SpatialPadd,
    RandCropByPosNegLabeld,
    # SaveImaged,
    ScaleIntensityRangePercentilesd,
    ScaleIntensityRanged,
    ResizeWithPadOrCropd,
    RandRotated,
    RandGaussianNoised,
    Spacingd,
    ToNumpyd,
    ToTensord,
    ThresholdIntensityd
)
from transforms import *
from transforms2 import *
from GAN_old_discriminator_no_patches import GAN


if __name__ == "__main__":

    checkpoints_dir = "/Shared/sinapse/aml/old_discriminator_log"
    inferrence_dir = "/home/mbrzus/Desktop"

    # define model and load its parameters


    model = GAN.load_from_checkpoint(
        channels=1,
        width=128,
        height=128,
        depth=128,
        checkpoint_path=f"{checkpoints_dir}/gen_recon_epoch=24-g_loss=100.03-g_recon_loss=0.03-d_loss=45.00.ckpt",
        hparams_file=f"{checkpoints_dir}/default/version_13/hparams.yaml",
        img_shape=(128, 128, 128),
        strict=False
    )
    device = torch.device("cuda:0")
    model.to(device)
    model.eval()
    model.freeze()

    # prepare test data
    # dic = {"t1w": "/home/mbrzus/programming/gitlab/MiniPigMALF/BIDS/sub-Ada165/ses-20130321/anat/sub-Ada165_ses-20130321_run-094402_T1w.nii.gz"}
    dic = {"t1w": "/home/mbrzus/programming/gitlab/MiniPigMALF/BIDS/derivatives/fusionLabel/sub-Ada165/ses-20130321/anat/sub-Ada165_ses-20130321_run-094402_T1w.nii.gz"}

    # define transforms for the data
    transforms = Compose(
        [
            LoadITKImaged(keys=["t1w"]),
            MiniPigResampled(keys=["t1w"], output_size=(128, 128, 128)),
            ITKImageToNumpyd(keys=["t1w"]),
            ScaleIntensityRangePercentilesd(
                keys=["t1w"],
                lower=1.0,
                upper=99.0,
                b_min=-1.0,
                b_max=1.0,
                clip=True,
                relative=False,
            ),
            AddChanneld(keys=["t1w"]),
            ToTensord(keys=["t1w"]),
        ]
    )

    # create a test dataset with the preprocessed images
    #test_dataset = CacheDataset(data=dic, transform=transforms, cache_rate=1.0, num_workers=4)
    test_dataset = transforms(dic)
    # Loop to accessed images after transformations
    for i in range(1): #range(len(test_T2s)):
        print(test_dataset.keys())
        item = test_dataset#.__getitem__(i)  # extract image and label from loaded dataset
        print(item.keys())
        print(item['t1w'].size())
        print(item['t1w'].unsqueeze(dim=0).size())

        # perform the inference
        with torch.no_grad():
            # roi_size = (128, 128, 128)
            # sw_batch_size = 12
            # test_output = sliding_window_inference(
            #     item['t1w'].unsqueeze(dim=0).to(device), roi_size, sw_batch_size, model
            # )
            test_output = model.generator.forward(item['t1w'].unsqueeze(dim=0).to(device))

        item['t2w_generated'] = test_output.detach().cpu()
        item['t2w_generated_meta_dict'] = item['t1w_meta_dict']
        item['t2w_generated_meta_dict']['filename'] = "t2_inferred.nii.gz"
        #item['t2w_meta_dict']['filename'] = "t2_truth.nii.gz"

        out_transforms = Compose([
            ToNumpyd(keys=["t2w_generated"]),
            ToITKImaged(keys=["t2w_generated"]),
            SaveITKImaged(keys=["t2w_generated"], out_dir=inferrence_dir, output_postfix="crop")
        ])

        out_transforms(item)

