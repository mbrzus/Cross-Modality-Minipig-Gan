import glob
import os
import shutil
import time
import itk

from pathlib import Path
import matplotlib.pyplot as plt
import pytorch_lightning as pl
# from pytorch_lightning import LightningModule 
import torchmetrics
import torch
from joblib import Parallel, delayed
import json
import numpy as np
from itertools import product as cartesian_product
from monai.data import CacheDataset, NiftiSaver
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
    ThresholdIntensityd,
    Lambdad
)
from transforms import *
from transforms2 import *
from GAN_old_discriminator_no_patches import GAN
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


if __name__ == "__main__":

    path_to_data = "/Shared/sinapse/aml/inferrence_rescaled"
    t1_gt = []
    t2_gen = []
    t2_gt = []

    for i in Path(path_to_data).glob('*'):
        if "truth" in str(i):
            t2_gt.append(str(i))
        elif "inferred" in str(i):
            t2_gen.append(str(i))
        else:
            t1_gt.append(str(i))

    test_files = [
        {"t1_gt": t1, "t2_gt": t2, "t2_gen": t2_gan}
        for t1, t2, t2_gan in zip(t1_gt, t2_gt, t2_gen)
    ]

    transforms = Compose(
        [
            LoadITKImaged(keys=["t1_gt", "t2_gt", "t2_gen"]),
            ITKImageToNumpyd(keys=["t1_gt", "t2_gt", "t2_gen"]),
        ]
    )

    test_dataset = CacheDataset(data=test_files, transform=transforms, cache_rate=1.0, num_workers=4)

    psnr_t1_total = 0
    psnr_t2_gen_total = 0
    ssim_t1_total = 0
    ssim_t2_gen_total = 0
    n=len(test_files)
    print(n)
    for i in range(n):
        item = test_dataset.__getitem__(i)  # extract image and label from loaded dataset
        t1 = item['t1_gt']
        t2 = item['t2_gt']
        t2_gen = item['t2_gen']

        psnr_t1 = peak_signal_noise_ratio(t2, t1, data_range=256)
        psnr_t2_gen = peak_signal_noise_ratio(t2, t2_gen, data_range=256)
        psnr_t1_total += psnr_t1
        psnr_t2_gen_total += psnr_t2_gen

        ssim_t1 = structural_similarity(t2, t1, data_range=256)
        ssim_t2_gen = structural_similarity(t2, t2_gen, data_range=256)
        ssim_t1_total += ssim_t1
        ssim_t2_gen_total += ssim_t2_gen

    psnr_t1_ave = psnr_t1_total / n
    psnr_t2_gen_ave = psnr_t2_gen_total / n
    ssim_t1_ave = ssim_t1_total / n
    ssim_t2_gen_ave = ssim_t2_gen_total / n

    print(f"Average PSNR t2 vs t1: {psnr_t1_ave}")
    print(f"Average PSNR t2 vs t2 gen: {psnr_t2_gen_ave}")
    print(f"Average SSIM t2 vs t1: {ssim_t1_ave}")
    print(f"Average SSIM t2 vs t2 gen: {ssim_t2_gen_ave}")

