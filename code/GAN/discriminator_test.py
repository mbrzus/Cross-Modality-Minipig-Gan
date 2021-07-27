import glob
import json
import os
import shutil
import time
from argparse import ArgumentParser
from collections import OrderedDict
from itertools import product as cartesian_product
from pathlib import Path

import apex
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import CacheDataset, list_data_collate
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import compute_hausdorff_distance, compute_meandice
from monai.networks.layers import Norm
from monai.networks.nets import UNet
from monai.networks.nets import Discriminator as MONAIDiscriminator
from monai.transforms import (
    AddChanneld,
    Compose,
    LoadImaged,
    Orientationd,
    RandGaussianNoised,
    RandSpatialCropSamplesd,
    RandRotated,
    ResizeWithPadOrCropd,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
    ToTensord,
    ScaleIntensityRangePercentilesd,
    Resized,
)
from monai.utils import set_determinism
from torch.utils.data import DataLoader, random_split
from monai.visualize.img2tensorboard import plot_2d_or_3d_image


class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super().__init__()

        kernel = (3, 3, 3)
        stride = (1, 1, 1)
        self.model_conv = nn.Sequential(
            # Block 1
            nn.Conv3d(
                in_channels=1, out_channels=64, kernel_size=kernel, stride=stride
            ),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # Block 2
            nn.Conv3d(
                in_channels=64, out_channels=128, kernel_size=kernel, stride=stride
            ),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # Block 3
            nn.Conv3d(
                in_channels=128,
                out_channels=256,
                kernel_size=(4, 4, 4),
                stride=(2, 2, 2),
            ),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # Block 4
            nn.Conv3d(
                in_channels=256,
                out_channels=256,
                kernel_size=(4, 4, 4),
                stride=(2, 2, 2),
            ),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.model_linear = nn.Sequential(
            # Sigmoid
            nn.Flatten(),
            nn.Linear(256 * 24 * 24 * 24, 32),
            nn.Sigmoid(),
        )

    def forward(self, img):
        print("Discriminator forward")
        out = self.model_conv(img)
        print(out.shape)
        validity = self.model_linear(out)
        return validity


if __name__ == "__main__":
    arr = np.ones((128, 128, 128))
    arr = torch.from_numpy(arr)
    arr = arr.unsqueeze(0).unsqueeze(0).cuda().type(torch.cuda.FloatTensor)

    model = Discriminator((128, 128, 128))
    model.cuda()
    print(model)
    print(model.forward(arr))
    print(model.forward(arr).shape)
    # print(model.parameters())


### Testing patching for use in the discriminator ###
# arr1 = np.ones((1, 1, 128, 128, 128))
# arr1 = torch.from_numpy(arr1)
# arr2 = np.ones((1, 1, 128, 128, 128))*0.5
# arr2 = torch.from_numpy(arr2)
#
# batch_data = [
#     {"t1w": t1, "t2w": t2}
#     for t1, t2 in zip(arr1, arr2)
# ]
# # print(batch_data)
# # print(batch_data["t1w"])
# # print(type(batch_data["t1w"]))
# transforms = Compose([
#     RandSpatialCropSamplesd(keys=["t1w", "t2w"],
#                             roi_size=(32, 32, 32),
#                             num_samples=4,
#                             random_size=False)
# ])
# patch_data = transforms(batch_data)
# print(type(patch_data))
# print(len(patch_data))
# print(type(patch_data[0]))
# print(len(patch_data[0]))
# print(type(patch_data[0][0]))
# print(patch_data[0][0].keys())
#
# # print(patch_data[0][0].keys())
# print(type(patch_data[0][0]["t1w"]))
# print(patch_data[0][0]["t1w"].size())
# print(patch_data[0][1]["t1w"].size())
# print(patch_data[0][2]["t1w"].size())
# print(patch_data[0][3]["t1w"].size())
#
# print(patch_data[0][0]["t2w"].size())
# print(patch_data[0][1]["t2w"].size())
# print(patch_data[0][2]["t2w"].size())
# print(patch_data[0][3]["t2w"].size())
#
# model = CasNetGenerator((128, 128, 128))
# model.cuda()
# print(model)
# print(model.forward(arr))
# print(model.forward(arr).shape)
# #print(model.parameters())
