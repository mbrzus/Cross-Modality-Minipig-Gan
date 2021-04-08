import glob
import json
import os
import shutil
import time
from argparse import ArgumentParser
from collections import OrderedDict
from itertools import product as cartesian_product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
# import pytorch_lightning as pl
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

class CasNetGenerator(nn.Module):
    # source: https://arxiv.org/pdf/1806.06397.pdf
    def __init__(
        self, img_shape, n_unet_blocks=6 # The MEDGAN paper had the best results with 6 unet blocks
    ):  # TODO: change num u_net blocks for actual trraining
        super().__init__()
        self.img_shape = img_shape

        def unet_block(
            in_channels,
            out_channels,
            channels=(64, 128, 256, 512, 512, 512, 512),#, 512),
            strides=(2, 2, 2, 2, 2, 2, 2),#, 2),
        ):
            return UNet(
                dimensions=3,
                in_channels=in_channels,
                out_channels=out_channels,
                channels=channels,
                strides=strides,
                num_res_units=2,
                norm=Norm.BATCH,
            )

        u_net_list = [unet_block(1, 1) for _ in range(n_unet_blocks)]
        u_net_list.append(nn.Tanh())

        self.model = nn.Sequential(*u_net_list)

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    arr = np.ones((128, 128, 128))
    arr = torch.from_numpy(arr)
    arr = arr.unsqueeze(0).unsqueeze(0).cuda().type(torch.cuda.FloatTensor)

    model = CasNetGenerator((128, 128, 128))
    model.cuda()
    print(model)
    print(model.forward(arr))
    print(model.forward(arr).shape)
    #print(model.parameters())
