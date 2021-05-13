import os
from pathlib import Path
from transforms import *
from transforms2 import *
import glob
import os
import shutil
import time
import itk
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import json

import numpy as np
from itertools import product as cartesian_product
from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import CacheDataset, NiftiSaver
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import compute_meandice, compute_hausdorff_distance
from monai.networks.layers import Norm
from monai.networks.nets import UNet
from monai.transforms import (
    AddChanneld,
    Compose,
    ScaleIntensityd,
    ScaleIntensityRangePercentilesd,
    Resized,
    SaveImaged,
    Affined,
    CropForegroundd,
    Lambdad,
    LoadImaged,
    SpatialPadd,
    Orientationd,
    RandCropByPosNegLabeld,
    ResizeWithPadOrCropd,
    SpatialCropd,
    ScaleIntensityRanged,
    RandRotated,
    RandGaussianNoised,
    Rand3DElasticd,
    Spacingd,
    ToTensord,
    ThresholdIntensityd
)

# Read data from the json files - json files store paths to images and labels
meta_dir = str(Path(".").absolute().parent / "metadata")
with open(f"{meta_dir}/structure.json", "r") as openfile:
    structure = json.load(openfile)

train_structure = structure["train"]
# validation_structure = structure["validation"]
test_structure = structure["test"]
# zip two arrays into a dictionary form expected by MONAI


def structure_to_monai_dict(structure_dict):
    output_list_of_dicts = []
    for subject_id in structure_dict.keys():
        for session_id in structure_dict[subject_id].keys():
            if (
                    len(structure_dict[subject_id][session_id]["t1w"]) > 0
                    and len(structure_dict[subject_id][session_id]["t2w"]) > 0
            ):
                # TODO: add structure here to filter by age, gender, site, scanner, T-value, etc..
                [
                    output_list_of_dicts.append(
                        {"t1w": t1_file, "t2w": t2_file}
                    )
                    for t1_file, t2_file in cartesian_product(
                    structure_dict[subject_id][session_id]["t1w"],
                    structure_dict[subject_id][session_id]["t2w"],
                )
                ]

    return output_list_of_dicts


test_files = structure_to_monai_dict(test_structure)
test_files = test_files[:1]


# path to directory where the tested images will be stored
#path_to_write = "/home/mbrzus/programming/masterthesis/code/test/image_transform_pipeline_test"
path_to_write = "/home/mbrzus/Desktop"

transforms = Compose(
    [
        LoadITKImaged(keys=["t1w", "t2w"]),
        ResampleT1T2d(keys=["t1w", "t2w"], output_size=[128, 128, 128]),
        ITKImageToNumpyd(keys=["t1w", "t2w"]),
        ScaleIntensityRangePercentilesd(
            keys=["t1w", "t2w"],
            lower=1.0,
            upper=99.0,
            b_min=-1.0,
            b_max=1.0,
            clip=True,
            relative=False,
        ),
        AddChanneld(keys=["t1w", "t2w"]),
        ToTensord(keys=["t1w", "t2w"]),
    ]
)

# MONAI Cache dataset function uses cache to efficiently load the images with transformations
train_dataset = CacheDataset(data=test_files, transform=transforms, cache_rate=1.0, num_workers=4)

# Loop to accessed images after transformations
for i in range(1):
    item = train_dataset.__getitem__(i)  # extract image and label from loaded dataset
    t1w_np = item['t1w'].squeeze(dim=0).numpy()
    t2w_np = item['t2w'].squeeze(dim=0).numpy()
    print(t1w_np.shape)
    item['t1w'] = t1w_np
    item['t2w'] = t2w_np

    out_transforms = Compose([
        ToITKImaged(keys=["t1w", "t2w"]),
        SaveITKImaged(keys=["t1w", "t2w"], out_dir=path_to_write, output_postfix="test")
    ])
    out_transforms(item)
