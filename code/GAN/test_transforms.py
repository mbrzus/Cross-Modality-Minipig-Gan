import glob
import os
import shutil
import time
import itk
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import json
import apex
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
    CropForegroundd,
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
from monai.utils import set_determinism

with open("../metadata/structure.json", "r") as openfile:
    structure = json.load(openfile)

train_structure = structure["train"]
validation_structure = structure["validation"]
test_structure = structure["test"]

# if a session has at least one t1w image and at least one t2w image,
# add the cartesian product of the two lists
# this should maximally use the data by mapping all t1w to all t2w within a session
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


# organize files into pairs (t1w, t2w) for MONAI dictionary workflow
train_files = structure_to_monai_dict(train_structure)
val_files = structure_to_monai_dict(validation_structure)
test_files = structure_to_monai_dict(test_structure)

# get just a very small portion of the data for initial test (fail fast)
# TODO: look at splitting these for different training phases


train_files = train_files[:1]
# val_files = val_files[:5]
# test_files = test_files[:5]
# path to directory where the tested images will be stored
path_to_write = "/Shared/sinapse/mbrzus/transform_test"
saver = NiftiSaver(output_dir=path_to_write, output_postfix="pad",
                   dtype=None)

# MONAI transforms used for preprocessing image
# detailed description of the transforms and their behavior is in the preprocessing_transforms.pptx presentation
transforms = Compose(
    [
        LoadImaged(keys=["t1w", "t2w"]),
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
        Spacingd(keys=["t1w", "t2w"], pixdim=(1, 1, 1)),
        SpatialPadd(keys=["t1w", "t2w"], spatial_size=(300, 300, 300)),
        #ResizeWithPadOrCropd(keys=["t1w", "t2w"], spatial_size=(300, 300, 300)),
        ToTensord(keys=["t1w", "t2w"]),
    ]
)
# MONAI Cache dataset function uses cache to efficiently load the images with transformations
train_dataset = CacheDataset(data=train_files, transform=transforms, cache_rate=1.0, num_workers=4,)


# Loop to accessed images after transformations
for i in range(1): #range(len(train_labels)):
    item = train_dataset.__getitem__(i)  # extract image and label from loaded dataset
    print(item.keys())
    tensor_im = item['t1w']  # get the image after transformations
    print(tensor_im.size())
    tensor_im2 = item['t2w']  # get the image after transformations
    print(tensor_im2.size())
    item['t1w_meta_dict']['spatial_shape'] = np.array([300, 300, 300])
    print(item['t1w_meta_dict'])
    item['t2w_meta_dict']['spatial_shape'] = np.array([300, 300, 300])
    print(item['t2w_meta_dict'])
    saver.save(data=item['t1w'], meta_data=item['t1w_meta_dict'])
    saver.save(data=item['t2w'], meta_data=item['t2w_meta_dict'])




