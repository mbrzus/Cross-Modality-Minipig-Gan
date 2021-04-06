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

# Read data from the json files - json files store paths to images and labels
metadata_dir = "/home/mbrzus/programming/masterthesis/code/metadata"
with open(f"{metadata_dir}/minipig_label_paths.json", 'r') as openfile:
    label_json = json.load(openfile)
with open(f"{metadata_dir}/minipig_image_paths.json", 'r') as openfile:
    image_json = json.load(openfile)

# read the data from dictionary into array
train_labels = label_json["test"]
train_images = image_json["test"]
# train_labels = label_json['train'] + label_json['val'] + label_json['test']
# train_images = image_json['train'] + image_json['val'] + image_json['test']

# zip two arrays into a dictionary form expected by MONAI
train_files = [
            {"image": image_name, "label": label_name}
            for image_name, label_name in zip(train_images, train_labels)
        ]
train_files = train_files[:1]  # get just one training (image, label) pair for inspection

# path to directory where the tested images will be stored
path_to_write = "/home/mbrzus/programming/masterthesis/code/test/image_transform_pipeline_test"
saver = NiftiSaver(output_dir=path_to_write,
                   dtype=None)

# Original image and label
orig_image = train_files[0]['image']  # path to the original image
orig_im_itk = itk.imread(orig_image)  # read the image using itk

# orig_spacing = orig_im_itk.GetSpacing()
# print(orig_im_itk.GetSpacing())
#
# print(type(orig_im_itk.GetSpacing()))
# new_sp = itk.GetVnlVectorFromArray(np.array([0.5, 0.5, 0.5]))
# print(new_sp)
# print(type(new_sp))

np_orig_im = itk.array_from_image(orig_im_itk)  # get a numpy array from the itk image
print(f"Initial image shape: {np_orig_im.shape}")  # print shape of the numpy array
itk.imwrite(orig_im_itk, f"{path_to_write}/original_image.nii.gz")  # write itk image to disk
# same for label
orig_label = train_files[0]['label']
orig_lbl_itk = itk.imread(orig_label)
np_orig_lbl = itk.array_from_image(orig_lbl_itk)
print(f"Initial label shape: {np_orig_lbl.shape}")
itk.imwrite(orig_lbl_itk, f"{path_to_write}/original_label.nii.gz")


# MONAI transforms used for preprocessing image
# detailed description of the transforms and their behavior is in the preprocessing_transforms.pptx presentation
train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ThresholdIntensityd(keys=["label"], threshold=1, above=False, cval=1),
        Spacingd(keys=["image", "label"], pixdim=(3, 3, 3), mode=("bilinear", "nearest")),
        #ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=[96, 96, 96]),
        # RandRotated(keys=["image", "label"],
        #             range_x=(-20, 20),
        #             range_y=(-20, 20),
        #             range_z=(-20, 20),
        #             prob=0.2,
        #             ),
        # RandGaussianNoised(keys=["image"]),
        ScaleIntensityd(keys=["image"], minv=-1.0, maxv=1.0),
        ToTensord(keys=["image", "label"]),
    ]
)

# MONAI Cache dataset function uses cache to efficiently load the images with transformations
train_dataset = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=4,)


# Loop to accessed images after transformations
for i in range(1): #range(len(train_labels)):
    item = train_dataset.__getitem__(i)  # extract image and label from loaded dataset
    print(item.keys())
    tensor_im = item['image']  # get the image after transformations
    print(tensor_im.size())
    meta = item['image_meta_dict']
    # new_meta_dict = {}
    # new_meta_dict['affine'] = meta['affine']#[0].numpy()
    # new_meta_dict['original_affine'] = meta['original_affine']#[0].numpy()
    # new_meta_dict['spatial_shape'] = meta['spatial_shape']#[0].numpy()
    # new_meta_dict['filename_or_obj'] = meta['filename_or_obj']#[0]
    # print(new_meta_dict)
    # print(type(new_meta_dict['affine'][0][0]))
    # print(type(new_meta_dict['original_affine'][0][0]))
    # print(type(new_meta_dict['spatial_shape'][0]))
    # print(type(new_meta_dict['filename_or_obj']))
    saver.save(data=item['image'], meta_data=item['image_meta_dict'])
    saver.save(data=item['label'], meta_data=item['label_meta_dict'])
    #print(type(item['image_meta_dict']))
    #print(item['image_meta_dict'])



