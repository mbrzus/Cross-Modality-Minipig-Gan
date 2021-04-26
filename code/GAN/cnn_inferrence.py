import glob
import os
import shutil
import time
import itk

import matplotlib.pyplot as plt
import pytorch_lightning as pl
# from pytorch_lightning import LightningModule 
import torch
from joblib import Parallel, delayed
import json
import numpy as np

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
from transforms import LoadITKImaged, ITKImageToNumpyd, ResampleT1T2d, ToITKImaged, SaveITKImaged
import torch.nn as nn


class CasNetGenerator(pl.LightningModule):
    # source: https://arxiv.org/pdf/1806.06397.pdf
    def __init__(
        self, img_shape, n_unet_blocks=6 # The MEDGAN paper had the best results with 6 unet blocks
    ):  # TODO: change num u_net blocks for actual trraining
        super().__init__()
        self.img_shape = img_shape

        def unet_block(
            in_channels,
            out_channels,
            channels=(16, 32, 64, 128),#, 512),
            strides=(2, 2, 2),#, 2),
            # channels=(16, 32, 64, 128),
            # strides=(2, 2, 2),
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
        print("Generator forward")
        return self.model(x)


if __name__ == "__main__":

    checkpoints_dir = "/Shared/sinapse/aml/correct-resampler/casnet-gen_patchgan-disc"

    # define model and load its parameters
    model = CasNetGenerator.load_from_checkpoint(
        checkpoint_path=f"{checkpoints_dir}/gen_epoch=60-g_loss=100.00-g_recon_loss=0.00-d_loss=45.00.ckpt",
        hparams_file=f"{checkpoints_dir}/default/version_0/hparams.yaml",
        img_shape=(128, 128, 128),
        strict=False
    )
    device = torch.device("cuda:0")
    model.to(device)

    # define path for inference and the MONAI savers
    inferrence_dir = "/Shared/sinapse/cjohnson/inferrence"
    saver_t1w = NiftiSaver(output_dir=f"{inferrence_dir}/t1w", output_postfix="_inputT1")
    saver_label = NiftiSaver(output_dir=f"{inferrence_dir}/t2w", output_postfix="_label")
    saver_predicted = NiftiSaver(output_dir=f"{inferrence_dir}/predicted_t2w", output_postfix="predicted")

    # load the test data
    with open('/Shared/sinapse/mbrzus/Cross-Modality-Minipig-Gan/code/metadata/T1w_paths.json', 'r') as openfile:
        t1_json = json.load(openfile)
    with open('/Shared/sinapse/mbrzus/Cross-Modality-Minipig-Gan/code/metadata/T2w_paths.json', 'r') as openfile:
        t2_json = json.load(openfile)


    test_T1s = t1_json["test"]
    test_T2s = t2_json["test"]

    # zip the test data into a dictionary form
    test_files = [
        {'t1w': t1, 't2w': t2}
        for t2, t1 in zip(test_T2s, test_T1s)
    ]
    test_files = test_files[:1] # for testing use just 1 image

    # define transforms for the data
    transforms = Compose(
        [
            LoadITKImaged(keys=["t1w", "t2w"]),
            ResampleT1T2d(keys=["t1w", "t2w"], output_size=(128, 128, 128)),
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

    

    # create a test dataset with the preprocessed images
    test_dataset = CacheDataset(data=test_files, transform=transforms, cache_rate=1.0, num_workers=4)

    # define matric classes
    dice = DiceMetric()
    hausdorff_distance = HausdorffDistanceMetric()

    # Loop to accessed images after transformations
    for i in range(1): #range(len(test_T2s)):
        item = test_dataset.__getitem__(i)  # extract image and label from loaded dataset
        print(item.keys())
        # save the t1w and label image
        # saver_t1w.save(data=item['t1w'], meta_data=item['T1_meta_dict'])
        # saver_label.save(data=item['t2w'], meta_data=item['T2_meta_dict'])

        # perform the inference
        with torch.no_grad():
            roi_size = (128, 128, 128)
            sw_batch_size = 1
            test_output = sliding_window_inference(
                item['t1w'].unsqueeze(dim=0).to(device), roi_size, sw_batch_size, model
            )
            # out_im = torch.argmax(test_output, dim=1).detach().cpu() # convert from one hot encoding to 1 dimensional
        # test_output = test_output.numpy()
        print(f"test_output.shape(): {test_output.size()}")
        # output_dict = {
        #     "generated_t2w": test_output,
        #     "truth_t2w": item['t2w']
        # }

        item['t2w_generated'] = test_output.squeeze(dim=0).squeeze(dim=0)
        item['t2w'] = item['t2w'].squeeze(dim=0).squeeze(dim=0)
        item['t2w_generated_meta_dict'] = item['t1w_meta_dict']
        item['t2w_generated_meta_dict']['filename'] = "t2_inferred.nii.gz"
        item['t2w_meta_dict']['filename'] = "t2_truth.nii.gz"
        

        print(item.keys())
        print(item['t2w_generated_meta_dict'])
        out_transforms = Compose([
            ToNumpyd(keys=["t2w_generated", "t2w"]),
            ToITKImaged(keys=["t2w_generated", "t2w"]),
            SaveITKImaged(keys=["t2w_generated", "t2w"], out_dir=inferrence_dir, output_postfix="inferred")
        ])

        out_transforms(item)

        # create a metadata dictionary for the output by manipulating the input image meta data
        # out_meta_dict = item['T1_meta_dict']
        # out_meta_dict['filename_or_obj'] = out_meta_dict['filename_or_obj'].replace('T1w', 'predicted')
        # saver_predicted.save(data=out_im, meta_data=out_meta_dict)  # save the predicted label to disk

        # # # Prediction evaluation - metrics
        # print(test_output) #TODO: analyze the intensity values of the one hot inferrence output
        # # create one hot encoding from the ground truth label
        # one_hot_label = one_hot(item['t2w'].unsqueeze(dim=0), 2, dim=1)

        # # Run Mean Dice and Hausdorff Distance metrics using 2 different ways
        # mean_dice = compute_meandice(test_output.detach().cpu(), one_hot_label)
        # mean_dice2 = dice(test_output.detach().cpu(), one_hot_label)
        # hausdorff = compute_hausdorff_distance(test_output.detach().cpu(), one_hot_label)
        # hausdorff2 = hausdorff_distance(test_output.detach().cpu(), one_hot_label)
        # # print the results
        # print(item['T1_meta_dict']['filename_or_obj'])
        # print(f"Mean Dice: {mean_dice}")
        # print(f"Mean Dice: {mean_dice2}")
        # print(f"Hausdorff Distance: {hausdorff}")
        # print(f"Hausdorff Distance: {hausdorff2}")
