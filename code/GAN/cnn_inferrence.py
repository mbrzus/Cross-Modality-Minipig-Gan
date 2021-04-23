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
    ToTensord,
    ThresholdIntensityd
)
from transforms import LoadITKImaged, ITKImageToNumpyd, ResampleT1T2d


class BrainExtraction(pl.LightningModule):
    def __init__(self, lr=0.001):
        super().__init__()
        self.model = UNet(
            dimensions=3,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128),
            strides=(2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH
        )

        self.learning_rate = lr
        self.loss = DiceLoss(to_onehot_y=True, softmax=True)
        self.scaler = torch.cuda.amp.GradScaler()

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":

    checkpoints_dir = "/Shared/sinapse/aml/correct-resampler/casnet-gen_patchgan-disc"

    # define model and load its parameters
    model = BrainExtraction.load_from_checkpoint(
        checkpoint_path=f"{checkpoints_dir}/gen_epoch=20-g_loss=100.00-g_recon_loss=0.00-d_loss=45.00.ckpt",
        hparams_file=f"{checkpoints_dir}/default/version_0/hparams.yaml",
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
        {"T1": t1, "T2": t2}
        for t2, t1 in zip(test_T2s, test_T1s)
    ]
    test_files = test_files[:1] # for testing use just 1 image

    # define transforms for the data
    transforms = Compose(
        [
            LoadImaged(keys=["T1", "T2"]),
            AddChanneld(keys=["T1", "T2"]),
            Orientationd(keys=["T1", "T2"], axcodes="RAS"),
            ThresholdIntensityd(keys=["T2"], threshold=1, above=False, cval=1),
            Spacingd(keys=["T1", "T2"], pixdim=(3, 3, 3), mode=("bilinear", "nearest")),
            ResizeWithPadOrCropd(keys=["T1", "T2"], spatial_size=[96, 96, 96]),
            ScaleIntensityRangePercentilesd(
                keys=["T1"],
                lower=1.0,
                upper=99.0,
                b_min=-1.0,
                b_max=1.0,
                clip=True,
                relative=False,
            ),
            ToTensord(keys=["T1", "T2"]),
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
        print(item)
        # save the t1w and label image
        saver_t1w.save(data=item['T1'], meta_data=item['T1_meta_dict'])
        saver_label.save(data=item['T2'], meta_data=item['T2_meta_dict'])

        # perform the inference
        with torch.no_grad():
            roi_size = (96, 96, 96)
            sw_batch_size = 1
            test_output = sliding_window_inference(
                item['T1'].unsqueeze(dim=0).to(device), roi_size, sw_batch_size, model
            )
            out_im = torch.argmax(test_output, dim=1).detach().cpu() # convert from one hot encoding to 1 dimensional

        # create a metadata dictionary for the output by manipulating the input image meta data
        out_meta_dict = item['T1_meta_dict']
        out_meta_dict['filename_or_obj'] = out_meta_dict['filename_or_obj'].replace('T1w', 'predicted')
        saver_predicted.save(data=out_im, meta_data=out_meta_dict)  # save the predicted label to disk

        # # Prediction evaluation - metrics
        print(test_output) #TODO: analyze the intensity values of the one hot inferrence output
        # # create one hot encoding from the ground truth label
        # one_hot_label = one_hot(item["T2"].unsqueeze(dim=0), 2, dim=1)

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
