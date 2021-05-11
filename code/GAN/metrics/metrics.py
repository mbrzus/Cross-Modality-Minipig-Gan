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
from monai.inferers import sliding_window_inference, SimpleInferer, SlidingWindowInferer
from monai.losses import DiceLoss, GlobalMutualInformationLoss
from monai.metrics import compute_meandice, DiceMetric, compute_hausdorff_distance, HausdorffDistanceMetric
from monai.metrics import *
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
    ThresholdIntensityd,
    Lambdad
)
from transforms import *
from transforms2 import *
from GAN_old_discriminator_no_patches import GAN
from sklearn.metrics import mean_absolute_error



def get_test_data():
    meta_dir = str(Path(".").absolute().parent / "metadata")
    with open(f"{meta_dir}/structure.json", "r") as openfile:
        structure = json.load(openfile)

    test_structure = structure["test"]

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
    return test_files



if __name__ == "__main__":

    checkpoints_dir = "/Shared/sinapse/aml/old_discriminator_log"
    inferrence_dir = "/Shared/sinapse/cjohnson/inferrence"

    # define model and load its parameters


    model = GAN.load_from_checkpoint(
        channels=1,
        width=128,
        height=128,
        depth=128,
        checkpoint_path=f"{checkpoints_dir}/gen_recon_epoch=30-g_loss=100.03-g_recon_loss=0.03-d_loss=45.00.ckpt",
        hparams_file=f"{checkpoints_dir}/default/version_13/hparams.yaml",
        img_shape=(128, 128, 128),
        strict=False
    )
    device = torch.device("cuda:0")
    model.to(device)
    model.eval()
    model.freeze()

    # prepare test data
    test_files = get_test_data()
    # test_files = test_files[:1]

    print(f"Performing inferrence on {len(test_files)} files")

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

    mean_absolute_error = torchmetrics.MeanAbsoluteError()
    # ssim = torchmetrics.SSIM()

    mae_by_t1 = {}
    # mae_metric = MAEMetric()
    # Loop to accessed images after transformations
    for i in range(len(test_files)):
        item = test_dataset.__getitem__(i)  # extract image and label from loaded dataset

        t1_transforms = Compose([
            # ToNumpyd(keys=["t1w"]),
            ScaleIntensityRangePercentilesd(
                keys=["t1w"],
                lower=0,
                upper=100,
                b_min=0,
                b_max=255,
                clip=True,
                relative=False,
            ),
            Lambdad(keys=["t1w"], func=lambda x: np.round(x)),
            ToTensord(keys=["t1w"])
            # ToITKImaged(keys=["t1w"]),
            # SaveITKImaged(keys=["t1w"], out_dir="/Shared/sinapse/aml/inferrence_rescaled", output_postfix="")
        ])

        t1_dict = t1_transforms(item)

        # perform the inference
        with torch.no_grad():
            test_output = model.generator.forward(item['t1w'].unsqueeze(dim=0).to(device))

        item['t2w_generated'] = test_output.detach().cpu()
        item['t2w_generated_meta_dict'] = item['t1w_meta_dict']
        item['t2w_generated_meta_dict']['filename'] = "t2_inferred.nii.gz"
        item['t2w_meta_dict']['filename'] = "t2_truth.nii.gz"

        # Use mean absolute error
        # print(f"mae_metric: {mae_metric(item['t2w_generated'], item['t2w'])}")


        # print(item['t2w_generated'].shape)
        # print(item['t2w'].shape)

        out_transforms = Compose([
            # ToNumpyd(keys=["t2w_generated", "t2w"]),
            ScaleIntensityRangePercentilesd(
                keys=["t2w_generated", "t2w"],
                lower=0,
                upper=100,
                b_min=0,
                b_max=255,
                clip=True,
                relative=False,
            ),
            Lambdad(keys=["t2w_generated", "t2w"], func=lambda x: np.round(x)),
            ToTensord(keys=["t2w_generated", "t2w"])
            # ToITKImaged(keys=["t2w_generated", "t2w"]),
            # SaveITKImaged(keys=["t2w_generated", "t2w"], out_dir="/Shared/sinapse/aml/inferrence_rescaled", output_postfix=Path(test_files[i].get('t1w')).with_suffix('').with_suffix('').name)
        ])


        t2_dict = out_transforms(item)
        # print(t2_dict.keys())
        # print(type(t1_dict['t1w']))
        # print(t1_dict['t1w'].shape)


        t1_gt = t1_dict['t1w'].squeeze(dim=0)
        t2_gen = t2_dict['t2w_generated'].squeeze(dim=0).squeeze(dim=0)
        t2_gt = t2_dict['t2w'].squeeze(dim=0)
        print(float(mean_absolute_error(t2_gen, t2_gt)))
        # print(ssim(t2_gen, t2_gt))
        mae_by_t1[Path(test_files[i].get('t1w')).with_suffix('').with_suffix('').name] = {}
        mae_by_t1[Path(test_files[i].get('t1w')).with_suffix('').with_suffix('').name]['t2gen_vs_t2gt'] = float(mean_absolute_error(t2_gen, t2_gt))
        mae_by_t1[Path(test_files[i].get('t1w')).with_suffix('').with_suffix('').name]['t2gt_vs_t2gt'] = float(mean_absolute_error(t2_gt, t2_gt))
        mae_by_t1[Path(test_files[i].get('t1w')).with_suffix('').with_suffix('').name]['t1gt_vs_t2gt'] = float(mean_absolute_error(t1_gt, t2_gt))
         

# print(mae_by_t1)
with open('mean_absolute_error.json', 'w') as outfile:
    json.dump(mae_by_t1, outfile)