import glob
import os
import shutil
import time
import itk

import matplotlib.pyplot as plt
import pytorch_lightning as pl
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
    SaveImaged,
    ScaleIntensityRangePercentilesd,
    ScaleIntensityRanged,
    ResizeWithPadOrCropd,
    RandRotated,
    RandGaussianNoised,
    Spacingd,
    ToTensord,
    ThresholdIntensityd
)


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
    # define path to trained model parameters
    # checkpoints_dir = "/Shared/sinapse/cjohnson/CNN/logs"
    # params_dir = "/Shared/sinapse/cjohnson/CNN/logs/default/minipig_3mm"

    root_dir = str(Path(".").absolute().parent)  # use relative path

    # set up loggers and checkpoints
    checkpoints_dir = os.path.join(root_dir, "GAN/casnet-gen_michal-disc")

    # define model and load its parameters
    model = BrainExtraction.load_from_checkpoint(
        checkpoint_path=f"{checkpoints_dir}/minipig_3mm.ckpt",
        hparams_file=f"{checkpoints_dir}/hparams.yaml"
        # hparams_file=f"{params_dir}/hparams.yaml"
    )
    device = torch.device("cuda:0")
    model.to(device)

    # define path for inference and the MONAI savers
    inferrence_dir = "/Shared/sinapse/cjohnson/CNN/inferred_test_images/minipig_3mm"
    saver_t1w = NiftiSaver(output_dir=f"{inferrence_dir}/t1w", output_postfix="")
    saver_label = NiftiSaver(output_dir=f"{inferrence_dir}/label", output_postfix="")
    saver_predicted = NiftiSaver(output_dir=f"{inferrence_dir}/predicted_label", output_postfix="")

    # load the test data
    with open('../metadata/minipig_label_paths.json', 'r') as openfile:
        label_json = json.load(openfile)
    with open('../metadata/minipig_image_paths.json', 'r') as openfile:
        image_json = json.load(openfile)

    test_labels = label_json["test"]
    test_images = image_json["test"]

    # zip the test data into a dictionary form
    test_files = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(test_images, test_labels)
    ]
    test_files = test_files[:1] # for testing use just 1 image

    # define transforms for the data
    transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ThresholdIntensityd(keys=["label"], threshold=1, above=False, cval=1),
            Spacingd(keys=["image", "label"], pixdim=(3, 3, 3), mode=("bilinear", "nearest")),
            ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=[96, 96, 96]),
            ScaleIntensityRangePercentilesd(
                keys=["image"],
                lower=1.0,
                upper=99.0,
                b_min=-1.0,
                b_max=1.0,
                clip=True,
                relative=False,
            ),
            ToTensord(keys=["image", "label"]),
        ]
    )

    # create a test dataset with the preprocessed images
    test_dataset = CacheDataset(data=test_files, transform=transforms, cache_rate=1.0, num_workers=4)

    # define matric classes
    dice = DiceMetric()
    hausdorff_distance = HausdorffDistanceMetric()

    # Loop to accessed images after transformations
    for i in range(1): #range(len(test_images)):
        item = test_dataset.__getitem__(i)  # extract image and label from loaded dataset
        # save the t1w and label image
        saver_t1w.save(data=item['image'], meta_data=item['image_meta_dict'])
        saver_label.save(data=item['label'], meta_data=item['label_meta_dict'])

        # perform the inference
        with torch.no_grad():
            roi_size = (96, 96, 96)
            sw_batch_size = 1
            test_output = sliding_window_inference(
                item['image'].unsqueeze(dim=0).to(device), roi_size, sw_batch_size, model
            )
            out_im = torch.argmax(test_output, dim=1).detach().cpu() # convert from one hot encoding to 1 dimensional

        # create a metadata dictionary for the output by manipulating the input image meta data
        out_meta_dict = item['image_meta_dict']
        out_meta_dict['filename_or_obj'] = out_meta_dict['filename_or_obj'].replace('T1w', 'predicted')
        saver_predicted.save(data=out_im, meta_data=out_meta_dict)  # save the predicted label to disk

        # Prediction evaluation - metrics
        print(test_output) #TODO: analyze the intensity values of the one hot inferrence output
        # create one hot encoding from the ground truth label
        one_hot_label = one_hot(item["label"].unsqueeze(dim=0), 2, dim=1)

        # Run Mean Dice and Hausdorff Distance metrics using 2 different ways
        mean_dice = compute_meandice(test_output.detach().cpu(), one_hot_label)
        mean_dice2 = dice(test_output.detach().cpu(), one_hot_label)
        hausdorff = compute_hausdorff_distance(test_output.detach().cpu(), one_hot_label)
        hausdorff2 = hausdorff_distance(test_output.detach().cpu(), one_hot_label)
        # print the results
        print(item['image_meta_dict']['filename_or_obj'])
        print(f"Mean Dice: {mean_dice}")
        print(f"Mean Dice: {mean_dice2}")
        print(f"Hausdorff Distance: {hausdorff}")
        print(f"Hausdorff Distance: {hausdorff2}")
