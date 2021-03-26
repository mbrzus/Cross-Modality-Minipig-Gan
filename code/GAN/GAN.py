# base on monai-tutorials 3 spleen segmentation using pytorch lightning notebook

import glob
import os
import shutil
import time


import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import json
import apex
import numpy as np

from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import CacheDataset, list_data_collate
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import compute_meandice, compute_hausdorff_distance
from monai.networks.layers import Norm
from monai.networks.nets import UNet
from monai.transforms import (
    AddChanneld,
    Compose,
    LoadImaged,
    Orientationd,
    SpatialPadd,
    ScaleIntensityRanged,
    ResizeWithPadOrCropd,
    RandRotated,
    RandGaussianNoised,
    Spacingd,
    ToTensord,
)
from monai.utils import set_determinism


class CrossModGan(pl.LightningModule):
    def __init__(self, lr=0.001):
        super().__init__()
        self.model = UNet(
            dimensions=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH
        )

        self.learning_rate = lr
        self.loss = DiceLoss(to_onehot_y=True, softmax=True)
        self.scaler = torch.cuda.amp.GradScaler()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        for i in batch:
            images, labels = i["image"], i["label"]
            output = self.forward(images)
            loss = self.loss(output, labels)
            self.log("train_loss", loss.item(), on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        roi_size = (160, 160, 160)
        sw_batch_size = 4
        outputs = sliding_window_inference(images, roi_size, sw_batch_size, self.forward)
        val_loss = self.loss(outputs, labels)
        mean_dice_value = compute_meandice(y_pred=outputs, y=labels, include_background=False)
        self.log("val_loss", val_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("mean_dice", mean_dice_value, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return val_loss

    def prepare_data(self):
        with open('../metadata/T1w_paths.json', 'r') as openfile:
            t1w_json = json.load(openfile)
        with open('../metadata/T2w_paths.json', 'r') as openfile:
            t2w_json = json.load(openfile)

        train_labels = t1w_json["train"]
        train_images = t2w_json["train"]
        val_labels = t1w_json["val"]
        val_images = t2w_json["val"]
        test_labels = t1w_json["test"]
        test_images = t2w_json["test"]

        train_files = [
            {"T1w": t1w_path, "T2w": t2w_path}
            for t1w_path, t2w_path in zip(train_images, train_labels)
        ]
        val_files = [
            {"T1w": t1w_path, "T2w": t2w_path}
            for t1w_path, t2w_path in zip(val_images, val_labels)
        ]
        test_files = [
            {"T1w": t1w_path, "T2w": t2w_path}
            for t1w_path, t2w_path in zip(test_images, test_labels)
        ]

        train_files = train_files[:150]
        val_files = val_files[:15]
        test_files = test_files[:15]

        transforms = Compose(
            [
                LoadImaged(keys=["t1w", "t2w"]),
                AddChanneld(keys=["t1w", "t2w"]),
                #Orientationd(keys=["t1w", "t2w"], axcodes="RAS"),
                ResizeWithPadOrCropd(keys=["t1w", "t2w"], spatial_size=[288, 288, 288]),
                ToTensord(keys=["image", "label"]),
            ]
        )


        # we use cached datasets - these are 10x faster than regular datasets
        self.train_dataset = CacheDataset(data=train_files, transform=transforms, cache_rate=1.0, num_workers=4,)
        self.val_dataset = CacheDataset(data=val_files, transform=transforms, cache_rate=1.0, num_workers=4,)
        self.test_dataset = CacheDataset(data=test_files, transform=transforms, cache_rate=1.0, num_workers=4, )

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=5, shuffle=True, num_workers=4)
        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=1, num_workers=4)
        return val_loader

    def test_dataloader(self):
        test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=1, num_workers=4)
        return test_loader


if __name__ == "__main__":
    print_config()
    root_dir = "/home/mbrzus/programming/Cross-Modality-Minipig-Gan/code"

    # set up loggers and checkpoints
    log_dir = os.path.join(root_dir, "GAN/logs")
    tb_logger = pl.loggers.TensorBoardLogger(save_dir=log_dir)
    checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(
        dirpath=log_dir,
        filename="{epoch}-{val_loss:.2f}-{val_dice:.2f}",
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )

    # initialise Lightning's trainer.
    trainer = pl.Trainer(
        gpus=[0],
        max_epochs=5,
        logger=tb_logger,
        checkpoint_callback=checkpoint_callback,
        # precision=16
        # amp_backend='apex',
        # amp_level='O3'
        # num_sanity_val_steps=1,
        # auto_lr_find=False
    )

    model = CrossModGan()
    # trainer.tune(model)
    import time

    start_time = time.time()
    trainer.fit(model)
    print("--- %s seconds ---" % (time.time() - start_time))

    model.eval()
    device = torch.device("cuda:0")
    model.to(device)
    with torch.no_grad():
        for i, test_data in enumerate(model.test_dataloader()):
            roi_size = (160, 160, 160)
            sw_batch_size = 4
            test_outputs = sliding_window_inference(
                test_data["image"].to(device), roi_size, sw_batch_size, model
            )
            # plot the slice [:, :, 80]

            np_image = torch.argmax(test_outputs, dim=1).detach().cpu().numpy()
            meta = test_data['image_meta_dict']
            print(meta)

            plt.figure("check", (18, 6))
            plt.subplot(1, 3, 1)
            plt.title(f"image {i}")
            plt.imshow(test_data["image"][0, 0, :, :, 80], cmap="gray")
            plt.subplot(1, 3, 2)
            plt.title(f"label {i}")
            plt.imshow(test_data["label"][0, 0, :, :, 80])
            plt.subplot(1, 3, 3)
            plt.title(f"output {i}")
            plt.imshow(torch.argmax(test_outputs, dim=1).detach().cpu()[0, :, :, 80])
            plt.show()

