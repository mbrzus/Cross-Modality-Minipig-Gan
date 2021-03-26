# base on monai-tutorials 3 spleen segmentation using pytorch lightning notebook

import glob
import os
import shutil
import time


import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import json
import itk
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
    CropForegroundd,
    LoadNiftid,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    ToTensord,
    ThresholdIntensityd
)
from monai.utils import set_determinism


class BrainExtraction(pl.LightningModule):
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
        with open('../metadata/label_paths.json', 'r') as openfile:
            label_json = json.load(openfile)
        with open('../metadata/image_paths.json', 'r') as openfile:
            image_json = json.load(openfile)

        train_labels = label_json["train"]
        train_images = image_json["train"]
        val_labels = label_json["val"]
        val_images = image_json["val"]
        test_labels = label_json["test"]
        test_images = image_json["test"]

        train_files = [
            {"image": image_name, "label": label_name}
            for image_name, label_name in zip(train_images, train_labels)
        ]
        val_files = [
            {"image": image_name, "label": label_name}
            for image_name, label_name in zip(val_images, val_labels)
        ]
        test_files = [
            {"image": image_name, "label": label_name}
            for image_name, label_name in zip(test_images, test_labels)
        ]

        train_files = train_files[:150]
        val_files = val_files[:15]
        test_files = test_files[:15]

        train_transforms = Compose(
            [
                LoadNiftid(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Spacingd(keys=["image", "label"], pixdim=(1, 1, 1), mode=("bilinear", "nearest")),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                ThresholdIntensityd(keys=["label"], threshold=1, above=False, cval=1),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                # randomly crop out patch samples from big image based on pos / neg ratio
                # the image centers of negative samples must be in valid image area
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(96, 96, 96),
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=0,
                ),
                ToTensord(keys=["image", "label"]),

            ]
        )
        val_transforms = Compose(
            [
                LoadNiftid(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Spacingd(
                    keys=["image", "label"], pixdim=(1, 1, 1), mode=("bilinear", "nearest"),
                ),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                ThresholdIntensityd(
                    keys=["label"],
                    threshold=1,
                    above=False,
                    cval=1
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                ToTensord(keys=["image", "label"]),
            ]
        )
        test_transforms = Compose(
            [
                LoadNiftid(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Spacingd(
                    keys=["image", "label"], pixdim=(1, 1, 1), mode=("bilinear", "nearest"),
                ),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                ThresholdIntensityd(
                    keys=["label"],
                    threshold=1,
                    above=False,
                    cval=1
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                ToTensord(keys=["image", "label"]),
            ]
        )

        # we use cached datasets - these are 10x faster than regular datasets
        self.train_dataset = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=4,)
        #print(self.train_ds)
        #for i in range(5):
        #    print(np.shape(self.train_ds[i]))
        # for i in range(0,1000):
        #     for j in range(0,10):
        #         current = self.train_ds[i]
        #         #ve_to_disc(f"temp image_{i}_rand{j}.png", current) #for the first 1000 images in the cache, get 10 samples from the image
        #         #write a function save to disc that save a html report containing the center slices of the 96x96x96 image
        self.val_dataset = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=4,)
        self.test_dataset = CacheDataset(data=test_files, transform=test_transforms, cache_rate=1.0, num_workers=4, )

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
    root_dir = "/home/mbrzus/programming/masterthesis/code"

    # set up loggers and checkpoints
    log_dir = os.path.join(root_dir, "CNN/logs")
    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir=log_dir
    )
    checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(
        filepath=os.path.join(log_dir, "{epoch}-{val_loss:.2f}-{val_dice:.2f}")
    )

    # initialise Lightning's trainer.
    trainer = pl.Trainer(
        gpus=[0],
        max_epochs=1,
        logger=tb_logger,
        checkpoint_callback=checkpoint_callback,
        # precision=16
        amp_backend='apex',
        amp_level='O3'
        #num_sanity_val_steps=1,
        #auto_lr_find=False
    )

    model = BrainExtraction()
    #trainer.tune(model)
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

            # Inference part with saving image to disk
            if i == 0:
                path_to_first_input_test = test_data.__getitem__(0)
            np_image = torch.argmax(test_outputs, dim=1).detach().cpu().numpy()[0]
            np_image = np_image.astype('int16')  # change the number data type
            nifti_image = itk.image_from_array(np_image) # look deeper into how to change this with the use ITK and make sure the image is properly configuret
            # make sure it is in the same physical space as the input - working on it in the itk_inference_postprocessing file
            #output_name = f"inferred_{i}_{os.path.basename(model.train_dataset.data[i]['label'])}"
            #itk.imwrite(nifti_image, f"inferred_test_images/{output_name}")


            # plot the slice [:, :, 80]
            plt.figure("check", (18, 6))
            plt.subplot(1, 3, 1)
            plt.title(f"image {i}")
            plt.imshow(test_data["image"][0, 0, :, :, 80], cmap="gray")
            plt.subplot(1, 3, 2)
            plt.title(f"label {i}")
            plt.imshow(test_data["label"][0, 0, :, :, 80])
            plt.subplot(1, 3, 3)
            plt.title(f"output {i}")
            plt.imshow(torch.argmax(test_outputs, dim=1).detach().cpu()[0, :, :, 80]) # analyze what number should be here
            plt.show()
    print(path_to_first_input_test)
