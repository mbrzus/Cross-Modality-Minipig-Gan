import glob
import os
import shutil
import time

import os
from argparse import ArgumentParser
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl

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

#TODO: Alex, I commented the data module and the main function. The HumanData... module should be done. The main function
# probably will need adjustment to the data visualization (last loop at the very end) but the logger trainer and all of that should be ready
# I copied and pasted the Generator, Discriminator and GAN from the lightning example but it still need to be changed
# to use MONAI generator and discriminator but I am leaving that to you for now.
# There is also way too much stuff in the "import" part but no need to be to take care of that for now.
# Let me know if you have any questions and have fun playing with it

#TODO: work on those 2 https://colab.research.google.com/github/PytorchLightning/pytorch-lightning/blob/master/notebooks/03-basic-gan.ipynb
#TODO: https://github.com/Project-MONAI/tutorials/blob/master/modules/mednist_GAN_tutorial.ipynb


class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super().__init__()
        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity


class GAN(pl.LightningModule):

    def __init__(
            self,
            channels,
            width,
            height,
            latent_dim: int = 100,
            lr: float = 0.0002,
            b1: float = 0.5,
            b2: float = 0.999,
            batch_size: int = 64,
            **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        # networks
        data_shape = (channels, width, height)
        self.generator = Generator(latent_dim=self.hparams.latent_dim, img_shape=data_shape)
        self.discriminator = Discriminator(img_shape=data_shape)

        self.validation_z = torch.randn(8, self.hparams.latent_dim)

        self.example_input_array = torch.zeros(2, self.hparams.latent_dim)

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, _ = batch

        # sample noise
        z = torch.randn(imgs.shape[0], self.hparams.latent_dim)
        z = z.type_as(imgs)

        # train generator
        if optimizer_idx == 0:
            # generate images
            self.generated_imgs = self(z)

            # log sampled images
            sample_imgs = self.generated_imgs[:6]
            grid = torchvision.utils.make_grid(sample_imgs)
            self.logger.experiment.add_image('generated_images', grid, 0)

            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training_loop
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            # adversarial loss is binary cross-entropy
            g_loss = self.adversarial_loss(self.discriminator(self(z)), valid)
            tqdm_dict = {'g_loss': g_loss}
            output = OrderedDict({
                'loss': g_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

        # train discriminator
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples

            # how well can it label as real?
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            real_loss = self.adversarial_loss(self.discriminator(imgs), valid)

            # how well can it label as fake?
            fake = torch.zeros(imgs.size(0), 1)
            fake = fake.type_as(imgs)

            fake_loss = self.adversarial_loss(
                self.discriminator(self(z).detach()), fake)

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2
            tqdm_dict = {'d_loss': d_loss}
            output = OrderedDict({
                'loss': d_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

    def on_epoch_end(self):
        z = self.validation_z.type_as(self.generator.model[0].weight)

        # log sampled images
        sample_imgs = self(z)
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image('generated_images', grid, self.current_epoch)


#TODO: this module is ready. It might need some changes if we will make changes to the data.
class HumanBrainDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.meta_dir = "/home/mbrzus/programming/Cross-Modality-Minipig-Gan/code/metadata/"

    def prepare_data(self):
        # read the json files
        with open(f"{self.meta_dir}T1w_paths.json", 'r') as openfile:
            t1w_json = json.load(openfile)
        with open(f"{self.meta_dir}T2w_paths.json", 'r') as openfile:
            t2w_json = json.load(openfile)

        # organize the data to train, test and val
        train_t1w = t1w_json["train"]
        train_t2w = t2w_json["train"]
        val_t1w = t1w_json["val"]
        val_t2w = t2w_json["val"]
        test_t1w = t1w_json["test"]
        test_t2w = t2w_json["test"]

        # organize files into pairs (t1w, t2w) for MONAI dictionary workflow
        train_files = [
            {"T1w": t1w_path, "T2w": t2w_path}
            for t1w_path, t2w_path in zip(train_t2w, train_t1w)
        ]
        val_files = [
            {"T1w": t1w_path, "T2w": t2w_path}
            for t1w_path, t2w_path in zip(val_t2w, val_t1w)
        ]
        test_files = [
            {"T1w": t1w_path, "T2w": t2w_path}
            for t1w_path, t2w_path in zip(test_t2w, test_t1w)
        ]

        # get just a very small portion of the data for initial test (fail fast)
        train_files = train_files[:15]
        val_files = val_files[:5]
        test_files = test_files[:5]

        # transforms to prepare the data for pytorch monai training
        transforms = Compose(
            [
                LoadImaged(keys=["t1w", "t2w"]),
                AddChanneld(keys=["t1w", "t2w"]),
                #optional orientation change, I dont think we need it with our data
                #Orientationd(keys=["t1w", "t2w"], axcodes="RAS"),
                # we probably want to pad images to the same size (i didn't check the data so this probably will need an update)
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

    data = HumanBrainDataModule()
    model = GAN(*data.size())
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
    trainer.fit(model, data)
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

