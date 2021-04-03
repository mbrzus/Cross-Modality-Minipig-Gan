import glob
import json
import os
import shutil
import time
from argparse import ArgumentParser
from collections import OrderedDict
from itertools import product as cartesian_product
from pathlib import Path

import apex
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import CacheDataset, list_data_collate
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import compute_hausdorff_distance, compute_meandice
from monai.networks.layers import Norm
from monai.networks.nets import UNet
from monai.networks.nets import Discriminator as MONAIDiscriminator
from monai.transforms import (
    AddChanneld,
    Compose,
    LoadImaged,
    Orientationd,
    RandGaussianNoised,
    RandRotated,
    ResizeWithPadOrCropd,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
    ToTensord,
    ScaleIntensityRangePercentilesd,
    Resized,
)
from monai.utils import set_determinism
from torch.utils.data import DataLoader, random_split
from monai.visualize.img2tensorboard import plot_2d_or_3d_image

# TODO: Alex, I commented the data module and the main function. The HumanData... module should be done. The main function
# probably will need adjustment to the data visualization (last loop at the very end) but the logger trainer and all of that should be ready
# I copied and pasted the Generator, Discriminator and GAN from the lightning example but it still need to be changed
# to use MONAI generator and discriminator but I am leaving that to you for now.
# There is also way too much stuff in the "import" part but no need to be to take care of that for now.
# Let me know if you have any questions and have fun playing with it

# TODO: work on those 2 https://colab.research.google.com/github/PytorchLightning/pytorch-lightning/blob/master/notebooks/03-basic-gan.ipynb
# TODO: https://github.com/Project-MONAI/tutorials/blob/master/modules/mednist_GAN_tutorial.ipynb


# taking CasNet Generator from: https://arxiv.org/pdf/1806.06397.pdf
# using normal discriminator

# replaced w/ CasNetGenerator
# class Generator(nn.Module):
#     def __init__(self, latent_dim, img_shape):
#         super().__init__()
#         self.img_shape = img_shape

#         def block(in_feat, out_feat, normalize=True):
#             layers = [nn.Linear(in_feat, out_feat)]
#             if normalize:
#                 layers.append(nn.BatchNorm1d(out_feat, 0.8))
#             layers.append(nn.LeakyReLU(0.2, inplace=True))
#             return layers

#         self.model = nn.Sequential(
#             *block(latent_dim, 128, normalize=False),
#             *block(128, 256),
#             *block(256, 512),
#             *block(512, 1024),
#             nn.Linear(1024, int(np.prod(img_shape))),
#             nn.Tanh(),
#         )

#     def forward(self, z):
#         img = self.model(z)
#         img = img.view(img.size(0), *self.img_shape)
#         return img


class CasNetGenerator(nn.Module):
    # source: https://arxiv.org/pdf/1806.06397.pdf
    def __init__(
        self, img_shape, n_unet_blocks=3
    ):  # TODO: change num u_net blocks for actual trraining
        super().__init__()
        self.img_shape = img_shape

        def unet_block(
            in_channels,
            out_channels,
            channels=(16, 32, 64, 128),
            strides=(2, 2, 2),
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
        return self.model(x)


# TODO: look at using Markovian discriminator (PatchGAN) from: https://arxiv.org/pdf/1611.07004.pdf
# TODO: improve discriminator by adding convolutional layers

# NOTE: temporarily switching to monai discriminator to try and get prelim results
class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super().__init__()

        self.model = MONAIDiscriminator(
            img_shape,
            channels=(8, 16, 32, 64, 128, 256, 1),
            strides=(2, 2, 2, 2, 2, 2, 2, 1),
            num_res_units=2,
            kernel_size=3,
            act="PRELU",
            norm=None,
            last_act="SIGMOID",
        )

    def forward(self, img):
        validity = self.model(img)
        return validity


class GAN(pl.LightningModule):
    # note: i don't think we need a latenet dim? -- isn't the t1w image just the latent space?
    def __init__(
        self,
        channels,
        width,
        height,
        depth,
        latent_dim: int = 100,
        lr: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
        batch_size: int = 64,
        example_data: torch.Tensor = None,
        one_sided_label_value: int = 0.9,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters("latent_dim", "lr", "b1", "b2", "batch_size", "one_sided_label_value")

        # networks
        data_shape = (channels, width, height, depth)
        self.generator = CasNetGenerator(img_shape=data_shape)
        self.discriminator = Discriminator(img_shape=data_shape)

        self.example_input_array = example_data["t1w"]

    def forward(self, x):
        return self.generator(x)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        t1w_images, t2w_images = batch["t1w"], batch["t2w"]

        # train generator
        if optimizer_idx == 0:
            # generate images
            self.generated_imgs = self(t1w_images)

            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training_loop

            valid = torch.ones(t1w_images.shape[0], 1)
            valid = valid.type_as(t1w_images)

            # adversarial loss is binary cross-entropy
            g_loss = self.adversarial_loss(self.discriminator(self(t1w_images)), valid)
            self.log("g_loss", g_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            return g_loss

        # train discriminator
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples

            # how well can it label as real?
            valid = torch.ones(t1w_images.shape[0], 1) * self.hparams.one_sided_label_value
            valid = valid.type_as(t1w_images)

            real_loss = self.adversarial_loss(self.discriminator(t2w_images), valid)

            # how well can it label as fake?
            fake = torch.zeros(t1w_images.shape[0], 1)
            fake = fake.type_as(t1w_images)

            fake_loss = self.adversarial_loss(
                self.discriminator(self(t1w_images).detach()), fake
            )

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2
            self.log("d_loss", d_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            return d_loss

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

    def on_epoch_end(self):
        # log sampled images -- just logs from the last batch run
        plot_2d_or_3d_image(self.generated_imgs, self.current_epoch, self.logger.experiment, tag='generated_t2w')

# TODO: this module is ready. It might need some changes if we will make changes to the data.
class HumanBrainDataModule(pl.LightningDataModule):
    # spatial size has a huge impact on the size of the gpu we need
    # potentially look at a sliding window method?
    def __init__(self, spatial_size=[128, 128, 128]):
        super().__init__(
            self,
        )
        # use relative path so this isn't machine specific
        self.meta_dir = str(Path(".").absolute().parent / "metadata")

        # dims is returned from calling .size() method
        self.spatial_size = spatial_size
        self.dims = (1, *self.spatial_size)

    def prepare_data(self):
        # read the json files
        with open(f"{self.meta_dir}/structure.json", "r") as openfile:
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
        # train_files = train_files[:50]
        # val_files = val_files[:5]
        # test_files = test_files[:5]

        # transforms to prepare the data for pytorch monai training
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
                # optional orientation change, I dont think we need it with our data
                # Orientationd(keys=["t1w", "t2w"], axcodes="RAS"),
                # we probably want to pad images to the same size (i didn't check the data so this probably will need an update)
                # NOTE: should we consider using a resize rather than a crop/pad?
                Resized(keys=["t1w", "t2w"], spatial_size=self.spatial_size),
                ToTensord(keys=["t1w", "t2w"]),
            ]
        )

        # we use cached datasets - these are 10x faster than regular datasets
        # TODO: adjust cache rate to fix memory problems
        self.train_dataset = CacheDataset(
            data=train_files,
            transform=transforms,
            cache_num=200,
            num_workers=10,
        )
        self.val_dataset = CacheDataset(
            data=val_files,
            transform=transforms,
            cache_num=10,
            num_workers=4,
        )
        self.test_dataset = CacheDataset(
            data=test_files,
            transform=transforms,
            cache_num=10,
            num_workers=4,
        )

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=5, shuffle=True, num_workers=4
        )
        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=1, num_workers=4
        )
        return val_loader

    def test_dataloader(self):
        test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=1, num_workers=4
        )
        return test_loader


if __name__ == "__main__":
    print_config()
    root_dir = str(Path(".").absolute().parent)  # use relative path

    # set up loggers and checkpoints
    log_dir = os.path.join(root_dir, "GAN/casnet-gen_monai-disc")
    tb_logger = pl.loggers.TensorBoardLogger(save_dir=log_dir)
    # TODO: find a good metric for determining the best model to checkpoint (naive using g_loss for now)
    generator_checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(
        dirpath=log_dir,
        filename="gen_{epoch}-{g_loss:.2f}-{d_loss:.2f}",
        save_top_k=1,
        verbose=True,
        monitor="g_loss_step",
        mode="min",
    )

    discriminator_checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(
        dirpath=log_dir,
        filename="dis_{epoch}-{g_loss:.2f}-{d_loss:.2f}",
        save_top_k=1,
        verbose=True,
        monitor="d_loss_step",
        mode="min",
    )

    data = HumanBrainDataModule()
    data.prepare_data()
    example = next(iter(data.test_dataloader()))
    model = GAN(*data.size(), example_data=example)
    # initialise Lightning's trainer.
    trainer = pl.Trainer(
        gpus=[0],
        max_epochs=100,
        logger=tb_logger,
        callbacks=[generator_checkpoint_callback, discriminator_checkpoint_callback]
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

    # this appears to be segmentation related -- DELETE
    # model.eval()
    # device = torch.device("cuda:2")
    # model.to(device)
    # with torch.no_grad():
    #     for i, test_data in enumerate(model.test_dataloader()):
    #         roi_size = (160, 160, 160)
    #         sw_batch_size = 4
    #         test_outputs = sliding_window_inference(
    #             test_data["image"].to(device), roi_size, sw_batch_size, model
    #         )
    #         # plot the slice [:, :, 80]

    #         np_image = torch.argmax(test_outputs, dim=1).detach().cpu().numpy()
    #         meta = test_data["image_meta_dict"]
    #         print(meta)

    #         plt.figure("check", (18, 6))
    #         plt.subplot(1, 3, 1)
    #         plt.title(f"image {i}")
    #         plt.imshow(test_data["image"][0, 0, :, :, 80], cmap="gray")
    #         plt.subplot(1, 3, 2)
    #         plt.title(f"label {i}")
    #         plt.imshow(test_data["label"][0, 0, :, :, 80])
    #         plt.subplot(1, 3, 3)
    #         plt.title(f"output {i}")
    #         plt.imshow(torch.argmax(test_outputs, dim=1).detach().cpu()[0, :, :, 80])
    #         plt.show()
