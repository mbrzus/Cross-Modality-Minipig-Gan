import argparse
from pathlib import Path
import itk
import imageio
from monai.transforms import ScaleIntensity
import numpy as np

AXIS_TO_INDEX = {"l": 0, "p": 1, "s": 2}


class ResampleT1T2d(object):
    def __init__(
        self, output_size: list = [128, 128, 128], image_type=itk.Image[itk.F, 3]
    ):

        self.output_size = output_size

        # define itk types
        self.image_type = image_type

        # linear iterpolation
        self.linear_interpolator = itk.LinearInterpolateImageFunction[
            self.image_type, itk.D
        ].New()
        # identity transform
        self.identity_transform = itk.IdentityTransform[itk.D, 3].New()

    def __call__(self, data):
        # print("starting functional resample")
        # print("start resampling")

        t1w_itk_image = data

        identity_direction = self.image_type.New().GetDirection()
        identity_direction.SetIdentity()
        self.linear_interpolator = itk.LinearInterpolateImageFunction.New(t1w_itk_image)

        physical_extent = np.array(
            t1w_itk_image.GetLargestPossibleRegion().GetSize()
        ) * np.array(t1w_itk_image.GetSpacing())
        output_spacing = physical_extent / np.array(self.output_size)

        required_size_mm = 256
        reference_image = type(t1w_itk_image).New()
        reference_image.SetOrigin(-np.array(self.output_size) / 2)  # change origin
        reference_image.SetSpacing(required_size_mm / np.array(self.output_size))
        reference_image.SetDirection(identity_direction)
        region = reference_image.GetLargestPossibleRegion()
        region.SetSize(self.output_size)
        reference_image.SetLargestPossibleRegion(region)

        data = itk.resample_image_filter(
            data,
            transform=self.identity_transform,
            interpolator=self.linear_interpolator,
            reference_image=reference_image,
            use_reference_image=True,
            number_of_work_units=1,
        )

        return data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", help="MRI to convert to GIF", required=True)
    parser.add_argument("-o", "--out-dir", help="output directory", default=None)
    parser.add_argument(
        "-a",
        "--axis",
        choices=["l", "p", "s"],
        help="axis along which gif will slide",
        required=True,
    )

    args = parser.parse_args()
    assert Path(args.image).is_file(), f"NO FILE: {args.image}"
    return args


def image_to_gif(image_filename, axis, out_dir=None):
    image = itk.imread(image_filename, itk.F)
    image = ResampleT1T2d()(image)
    image = itk.array_from_image(image)
    image = ScaleIntensity(minv=0, maxv=255)(image).astype(np.uint8)

    # get output filename
    image_filename = Path(str(image_filename))
    name = str(image_filename.name).split(".")[0]

    if out_dir is None:
        out_dir = image_filename.parent

    output_filename = str(Path(out_dir) / f"{name}_{axis}.gif")

    with imageio.get_writer(output_filename, mode="I") as writer:
        for slice_idx in list(range(image.shape[AXIS_TO_INDEX[axis]]))[::-1]:
            slice = None
            if axis == "l":
                slice = image[slice_idx, :, :]
            elif axis == "p":
                slice = image[:, slice_idx, :]
            else:  # axis == 's'
                slice = image[:, :, slice_idx]

            assert slice is not None, "IMPOSSIBLE FAILURE"

            writer.append_data(slice)

    # print(output_filename)


if __name__ == "__main__":
    args = parse_args()
    image_to_gif(args.image, args.axis, args.out_dir)
