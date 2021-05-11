import argparse
from pathlib import Path
import itk
import imageio
from monai.transforms import ScaleIntensity
import numpy as np

AXIS_TO_INDEX = {"l": 0, "p": 1, "s": 2}


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
