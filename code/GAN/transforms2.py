# This file was written by Alex Powers and then modified by Michal Brzus

import itk
import numpy as np
from pathlib import Path
import torch


class LoadITKImaged(object):
    def __init__(self, keys, pixel_type=itk.F):
        self.keys = keys
        self.pixel_type = pixel_type
        self.meta_updater = UpdateMetaDatad(keys=self.keys)

    def __call__(self, data):
        d = dict(data)
        for k in self.keys:
            # save off the file name
            if f"{k}_meta_dict" not in d.keys():
                d[f"{k}_meta_dict"] = {"filename": d[k]}
            else:
                d[f"{k}_meta_dict"]["filename"] = d[k]

            d[k] = itk.imread(d[k], self.pixel_type)

        d = self.meta_updater(d)

        return d


def get_direction_cos_from_image(image):
    dims = len(image.GetOrigin())
    arr = np.array([[0.0] * dims] * dims)
    mat = image.GetDirection().GetVnlMatrix()
    for i in range(dims):
        for j in range(dims):
            arr[i][j] = mat.get(i, j)
    return arr


class UpdateMetaDatad(object):
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        d = dict(data)
        for k in self.keys:
            image = d[k]
            if f"{k}_meta_dict" not in d.keys():
                d[f"{k}_meta_dict"] = {}
            d[f"{k}_meta_dict"]["origin"] = np.array(image.GetOrigin())
            d[f"{k}_meta_dict"]["spacing"] = np.array(image.GetSpacing())
            d[f"{k}_meta_dict"]["direction"] = get_direction_cos_from_image(image)

        return d


# conversion functions
class ITKImageToNumpyd(object):
    def __init__(self, keys):
        self.keys = keys
        self.meta_updater = UpdateMetaDatad(keys=self.keys)

    def __call__(self, data):
        d = dict(data)
        d = self.meta_updater(d)
        for k in self.keys:
            d[k] = itk.array_from_image(d[k])

        return d


class ToITKImaged(object):
    #TODO: apply changes and test like in the other transform file
    def __init__(self, keys):
        self.keys = keys
        pass

    def __call__(self, data):
        d = dict(data)
        for k in self.keys:
            if torch.is_tensor(d[k]):
                d[k] = d[k].numpy().astype(np.float16)
            if len(d[k].shape) == 5:
                d[k] = d[k].squeeze(axis=0).squeeze(axis=0)
            elif len(d[k].shape) == 4:
                d[k] = d[k].squeeze(axis=0)

            meta_data = d[f"{k}_meta_dict"]
            itk_image = itk.image_from_array(d[k])
            itk_image.SetOrigin(meta_data["origin"])
            itk_image.SetSpacing(meta_data["spacing"])
            itk_image.SetDirection(meta_data["direction"])

            d[k] = itk_image
        return d

class SaveITKImaged(object):
    def __init__(self, keys, out_dir, output_postfix="inf"):
        self.keys = keys
        self.postfix = output_postfix
        self.out_dir = out_dir

    def __call__(self, data):
        d = dict(data)
        for k in self.keys:
            input_filename = Path(d[f"{k}_meta_dict"]["filename"]).absolute()
            parent_dir = input_filename.parent
            basename = str(input_filename.name).split('.')[0]
            extension = '.'.join(str(input_filename).split('.')[-2:])

            output_filename = f"{self.out_dir}/{basename}_{self.postfix}.{extension}"
            print("writing to", output_filename)
            itk.imwrite(d[k], output_filename)
            pass

        return d


unsqueze_lambda = lambda x: x.squeeze(dim=0)
shape_lambda = lambda x: x.shape


class MiniPigResampled(object):
    def __init__(self, keys, output_size: list = [128, 128, 128], image_type=itk.Image[itk.F, 3]):
        #assert len(keys) == 2, "must pass in a t1w key and label key"
        self.t1w_key = keys[0]
        self.output_size = output_size

        # define itk types
        self.image_type = image_type

        # linear iterpolation
        self.linear_interpolator = itk.LinearInterpolateImageFunction[self.image_type, itk.D].New()

        # identity transform
        self.identity_transform = itk.IdentityTransform[itk.D, 3].New()
        # resampler
        self.t1w_resampler = itk.ResampleImageFilter[self.image_type, self.image_type].New()

        # configure t1w resampler
        self.t1w_resampler.SetSize(self.output_size)
        self.t1w_resampler.SetInterpolator(self.linear_interpolator)
        self.t1w_resampler.SetTransform(self.identity_transform)



    def __call__(self, data):
        d = dict(data)
        t1w_itk_image = d[self.t1w_key]

        # calculate parameters
        t1w_direction = t1w_itk_image.GetDirection()
        t1w_origin = np.asarray(t1w_itk_image.GetOrigin())
        t1w_spacing = t1w_itk_image.GetSpacing()
        t1w_physical_size = np.array(t1w_itk_image.GetLargestPossibleRegion().GetSize()) * np.array(
            t1w_spacing)
        t1w_output_spacing = t1w_physical_size / np.array(self.output_size)
        t1w_output_origin = np.add(np.subtract(np.asarray(t1w_origin), np.asarray(t1w_spacing)/2),
                                np.asarray(t1w_output_spacing)/2)

        # set t1w resampler parameters
        self.t1w_resampler.SetOutputOrigin(t1w_output_origin)
        self.t1w_resampler.SetOutputDirection(t1w_direction)
        self.t1w_resampler.SetOutputSpacing(t1w_output_spacing)
        # process t1w
        self.t1w_resampler.SetInput(t1w_itk_image)
        self.t1w_resampler.UpdateLargestPossibleRegion()

        # update the dictionary
        d[self.t1w_key] = self.t1w_resampler.GetOutput()
        return d


class BinaryThresholdd(object):
    def __init__(self, keys, low: int, high: int, threshold_value: int):
        assert len(keys) == 1, "ensure we are calling it only on the label"
        self.label_key = keys[0]
        self.low = low
        self.high = high
        self.threshold_value = threshold_value
        image_type = itk.Image[itk.F, 3]
        self.thresholdFilter = itk.BinaryThresholdImageFilter[image_type, image_type].New()

    def __call__(self, data):
        d = dict(data)
        label_itk_image = d[self.label_key]
        self.thresholdFilter.SetInput(label_itk_image)
        self.thresholdFilter.SetLowerThreshold(self.threshold_value)
        self.thresholdFilter.SetOutsideValue(self.low)
        self.thresholdFilter.SetInsideValue(self.high)
        self.thresholdFilter.Update()
        label = self.thresholdFilter.GetOutput()

        d[self.label_key] = label
        return d


    


