import itk
import numpy as np
from pathlib import Path


class LoadITKImaged(object):
    def __init__(self, keys, pixel_type=itk.F):
        self.keys = keys
        self.pixel_type = pixel_type
        self.meta_updater = UpdateMetaDatad(keys=self.keys)

    def __call__(self, data):
        d = dict(data)
        for k in self.keys:
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
    def __init__(self, keys):
        self.keys = keys
        pass

    def __call__(self, data):
        d = dict(data)
        for k in self.keys:
            meta_data = d[f"{k}_meta_dict"]
            itk_image = itk.image_from_array(d[k])
            itk_image.SetOrigin(meta_data["origin"])
            itk_image.SetSpacing(meta_data["spacing"])
            itk_image.SetDirection(meta_data["direction"])

            d[k] = itk_image
        return d

class ResampleT1T2d(object):
    def __init__(self, keys, output_size: list = [256, 256, 256], image_type=itk.Image[itk.F, 3]):
        assert len(keys) == 2, "must pass in a t1 key and t2 key: keys=['t1w', 't2w']"
        self.t1w_key = keys[0]
        self.t2w_key = keys[1]
        
        self.output_size = output_size

        # define itk types
        self.image_type = image_type
        
        # linear iterpolation
        self.linear_interpolator = itk.LinearInterpolateImageFunction[self.image_type, itk.D].New()
        self.nearest_interpolator = itk.NearestNeighborInterpolateImageFunction[self.image_type, itk.D].New()
        # identity transform
        self.identity_transform = itk.IdentityTransform[itk.D, 3].New()
        # resampler
        self.resampler = itk.ResampleImageFilter[self.image_type, self.image_type].New()
        
        # identity direction cosine
        self.identity_direction = self.image_type.New().GetDirection()
        self.identity_direction.SetIdentity()

        # configure resampler
        self.resampler.SetSize(self.output_size)
        self.resampler.SetInterpolator(self.linear_interpolator)
        self.resampler.SetTransform(self.identity_transform)
        self.resampler.SetOutputDirection(self.identity_direction)


        

    def __call__(self, data):
        d = dict(data)
        t1w_itk_image = d[self.t1w_key]
        t2w_itk_image = d[self.t2w_key]
        
        # set origin to t1w origin
        self.resampler.SetOutputOrigin(t1w_itk_image.GetOrigin())
        
        # calculate necessary spacing
        physical_extent = np.array(t1w_itk_image.GetLargestPossibleRegion().GetSize()) * np.array(t1w_itk_image.GetSpacing())
        output_spacing = physical_extent / np.array(self.output_size)
        self.resampler.SetOutputSpacing(output_spacing)
        
        # process t1w
        self.resampler.SetInput(t1w_itk_image)
        self.resampler.UpdateLargestPossibleRegion()
        resampled_t1w = self.resampler.GetOutput()
        
        # process t2w
        self.resampler.SetInput(t2w_itk_image)
        self.resampler.UpdateLargestPossibleRegion()
        resampled_t2w = self.resampler.GetOutput()
        
        d[self.t1w_key] = resampled_t1w
        d[self.t2w_key] = resampled_t2w
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

class ToITKImaged(object):
    def __init__(self, keys):
        self.keys = keys
        pass
    def __call__(self, data):
        d = dict(data)
        for k in self.keys:
            meta_data = d[f"{k}_meta_dict"]
            itk_image = itk.image_from_array(d[k])
            itk_image.SetOrigin(meta_data["origin"])
            itk_image.SetSpacing(meta_data["spacing"])
            itk_image.SetDirection(meta_data["direction"])
            d[k] = itk_image
        return d