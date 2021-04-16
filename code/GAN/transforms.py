import itk
import numpy as np


class LoadITKImaged(object):
    def __init__(self, keys, pixel_type=itk.F):
        self.keys = keys
        self.pixel_type = pixel_type
        self.meta_updater = UpdateMetaDatad(keys=self.keys)

    def __call__(self, data):
        d = dict(data)
        for k in self.keys:
            print(f"reading {d[k]}")
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
        print("saving meta")
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
        print('to np')
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
        

    def __call__(self, data):
        print("start resampling")

        print("define resampler")
        # configure resampler
        resampler = itk.ResampleImageFilter[self.image_type, self.image_type].New()
        resampler.SetSize(self.output_size)
        resampler.SetInterpolator(self.linear_interpolator)
        resampler.SetTransform(self.identity_transform)
        # identity direction cosine
        identity_direction = self.image_type.New().GetDirection()
        identity_direction.SetIdentity()
        resampler.SetOutputDirection(identity_direction)

        d = dict(data)
        t1w_itk_image = d[self.t1w_key]
        t2w_itk_image = d[self.t2w_key]
        
        # set origin to t1w origin
        resampler.SetOutputOrigin(t1w_itk_image.GetOrigin())
        
        # calculate necessary spacing
        physical_extent = np.array(t1w_itk_image.GetLargestPossibleRegion().GetSize()) * np.array(t1w_itk_image.GetSpacing())
        output_spacing = physical_extent / np.array(self.output_size)
        resampler.SetOutputSpacing(output_spacing)
        
        # process t1w
        resampler.SetInput(t1w_itk_image)
        print("updating resampler")
        # resampler.SetNumberOf
        resampler.UpdateLargestPossibleRegion()
        print("updated resampler")

        resampled_t1w = resampler.GetOutput()
        
        # process t2w
        resampler.SetInput(t2w_itk_image)
        resampler.UpdateLargestPossibleRegion()
        resampled_t2w = resampler.GetOutput()
        
        d[self.t1w_key] = resampled_t1w
        d[self.t2w_key] = resampled_t2w
        print("end resampling")

        return d
