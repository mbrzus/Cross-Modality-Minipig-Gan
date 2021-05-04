import itk
import numpy as np
from pathlib import Path
from monai.transforms.transform import Transform, MapTransform
# from monai.transforms.transform import Transform, MapTransform

class LoadITKImaged(MapTransform):
    def __init__(self, keys, pixel_type=itk.F):
        self.keys = keys
        self.pixel_type = pixel_type
        self.meta_updater = UpdateMetaDatad(keys=self.keys)

    def __call__(self, data):
        d = dict(data)
        for k in self.keys:
            # print(f"reading {d[k]}")
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

class UpdateMetaDatad(MapTransform):
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        d = dict(data)
        # print("saving meta")
        for k in self.keys:
            image = d[k]
            d[f"{k}_meta_dict"] = {}
            d[f"{k}_meta_dict"]["origin"] = np.array(image.GetOrigin())
            d[f"{k}_meta_dict"]["spacing"] = np.array(image.GetSpacing())
            d[f"{k}_meta_dict"]["direction"] = get_direction_cos_from_image(image)

        return d
    
# conversion functions
class ITKImageToNumpyd(MapTransform):
    def __init__(self, keys):
        self.keys = keys
        self.meta_updater = UpdateMetaDatad(keys=self.keys)


    def __call__(self, data):
        d = dict(data)
        d = self.meta_updater(d)
        # print('to np')
        for k in self.keys:
            d[k] = itk.array_from_image(d[k])
        
        return d
    
class ToITKImaged(MapTransform):
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

class ResampleT1T2d(MapTransform):
    def __init__(self, keys, output_size: list = [256, 256, 256], image_type=itk.Image[itk.F, 3]):
        assert len(keys) == 2, "must pass in a t1 key and t2 key: keys=['t1w', 't2w']"
        self.t1w_key = keys[0]
        self.t2w_key = keys[1]
        
        self.output_size = output_size

        # define itk types
        self.image_type = image_type

        # linear iterpolation
        self.linear_interpolator = itk.LinearInterpolateImageFunction[self.image_type, itk.D].New()
        # identity transform
        self.identity_transform = itk.IdentityTransform[itk.D, 3].New()
        

    def __call__(self, data):
        # print("starting functional resample")
        # print("start resampling")

        d = dict(data)
        t1w_itk_image = d[self.t1w_key]
        # print("define resampler")
        # configure resampler
        # resampler = itk.ResampleImageFilter[self.image_type, self.image_type].New()
        # resampler.SetSize(self.output_size)
        # resampler.SetInterpolator(self.linear_interpolator)
        # resampler.SetTransform(self.identity_transform)

        # # identity direction cosine

        identity_direction = self.image_type.New().GetDirection()
        identity_direction.SetIdentity()
        self.linear_interpolator = itk.LinearInterpolateImageFunction.New(t1w_itk_image)

        # resampler.SetOutputDirection(identity_direction)
        
        # set origin to t1w origin
        # resampler.SetOutputOrigin(t1w_itk_image.GetOrigin())
        
        # calculate necessary spacing
        physical_extent = np.array(t1w_itk_image.GetLargestPossibleRegion().GetSize()) * np.array(t1w_itk_image.GetSpacing())
        output_spacing = physical_extent / np.array(self.output_size)
        # resampler.SetOutputSpacing(output_spacing)
        
        # process t1w
        # resampler.SetInput(t1w_itk_image)
        # print("updating resampler")
        # resampler.SetNumberOf
        # resampler.UpdateLargestPossibleRegion()
        # print("updated resampler")

        # resampled_t1w = resampler.GetOutput()
        
        # process t2w
        # resampler.SetInput(t2w_itk_image)
        # resampler.UpdateLargestPossibleRegion()
        # resampled_t2w = resampler.GetOutput()
        # print("starting functional resample")

        required_size_mm = 256
        reference_image = type(t1w_itk_image).New()
        reference_image.SetOrigin(-np.array(self.output_size)/2) # change origin
        reference_image.SetSpacing(required_size_mm / np.array(self.output_size))
        reference_image.SetDirection(identity_direction)
        region = reference_image.GetLargestPossibleRegion()
        region.SetSize(self.output_size)
        reference_image.SetLargestPossibleRegion(region)

        # re-evaluate resampling method
        # 256 mm isotropic
        # output origin = center of physical space - 127.5 mm iso
        # brains abc has max size of human brain (mm)

        # new resample
        # know ac is at 0,0,0
        # sample a 256^3 1mm isotropic grid w/ orign at -127.5,-127.5,-127.5
        # identity direction 
        # identity transform
        # only works on ACPC aligned data



        # print("running resample")
        d[self.t1w_key] = itk.resample_image_filter(
            d[self.t1w_key],
            transform=self.identity_transform,
            interpolator=self.linear_interpolator,
            reference_image=reference_image,
            use_reference_image=True,
            number_of_work_units=1
        )
        # resampler.SetNumWorkUnits(1)
        # object oriented version
        
        d[self.t2w_key] = itk.resample_image_filter(
            d[self.t2w_key],
            transform=self.identity_transform,
            interpolator=self.linear_interpolator,
            reference_image=reference_image,
            use_reference_image=True,
            number_of_work_units=1 # converts to SetNumWorkUnits

            # set num workers
        )
        # print("done resample")



        # resampled = itk.resample_image_filter(d[self.t1w_key],
        #     transform=self.identity_transform,
        #     interpolator=self.linear_interpolator,
        #     size=self.output_size,
        #     output_spacing=output_spacing,
        #     output_direction=identity_direction,
        #     output_origin=t1w_itk_image.GetOrigin()
        # )
        # print(resampled, type(resampled))
        # d[self.t1w_key] = resampled


        # d[self.t2w_key] = itk.resample_image_filter(d[self.t2w_key],
        #     transform=self.identity_transform,
        #     interpolator=self.linear_interpolator,
        #     size=self.output_size,
        #     output_spacing=output_spacing,
        #     output_direction=identity_direction,
        #     output_origin=t1w_itk_image.GetOrigin()
        # )

        # d[self.t2w_key] = resampled_t2w
        # print("end resampling")

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
