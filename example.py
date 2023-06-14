import geosampler
import rasterio
import os

# class GridSampler:
#     def __init__(self: object, 
#                  input_dataset: rasterio.DatasetReader,
#                  target_dataset: rasterio.DatasetReader,
#                  patch_size: int,
#                  split: float,
#                  method: str,
#                  return_remaining_dataset: bool):
#         self.dataset = dataset
#         self.patch_size = patch_size
#         self.split = split
#         self.method = method
#         self.aois = []
#         self.input_patches = []
#         self.output_pathces = []
                
#     def _get_input_numpy_array(self):
#         return self.dataset.read()
    
#     def _get_target_numpy_array(self):
#         return self.dataset.read()
    
#     def _valid_aoi(self, aoi):
#         masks = np.merge(self.input_dataset.read_
    
#     def _create_grid(self):
#         input_array = self._get_input_numpy_array()

# Open a raster dataset
input_dataset_path = os.path.join('data', 'nyc_temp_NAIP_merged_aligned.tif')
target_dataset_path = os.path.join('data', 'nyc_LC_aligned.tif')
with rasterio.open(input_dataset_path) as input_dataset:
    with rasterio.open(target_dataset_path) as target_dataset:
        test_samples, test_dataset = geosampler.grid_sampler(
            input_dataset=input_dataset,
            target_dataset=target_dataset,
            patch_size=600,
            split=0.2,
            method='max_js_distance', # 'max_js_distance' or 'random'
            return_remaining_dataset=True, # returns samples and dataset minmus samples
        )
    # val_samples, train_dataset = geosampler.grid_sampler(
    #     dataset=test_dataset,
    #     patch_size=256,
    #     split=0.25,
    #     method='max_js_distance', # 'max_js_distance' or 'random'
    #     return_remaining_dataset=True, # returns samples and dataset minmus samples
    # )
    # train_samples = geosampler.random_sampler(
    #     dataset=train_dataset,
    #     patch_size=256,
    #     n_samples=3*len(val_samples),
    #     return_remaining_dataset=False, # returns samples and dataset minmus samples
    # )
