import geosampler
import rasterio
import os
import pickle
import numpy as np

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
        # Gather test and validation samples at the same time since same method is being used to sample
        test_val_samples = geosampler.grid_sampler(
            input_dataset=input_dataset,
            target_dataset=target_dataset,
            patch_size=600,
            split=0.25, # if using uniform_js_distance, this is the proportion of samples to use for validation (needs to be a reciprocal of 2, 3, 4, etc.)
            method='uniform_js_distance', # 'max_js_distance' or 'random'
            out_dir='data',
            name='test',
            save=False,
        )
        # samples are returned as a list of tuples (input_patch, target_patch, window)
        # where window is a rasterio window object
        # split test and validation samples
        # select every other sample for validation
        val_samples = [sample for sample in test_val_samples[::2]]
        test_samples = [sample for sample in test_val_samples[1::2]]
        
        
        # define previously sampled regions
        # probably will change how window objects are passed at some point
        prev_windows = [sample[2] for sample in test_val_samples]
        # with open(os.path.join('data', 'test_samples.pkl'), 'wb') as f:
        #     pickle.dump(test_samples, f)
        # val_samples = geosampler.grid_sampler(
        #     input_dataset=input_dataset,
        #     target_dataset=target_dataset,
        #     patch_size=600,
        #     split=0.2,
        #     method='uniform_js_distance', # 'max_js_distance' or 'random'
        #     previously_sampled_regions=prev_windows,
        #     out_dir='data',
        #     name='val',
        # )
        # prev_windows.extend([sample[2] for sample in val_samples])
        with open(os.path.join('data', 'val_samples.pkl'), 'wb') as f:
            pickle.dump(val_samples, f)
        with open(os.path.join('data', 'test_samples.pkl'), 'rb') as f:
            test_samples = pickle.load(f)
        # prev_windows = [sample[2] for sample in test_samples]
        
        # with open(os.path.join('data', 'test_samples.pkl'), 'rb') as f:
        #     test_samples = pickle.load(f)
        # with open(os.path.join('data', 'val_samples.pkl'), 'rb') as f:
        #     val_samples = pickle.load(f)
        # print(test_samples, val_samples)
        # prev_windows = [sample[2] for sample in test_samples]
        # prev_windows.extend([sample[2] for sample in val_samples])
        geosampler.save_patches(val_samples, input_dataset.meta, 'data', 'val')
        geosampler.save_patches(test_samples, input_dataset.meta, 'data', 'test')
        
        # can generate as much training data as needed now. In this example, we generate 5x the number of test samples
        train_samples = geosampler.random_sampler(
            input_dataset=input_dataset,
            target_dataset=target_dataset,
            patch_size=600,
            n_samples=5*len(test_samples),
            allow_overlap=True,
            previously_sampled_regions=prev_windows,
            out_dir='data',
            name='train',
        )
        
        with open(os.path.join('data', 'train_samples.pkl'), 'wb') as f:
            pickle.dump(train_samples, f)
