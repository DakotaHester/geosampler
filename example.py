import geosampler
import rasterio
import os

# Open a raster dataset
dataset_path = os.path.join('data', 'nyc_temp_NAIP_merged_aligned.tif')
with rasterio.open(dataset_path) as dataset:
    test_samples, test_dataset = geosampler.grid_sampler(
        dataset=dataset,
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
