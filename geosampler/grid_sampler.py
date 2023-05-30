import rasterio
from rasterio.plot import show
from rasterio.transform import Affine
import numpy as np
import matplotlib.pyplot as plt
from geosampler import grid_sampler
import os
from scipy.spatial.distance import jensenshannon

def grid_sampler(
    dataset: rasterio.io.DatasetReader | list[rasterio.io.DatasetReader], 
    patch_size: int | tuple[int, int], 
    split: float, 
    method: str = 'max_js_distance', 
    return_remaining_dataset: bool = True
) -> list[rasterio.io.DatasetReader] | tuple[list, rasterio.io.DatasetReader]:
    """Sample a raster dataset using a grid sampling method.
    
    Args:
        dataset (rasterio.io.DatasetReader): A raster dataset.
        patch_size (int | tuple[int, int]): The size of the patches to sample. E.g. 256 or (256, 512).
        split (float): The proportion of the dataset to sample. Should be a value between 0 and 1.
        method (str, optional): The sampling method to use. Defaults to 'max_js_distance'.
        return_remaining_dataset (bool, optional): Whether to return the remaining dataset. Defaults to True.
    Returns:
        list | tuple[list, rasterio.io.DatasetReader]: The samples and the remaining dataset (sans already sampled patches).
    """
    
    if type(dataset) == rasterio.io.DatasetReader:
        dataset = [dataset]
    
    show_datasets = False
    if show_datasets: 
        for dataset in dataset:
            fig, (axnir, axr, axg, axb) = plt.subplots(1, 4, figsize=(25, 10))
            show(dataset.read(4), transform=dataset.transform, ax=axnir, title='NIR', cmap='viridis')
            show(dataset.read(1), transform=dataset.transform, ax=axr, title='Red', cmap='Reds')
            show(dataset.read(2), transform=dataset.transform, ax=axg, title='Green', cmap='Greens')
            show(dataset.read(3), transform=dataset.transform, ax=axb, title='Blue', cmap='Blues')
            plt.show()
    
    patches = get_patches(dataset, patch_size)
    distances = get_distances(patches)
    print(distances)
    
    return None, None

def get_patches(
    dataset: rasterio.io.DatasetReader, 
    patch_size: int | tuple[int, int]
) -> list[rasterio.io.DatasetReader]:
    """Get a list of patches from a raster dataset.
    
    Args:
        dataset (rasterio.io.DatasetReader): A raster dataset.
        patch_size (int): The size of the patches to sample. E.g. 256 or (256, 512).
    Returns:
        list[rasterio.io.DatasetReader]: A list of patches.
    """
    if type(patch_size) == int: patch_size = (patch_size, patch_size)
    
    for dataset in dataset:
        dataset_np = dataset.read()
        
        original_transform = dataset.transform
        original_crs = dataset.crs
        
        patches = []
    
        for i in range(dataset_np.shape[1] // patch_size[0]):
            for j in range(dataset_np.shape[2] // patch_size[1]):
                
                # dataset[:, distance from top, distance from left]
                patch_np = dataset_np[:, i*patch_size[0]:(i+1)*patch_size[0], j*patch_size[1]:(j+1)*patch_size[1]]
                patch_transform = Affine(original_transform[0], original_transform[1], original_transform[2] + j*patch_size[1], # left
                                        original_transform[3], original_transform[4], original_transform[5] - i*patch_size[0]) # top
                
                patch = rasterio.open(
                    os.path.join('data', 'patches', f'patch_{i:05d}_{j:05d}.tif'),
                    'w',
                    driver='GTiff',
                    height=patch_np.shape[1],
                    width=patch_np.shape[2],
                    count=patch_np.shape[0],
                    dtype=patch_np.dtype,
                    crs=original_crs,
                    transform=patch_transform,
                )
                patches.append(patch)
                # patch.write(patch_np)
                
                # fig, ax = plt.subplots(1, 2, figsize=(25, 10))
                # show(dataset.read(1), transform=dataset.transform, ax=ax[0], title='Original')
                # show(patch.read(1), transform=patch.transform, ax=ax[1], title='Patch')
                # plt.show()
        
    return patches

def get_distances(patches: list[rasterio.io.DatasetReader]) -> np.ndarray:
    """Get the js distances between all patches.
    
    Args:
        patches (list[rasterio.io.DatasetReader]): A list of patches.
    Returns:
        np.ndarray: A matrix of distances between all patches.
    """
    
    distances = np.zeros((len(patches), len(patches)))
    for i in range(len(patches)):
        for j in range(len(patches)):
            if patches[i] == patches[j]: continue
            elif j > i: 
                distances[i, j] = distances[j, i]
                continue
            else:
                for band in range(patches[i].shape[2]):
                    histogram_ik = np.histogram(patches[i][:, :, band], bins=256, range=(0, 255))
                    histogram_jk = np.histogram(patches[j][:, :, band], bins=256, range=(0, 255))
                    print(histogram_ik, histogram_jk)
                    distances[i, j] += jensenshannon(histogram_ik, histogram_jk)