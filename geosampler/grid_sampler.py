import rasterio
from rasterio.plot import show
from rasterio.transform import Affine
from rasterio.windows import Window
import numpy as np
import matplotlib.pyplot as plt
from geosampler import grid_sampler
import os
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm

def grid_sampler(
    input_dataset: rasterio.io.DatasetReader,
    target_dataset: rasterio.io.DatasetReader,
    patch_size: int | tuple[int, int], 
    split: float, 
    method: str = 'max_js_distance', 
    return_remaining_dataset: bool = True
) -> tuple[list, rasterio.io.DatasetReader]:
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
    
    
    # show_datasets = False
    # if show_datasets: 
    #     for dataset in dataset:
    #         fig, (axnir, axr, axg, axb) = plt.subplots(1, 4, figsize=(25, 10))
    #         show(dataset.read(4), transform=dataset.transform, ax=axnir, title='NIR', cmap='viridis')
    #         show(dataset.read(1), transform=dataset.transform, ax=axr, title='Red', cmap='Reds')
    #         show(dataset.read(2), transform=dataset.transform, ax=axg, title='Green', cmap='Greens')
    #         show(dataset.read(3), transform=dataset.transform, ax=axb, title='Blue', cmap='Blues')
    #         plt.show()
    
    patches = get_patches(input_dataset, target_dataset, patch_size)
    # distances = get_distances(patches)
    # print(distances)
    
    return None, None

def is_aoi_valid(
    aoi: tuple[slice, slice],
    input_dataset: rasterio.io.DatasetReader,
    target_dataset: rasterio.io.DatasetReader
) -> bool:
    input_np, target_np = input_dataset.read_masks(window=Window.from_slices(aoi[0], aoi[1])), target_dataset.read_masks(window=Window.from_slices(aoi[0], aoi[1]))
    if input_dataset.nodata in input_np or target_dataset.nodata in target_np: return False
    else: return True

def get_patches(
    input_dataset: rasterio.io.DatasetReader, 
    target_dataset: rasterio.io.DatasetReader,
    patch_size: int | tuple[int, int]
) -> list[tuple[rasterio.io.DatasetReader, rasterio.io.DatasetReader]]:
    """Get a list of patches from a raster dataset.
    
    Args:
        dataset (rasterio.io.DatasetReader): A raster dataset.
        patch_size (int): The size of the patches to sample. E.g. 256 or (256, 512).
    Returns:
        list[rasterio.io.DatasetReader]: A list of patches.
    Raises:
        ValueError: If the input and target datasets have different dimensions.
    """
    if input_dataset.width != target_dataset.width or input_dataset.height != target_dataset.height:
        raise ValueError(f'Input and target datasets must have the same dimensions. \n Input dataset: {input_dataset.width}x{input_dataset.height} \n Target dataset: {target_dataset.width}x{target_dataset.height}')
    if type(patch_size) == int: patch_size = (patch_size, patch_size)
    
    input_patch_out_path = os.path.join('data', 'patches', 'input')
    target_patch_out_path = os.path.join('data', 'patches', 'target')
    if not os.path.exists(input_patch_out_path): os.makedirs(input_patch_out_path)
    if not os.path.exists(target_patch_out_path): os.makedirs(target_patch_out_path)
    
    height = input_dataset.height
    width = input_dataset.width
    patches_arrays = []
    
    patches = []
    
    # get all valid patches
    for i in range(height // patch_size[1]):
        for j in range(width // patch_size[0]):
            
            # dataset[:, distance from top, distance from left]
            aoi_slices = (slice(i*patch_size[0], (i+1)*patch_size[0]), slice(j*patch_size[1], (j+1)*patch_size[1]))
            aoi = (i*patch_size[0], (i+1)*patch_size[0], j*patch_size[1], (j+1)*patch_size[1])
            if not is_aoi_valid(aoi_slices, input_dataset, target_dataset): # invalid patch    
                continue
            
            patches_arrays.append((aoi, input_dataset.read(window=Window.from_slices(aoi_slices[0], aoi_slices[1])))) # target dataset should align with input dataset, so aoi is all that is needed
    
    # get js distance matrix
    # for aoi, patch in zip(patches_arrays):
    distance_matrix = np.zeros((len(patches_arrays), len(patches_arrays)))
        
    for i in range(len(patches_arrays)):
        for j in range(len(patches_arrays)):
            # matrix is symmetric
            if i < j:
                distance_matrix[j, i] = distance_matrix[i, j]
            # don't compare patch to itself
            elif i == j:
                continue
            # compute js distance
            else:
                for band in range(patches_arrays[0][1].shape[0]):
                    histogram_ik, _ = np.histogram(patches_arrays[i][1][band, :, :], bins=256, range=(0, 255))
                    histogram_jk, _ = np.histogram(patches_arrays[j][1][band, :, :], bins=256, range=(0, 255))
                    # print(histogram_ik.shape, histogram_jk.shape)
                    distance_matrix[i, j] += jensenshannon(histogram_ik, histogram_jk)

    print(distance_matrix)
        
    return patches