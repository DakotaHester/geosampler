import rasterio
from rasterio.plot import show
from rasterio.transform import Affine
from rasterio.windows import Window
import numpy as np
import matplotlib.pyplot as plt
from geosampler import grid_sampler
import os
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm, trange
from fast_histogram import histogram1d
from numba import njit 

def grid_sampler(
    input_dataset: rasterio.io.DatasetReader,
    target_dataset: rasterio.io.DatasetReader,
    patch_size: int | tuple[int, int], 
    split: float, 
    method: str = 'max_js_distance',
    previously_sampled_regions: list[Window] = None,
    out_dir: str = 'data',
    name: str = 'test',
    save: bool = True,
) -> list[tuple[np.ndarray[np.uint8], np.ndarray[np.uint8], Window]]:
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
    
    if type(patch_size) == int: patch_size = (patch_size, patch_size)
    if not previously_sampled_regions: previously_sampled_regions = []
    
    if method.endswith('js_distance'):
        # patches = 3-tuple(input, target, window)
        patches = get_full_grid(input_dataset, target_dataset, patch_size, previously_sampled_regions)
        n_samples = int(len(patches) * split)
    
        distance_matrix = get_js_distance_matrix(patches)
        sum_distances = np.sum(distance_matrix, axis=1)
        # sort patches by distance 
        # https://stackoverflow.com/a/6618543
        if method.startswith('max'):
            sorted_patches_by_distance = [patch for _, patch in sorted(zip(sum_distances, patches), key=lambda pair: pair[0], reverse=True)]
        else:
            sorted_patches_by_distance = [patch for _, patch in sorted(zip(sum_distances, patches), key=lambda pair: pair[0])]
        
        if method.startswith('uniform'):
            multiple = int(1.0 / split)
            sampled_patches = [patch for i, patch in enumerate(sorted_patches_by_distance) if i % multiple == 0]
        else:
            sampled_patches = sorted_patches_by_distance[:n_samples]
    elif method == 'random':
        raise NotImplementedError('Random sampling from grid not yet implemented')
        sampled_patches = np.random.choice(patches, n_samples, replace=False)
    else:
        raise ValueError(f'Invalid sampling method: {method}')
    
    if save: save_patches(sampled_patches, input_dataset.meta, out_dir, name)
    
    return sampled_patches

def random_sampler(
    input_dataset: rasterio.io.DatasetReader,
    target_dataset: rasterio.io.DatasetReader,
    patch_size: int | tuple[int, int],
    n_samples: int,
    allow_overlap: bool = True,
    previously_sampled_regions: list[Window] = None,
    out_dir: str = 'data',
    name: str = 'train',
    save: bool = True,
) -> list[tuple[np.ndarray[np.uint8], np.ndarray[np.uint8], Window]]:
    
    if type(patch_size) == int: patch_size = (patch_size, patch_size)
    if not previously_sampled_regions: previously_sampled_regions = []
    
    range_vertical = range(input_dataset.height - patch_size[0])
    range_horizontal = range(input_dataset.width - patch_size[1])
    
    patches = []
    with tqdm(total=n_samples, desc="Sampling patches randomly") as pbar:
        while len(patches) < n_samples:
            prev_windows = [patch[2] for patch in patches]
            sample_vertical, sample_horizontal = np.random.choice(range_vertical), np.random.choice(range_horizontal)
            # dataset[:, distance from top, distance from left]
            aoi_slices = (slice(sample_vertical, sample_vertical + patch_size[0]), slice(sample_horizontal, sample_horizontal + patch_size[1]))
            sample_window = get_window_from_aoi(aoi_slices)
            
            if sample_window in previously_sampled_regions or sample_window in prev_windows: continue # prevent dupes
            if window_already_sampled(sample_window, previously_sampled_regions): continue # prevent overlap with previously sampled regions
            if not allow_overlap and window_already_sampled(sample_window, prev_windows): continue # prevent overlap with current samples
            
            input_np, target_np = load_arrays_from_window(sample_window, input_dataset, target_dataset)
            if input_np is False or target_np is False: continue
            else: 
                patches.append((input_np, target_np, sample_window))
                pbar.update(1)
    
    if save: save_patches(patches, input_dataset.meta, out_dir, name)
    return patches
    
def get_window_from_aoi(
    aoi_slices: tuple[slice, slice],
) -> Window:
    return Window.from_slices(aoi_slices[0], aoi_slices[1])

def load_arrays_from_window(
    window: Window,
    input_dataset: rasterio.io.DatasetReader,
    target_dataset: rasterio.io.DatasetReader
) -> tuple[np.ndarray, np.ndarray]:
    input_np, target_np = input_dataset.read(window=window), target_dataset.read(window=window)
    if input_dataset.nodata in input_np or target_dataset.nodata in target_np: return False
    else: return input_np, target_np

def get_full_grid(
    input_dataset: rasterio.io.DatasetReader, 
    target_dataset: rasterio.io.DatasetReader,
    patch_size: int | tuple[int, int],
    already_sampled_windows: list[Window] = None,
) -> list[tuple[np.ndarray, np.ndarray, Window]]:

    if input_dataset.width != target_dataset.width or input_dataset.height != target_dataset.height:
        raise ValueError(f'Input and target datasets must have the same dimensions. \n Input dataset: {input_dataset.width}x{input_dataset.height} \n Target dataset: {target_dataset.width}x{target_dataset.height}')
    if type(patch_size) == int: patch_size = (patch_size, patch_size)
    
    
    height = input_dataset.height
    width = input_dataset.width
    patches_arrays = []
    
    # get all valid patches
    for i in trange(height // patch_size[1], desc='Sampling all patches from grid'):
        for j in range(width // patch_size[0]):
            
            # dataset[:, distance from top, distance from left]
            aoi_slices = (slice(i*patch_size[0], (i+1)*patch_size[0]), slice(j*patch_size[1], (j+1)*patch_size[1]))
            # aoi = (i*patch_size[0], (i+1)*patch_size[0], j*patch_size[1], (j+1)*patch_size[1])
            aoi_window = get_window_from_aoi(aoi_slices)
            if window_already_sampled(aoi_window, already_sampled_windows): continue
            input_np, target_np = load_arrays_from_window(aoi_window, input_dataset, target_dataset)
            
            patches_arrays.append((input_np, target_np, aoi_window)) # target dataset should align with input dataset, so aoi is all that is needed (and by extension, the window)
    
    return patches_arrays

# @njit(parallel=True)
def get_js_distance_matrix(patches: list[tuple[np.ndarray, np.ndarray, Window]]) -> np.ndarray[np.float64]:
    n_patches = len(patches)
    distance_matrix = np.zeros((n_patches, n_patches))
        
    for i in trange(n_patches-1, -1, -1, desc='Computing JS distances'): # 
        for j in range(n_patches):
            # matrix is symmetric, don't compute twice
            if j >= i:
                break
            # compute js distance
            else:
                for band in range(patches[0][1].shape[0]):
                    # print(histogram_ik.shape, histogram_jk.shape)
                    distance_matrix[i, j] += jensenshannon(
                        histogram1d(patches[i][0][band, :, :], bins=256, range=(0, 255)), 
                        histogram1d(patches[j][0][band, :, :], bins=256, range=(0, 255))
                    )    
    distance_matrix = distance_matrix + distance_matrix.T # reflect over diagonal

    return distance_matrix

def window_already_sampled(
    window: Window,
    previous_sampled: list[Window],
) -> bool:
    for sampled in previous_sampled:
        try:
            if rasterio.windows.intersect(window, sampled):
                return True
            else: continue
        except rasterio.errors.WindowError as e:
            RuntimeWarning(f'WindowError: {e}')
            continue
    return False

def save_patches(
    patches: list[tuple[np.ndarray, np.ndarray, Window]],
    original_dataset_metadata: dict,
    out_dir: str,
    name: str,
) -> None:
    original_transform = original_dataset_metadata['transform']
    original_crs = original_dataset_metadata['crs']
    
    input_out_dir = os.path.join(out_dir, name, 'input')
    target_out_dir = os.path.join(out_dir, name, 'target')
    if not os.path.exists(input_out_dir): os.makedirs(input_out_dir)
    if not os.path.exists(target_out_dir): os.makedirs(target_out_dir)
    
    for i, patch in enumerate(tqdm(patches, desc='Saving patches')):
        input_patch, target_patch, window = patch
        
        new_transform = rasterio.windows.transform(window, original_transform)
        with rasterio.open(
            os.path.join(input_out_dir, f'{i:05d}.tif'),
            'w',
            driver='GTiff',
            height=input_patch.shape[1],
            width=input_patch.shape[2],
            count=input_patch.shape[0],
            dtype=input_patch.dtype,
            crs=original_crs,
            transform=new_transform,
        ) as input_out_file: input_out_file.write(input_patch)

        with rasterio.open(
            os.path.join(target_out_dir, f'{i:05d}.tif'),
            'w',
            driver='GTiff',
            height=target_patch.shape[1],
            width=target_patch.shape[2],
            count=target_patch.shape[0],
            dtype=target_patch.dtype,
            crs=original_crs,
            transform=new_transform,
        ) as target_out_file: target_out_file.write(target_patch)