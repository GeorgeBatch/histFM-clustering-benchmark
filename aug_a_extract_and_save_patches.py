import argparse
import glob
import os
import sys
from pathlib import Path
import numpy as np
import random
import shutil
from concurrent.futures import ProcessPoolExecutor
from tiatoolbox.tools.patchextraction import SlidingWindowPatchExtractor
from tiatoolbox.utils import imwrite

# local imports
from source.constants import PATCHES_SAVE_DIR

def process_slide(slide_path, dataset, patch_size, num_patches, patch_resolution, patch_units):
    slide_path = Path(slide_path)
    wsi_id = slide_path.stem
    extractor = SlidingWindowPatchExtractor(
        slide_path,
        resolution=patch_resolution,
        units=patch_units,
        patch_size=(patch_size, patch_size),
        stride=(patch_size, patch_size),
        input_mask="otsu",
        min_mask_ratio=0.5,
    )

    # Set random seed
    random.seed(42)
    np.random.seed(42)

    # Get all coordinates
    all_coordinates = extractor.coordinate_list
    num_coordinates = all_coordinates.shape[0]

    # Sample random coordinates
    sample_indexes = np.random.choice(num_coordinates, min(num_patches, num_coordinates), replace=False)
    sample_coordinates = all_coordinates[sample_indexes]

    # Create slide directory if it doesn't exist
    slide_dir = Path(f"{PATCHES_SAVE_DIR}/{dataset}/{wsi_id}")
    slide_dir.mkdir(parents=True, exist_ok=True)

    # Save patches
    for idx, coord in enumerate(sample_coordinates):
        patch = extractor.wsi.read_rect(
            location=(coord[0], coord[1]),
            size=(patch_size, patch_size),
            resolution=patch_resolution,
            units=patch_units,
            pad_mode="constant",
            pad_constant_values=0,
            coord_space="resolution",
        )
        imwrite(f"{slide_dir}/{coord[0]}_{coord[1]}_{patch_size}.jpg", patch)
    
    num_extracted_patches = len(list(slide_dir.glob("*.jpg")))
    print(f"Successfully extracted {num_extracted_patches} patches for dataset={dataset} slide={wsi_id}")

def process_slide_wrapper(args):
    process_slide(*args)

def main(dataset, slide_format, overwrite, patch_size, num_patches, patch_resolution, patch_units, num_workers):
    # Create directory if it doesn't exist
    dataset_dir = Path(f"{PATCHES_SAVE_DIR}/{dataset}")
    if dataset_dir.exists():
        if overwrite:
            shutil.rmtree(dataset_dir)
            print(f"Overwrite invoked. Deleted existing directory: {dataset_dir}")
        else:
            raise FileExistsError(f"Directory {dataset_dir} already exists. Use --overwrite to delete it.")
    os.makedirs(dataset_dir, exist_ok=True)

    # Iterate over all slides in the dataset directory
    slide_paths = glob.glob(f"/well/rittscher-dart/users/qun786/projects/current/comp-path/dependency-mil-private/WSI/{dataset}/all_classes/*.{slide_format}")
    args_list = [(slide_path, dataset, patch_size, num_patches, patch_resolution, patch_units) for slide_path in slide_paths]
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        executor.map(process_slide_wrapper, args_list)

# default command
# Done
# python aug_a_extract_and_save_patches.py --dataset ouh_batch1_20x --slide_format ndpi --patch_size 448 --num_patches 50 --patch_resolution 0.5 --patch_units mpp --num_workers 24
# python aug_a_extract_and_save_patches.py --dataset ouh_batch1_40x --slide_format tif --patch_size 448 --num_patches 50 --patch_resolution 0.5 --patch_units mpp --num_workers 24
# python aug_a_extract_and_save_patches.py --dataset ouh_batch2_20x --slide_format tif --patch_size 448 --num_patches 50 --patch_resolution 0.5 --patch_units mpp --num_workers 24
# python aug_a_extract_and_save_patches.py --dataset ouh_batch3_40x --slide_format tif --patch_size 448 --num_patches 50 --patch_resolution 0.5 --patch_units mpp --num_workers 24
# python aug_a_extract_and_save_patches.py --dataset DART_001 --slide_format tif --patch_size 448 --num_patches 50 --patch_resolution 0.5 --patch_units mpp --num_workers 24
# python aug_a_extract_and_save_patches.py --dataset DART_002 --slide_format tif --patch_size 448 --num_patches 50 --patch_resolution 0.5 --patch_units mpp --num_workers 24
# python aug_a_extract_and_save_patches.py --dataset DART_003 --slide_format tif --patch_size 448 --num_patches 50 --patch_resolution 0.5 --patch_units mpp --num_workers 24
# python aug_a_extract_and_save_patches.py --dataset DART_004 --slide_format tif --patch_size 448 --num_patches 50 --patch_resolution 0.5 --patch_units mpp --num_workers 24
# python aug_a_extract_and_save_patches.py --dataset TCIA-CPTAC_test --slide_format svs --patch_size 448 --num_patches 50 --patch_resolution 0.5 --patch_units mpp --num_workers 24
# python aug_a_extract_and_save_patches.py --dataset TCIA-CPTAC --slide_format svs --patch_size 448 --num_patches 50 --patch_resolution 0.5 --patch_units mpp --num_workers 24
# python aug_a_extract_and_save_patches.py --dataset TCGA-lung --slide_format svs --patch_size 448 --num_patches 50 --patch_resolution 0.5 --patch_units mpp --num_workers 24
# python aug_a_extract_and_save_patches.py --dataset CAMELYON16 --slide_format tif --patch_size 448 --num_patches 50 --patch_resolution 0.5 --patch_units mpp --num_workers 24
# TODO:
# 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and save patches from WSI slides.")
    parser.add_argument("--dataset", type=str, default="ouh_batch1_20x", help="Path to the dataset directory containing WSI slides.")
    parser.add_argument("--slide_format", type=str, default="ndpi", help="Format of the slide files (e.g., ndpi, svs).")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Delete the dataset directory if it exists.")
    parser.add_argument("--patch_size", type=int, default=448, help="Size of the patches to extract.")
    parser.add_argument("--num_patches", type=int, default=50, help="Number of random patches to extract.")
    parser.add_argument("--patch_resolution", type=float, default=0.5, help="Resolution of the patches to extract.")
    parser.add_argument("--patch_units", type=str, default="mpp", help="Units of the patch resolution.")
    parser.add_argument("--num_workers", type=int, default=24, help="Number of worker processes to use.")
    args = parser.parse_args()

    print('\n', "-" * 48, '\n')
    print("Arguments passed to the script:")
    print("\nargs - raw:")
    print(sys.argv)
    print("\nargs:")
    print(args)
    print('\n', "-" * 48, '\n')

    main(args.dataset, args.slide_format, args.overwrite, args.patch_size, args.num_patches, args.patch_resolution, args.patch_units, args.num_workers)

