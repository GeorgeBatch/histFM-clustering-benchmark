import argparse
import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import shutil
from concurrent.futures import ProcessPoolExecutor

# local imports
from source.constants import PATCHES_SAVE_DIR, AUG_PATCHES_SAVE_DIR

def augment_patch(patch_path, output_dir, patch_size, num_patches):
    patch = Image.open(patch_path)
    base_name = os.path.basename(patch_path)
    coords = base_name.split('_')[0:2]
    
    for i in range(num_patches):
        degrees = (360 / num_patches) * i
        augmented_patch = patch.rotate(degrees).crop(
            ((patch.width - patch_size) // 2,
             (patch.height - patch_size) // 2,
             (patch.width + patch_size) // 2,
             (patch.height + patch_size) // 2)
        )
        augmented_patch.save(f"{output_dir}/{coords[0]}_{coords[1]}_rot{int(degrees)}.jpg")

def process_slide(slide_dir, output_base_dir, patch_size, num_patches):
    slide_id = os.path.basename(slide_dir)
    output_dir = Path(output_base_dir) / slide_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    patch_paths = list(Path(slide_dir).glob("*.jpg"))
    if not patch_paths:
        print(f"No patches found in {slide_dir}")
    for patch_path in patch_paths:
        augment_patch(patch_path, output_dir, patch_size, num_patches)
    print(f"Augmented patches saved for slide {slide_id}")

def process_slide_wrapper(args):
    process_slide(*args)

def main(dataset, patch_size, num_patches, overwrite, num_workers):
    base_dir = Path(f"{PATCHES_SAVE_DIR}/{dataset}")
    output_base_dir = Path(f"{AUG_PATCHES_SAVE_DIR}/{dataset}")
    if output_base_dir.exists():
        if overwrite:
            shutil.rmtree(output_base_dir)
            print(f"Overwrite invoked. Deleted existing directory: {output_base_dir}")
        else:
            raise FileExistsError(f"Directory {output_base_dir} already exists. Use --overwrite to delete it.")
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    slide_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir()])
    if not slide_dirs:
        print(f"No slide directories found in {base_dir}")
    else:
        print(f"Found {len(slide_dirs)} slide directories")
    
    args_list = [(slide_dir, output_base_dir, patch_size, num_patches) for slide_dir in slide_dirs]
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        executor.map(process_slide_wrapper, args_list)


# Done
# python aug_b_augment_and_save_patches.py --dataset ouh_batch1_20x --patch_size 224 --num_patches 10 --num_workers 24
# python aug_b_augment_and_save_patches.py --dataset ouh_batch1_40x --patch_size 224 --num_patches 10 --num_workers 24
# python aug_b_augment_and_save_patches.py --dataset ouh_batch2_20x --patch_size 224 --num_patches 10 --num_workers 24
# python aug_b_augment_and_save_patches.py --dataset ouh_batch3_40x --patch_size 224 --num_patches 10 --num_workers 24
# python aug_b_augment_and_save_patches.py --dataset DART_001 --patch_size 224 --num_patches 10 --num_workers 24
# python aug_b_augment_and_save_patches.py --dataset DART_002 --patch_size 224 --num_patches 10 --num_workers 24
# python aug_b_augment_and_save_patches.py --dataset DART_003 --patch_size 224 --num_patches 10 --num_workers 24
# python aug_b_augment_and_save_patches.py --dataset DART_004 --patch_size 224 --num_patches 10 --num_workers 24
# python aug_b_augment_and_save_patches.py --dataset TCIA-CPTAC_test --patch_size 224 --num_patches 10 --num_workers 24
# python aug_b_augment_and_save_patches.py --dataset TCIA-CPTAC --patch_size 224 --num_patches 10 --num_workers 24
# python aug_b_augment_and_save_patches.py --dataset TCGA-lung --patch_size 224 --num_patches 10 --num_workers 24
# python aug_b_augment_and_save_patches.py --dataset CAMELYON16 --patch_size 224 --num_patches 10 --num_workers 24
# TODO:
# 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment and save patches from extracted patches.")
    parser.add_argument("--dataset", type=str, default="ouh_batch1_20x", help="Path to the dataset directory containing extracted patches.")
    parser.add_argument("--patch_size", type=int, default=224, help="Size of the augmented patches.")
    parser.add_argument("--num_patches", type=int, default=10, help="Number of augmented patches to generate.")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Delete the augmented dataset directory if it exists.")
    parser.add_argument("--num_workers", type=int, default=24, help="Number of worker processes to use.")
    args = parser.parse_args()

    print('\n', "-" * 48, '\n')
    print("Arguments passed to the script:")
    print("\nargs - raw:")
    print(sys.argv)
    print("\nargs:")
    print(args)
    print('\n', "-" * 48, '\n')

    main(args.dataset, args.patch_size, args.num_patches, args.overwrite, args.num_workers)
