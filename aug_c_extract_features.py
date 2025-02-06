import argparse
import os
import sys
import json
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from pathlib import Path
from source.constants import ALL_IMG_NORMS, ALL_EXTRACTOR_MODELS
from source.constants import AUG_PATCHES_SAVE_DIR, FEATURE_VECTORS_SAVE_DIR, DATASET_SPECIFIC_NORMALIZATION_CONSTANTS_PATH
from source.feature_extraction.data import FeatureExtractionDataset, get_data_transform
from source.feature_extraction.get_model_with_transform import get_feature_extractor

def prepare_directories(dataset, extractor_name, img_norm):
    base_dir = Path(f"{AUG_PATCHES_SAVE_DIR}/{dataset}")
    feature_vectors_dir = Path(f"{FEATURE_VECTORS_SAVE_DIR}/{dataset}/{extractor_name}/{img_norm}")
    feature_vectors_dir.mkdir(parents=True, exist_ok=True)
    return base_dir, feature_vectors_dir

def make_pytorch_dataset(img_dir, data_transform):
    return FeatureExtractionDataset(
        img_dir=img_dir,
        transform=data_transform,
        img_ext='jpg',
        return_image_details=True,
    )

def make_pytorch_dataloader(dataset, batch_size):
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )

def prepare_feature_extractor(extractor_name, device):
    feature_extractor = get_feature_extractor(extractor_name)
    feature_extractor.eval()
    feature_extractor = feature_extractor.to(device)
    if device == 'cpu':
        print("CPU mode")
    else:
        if torch.cuda.device_count() == 1:
            print("Single GPU mode")
        elif torch.cuda.device_count() > 1:
            if device == 'cuda':
                print("Multiple GPU mode")
                feature_extractor = nn.DataParallel(feature_extractor)
            else:
                print("Single GPU mode")
        else:
            raise NotImplementedError
        print()
    return feature_extractor

def extract_features(feature_extractor, dataloader, device):
    current_ids_list = []
    current_features_list = []
    current_paths_list = []
    for batch in tqdm(dataloader):
        inputs = batch['image'].to(device)
        details = {detail: batch[detail] for detail in batch if detail.startswith('image_')}
        ids = details['image_id'].numpy()
        current_ids_list.append(ids)
        paths = details['image_path']
        current_paths_list.extend(paths)
        with torch.no_grad():
            features = feature_extractor(inputs).cpu().numpy()
        current_features_list.append(features)
    
    if len(current_features_list) == 0:
        raise ValueError("No features extracted. Ensure the dataset is not empty and the feature extractor is working correctly.")
    
    current_features_numpy_array = np.concatenate(current_features_list, axis=0)
    current_ids_numpy_array = np.concatenate(current_ids_list, axis=0)
    current_ids_2_img_paths = {str(i): current_paths_list[i] for i in current_ids_numpy_array}
    return {
        'features': current_features_numpy_array,
        'ids': current_ids_numpy_array,
        'ids_2_img_paths': current_ids_2_img_paths
    }

def save_features(contents, paths):
    np.save(paths['ids'], contents['ids'])
    np.save(paths['features'], contents['features'])
    with open(paths['ids_2_img_paths'], "w") as f:
        json.dump(contents['ids_2_img_paths'], f, indent=4)

# -----------------------------------------------------------------------------------------------
# gpu_short
# -----------------------------------------------------------------------------------------------

# ouh_batch1_20x for models: UNI, prov-gigapath, owkin-phikon-v2, virchow-v1-concat, simclr-tcga-lung_resnet18-10x
# Done
# python aug_c_extract_features.py --dataset ouh_batch1_20x --img_norm imagenet --extractor_name UNI --batch_size 256
# python aug_c_extract_features.py --dataset ouh_batch1_20x --img_norm imagenet --extractor_name prov-gigapath --batch_size 256
# python aug_c_extract_features.py --dataset ouh_batch1_20x --img_norm imagenet --extractor_name owkin-phikon-v2 --batch_size 256
# python aug_c_extract_features.py --dataset ouh_batch1_20x --img_norm imagenet --extractor_name virchow-v1-concat --batch_size 256
# python aug_c_extract_features.py --dataset ouh_batch1_20x --img_norm resize_only --extractor_name simclr-tcga-lung_resnet18-10x --batch_size 256
# python aug_c_extract_features.py --dataset ouh_batch1_20x --img_norm imagenet --extractor_name simclr-tcga-lung_resnet18-10x --batch_size 256 # job 25704684 (done)
# TODO:

# ouh_batch1_40x for models: UNI, prov-gigapath, owkin-phikon-v2, virchow-v1-concat, simclr-tcga-lung_resnet18-10x
# Done
# python aug_c_extract_features.py --dataset ouh_batch1_40x --img_norm imagenet --extractor_name UNI --batch_size 256
# python aug_c_extract_features.py --dataset ouh_batch1_40x --img_norm imagenet --extractor_name prov-gigapath --batch_size 256 # (job job 25605260: ouh_batch1_40x, prov-gigapath)
# python aug_c_extract_features.py --dataset ouh_batch1_40x --img_norm imagenet --extractor_name owkin-phikon-v2 --batch_size 25 # (job 25605309)
# python aug_c_extract_features.py --dataset ouh_batch1_40x --img_norm imagenet --extractor_name virchow-v1-concat --batch_size 256 # (job 25605311)
# python aug_c_extract_features.py --dataset ouh_batch1_40x --img_norm resize_only --extractor_name simclr-tcga-lung_resnet18-10x --batch_size 256 # (job 25605313)
# python aug_c_extract_features.py --dataset ouh_batch1_40x --img_norm imagenet --extractor_name simclr-tcga-lung_resnet18-10x --batch_size 256 # job 25704732 (done)
# TODO:

# ouh_batch2_20x for models: UNI, prov-gigapath, owkin-phikon-v2, virchow-v1-concat, simclr-tcga-lung_resnet18-10x
# Done
# python aug_c_extract_features.py --dataset ouh_batch2_20x --img_norm imagenet --extractor_name UNI --batch_size 256 # (job 25605340 failed), (job 25605358)
# python aug_c_extract_features.py --dataset ouh_batch2_20x --img_norm imagenet --extractor_name prov-gigapath --batch_size 256 # (job 25605343 failed), (job 25605362)
# python aug_c_extract_features.py --dataset ouh_batch2_20x --img_norm imagenet --extractor_name owkin-phikon-v2 --batch_size 256 # (job 25605348 failed), (job 25605369)
# python aug_c_extract_features.py --dataset ouh_batch2_20x --img_norm imagenet --extractor_name virchow-v1-concat --batch_size 256 # (job 25605373)
# python aug_c_extract_features.py --dataset ouh_batch2_20x --img_norm resize_only --extractor_name simclr-tcga-lung_resnet18-10x --batch_size 256 # (job 25605375)
# python aug_c_extract_features.py --dataset ouh_batch2_20x --img_norm imagenet --extractor_name simclr-tcga-lung_resnet18-10x --batch_size 256 # job 25704753 (done)
# TODO:

# ouh_batch3_40x for models: UNI, prov-gigapath, owkin-phikon-v2, virchow-v1-concat, simclr-tcga-lung_resnet18-10x
# Done:
# python aug_c_extract_features.py --dataset ouh_batch3_40x --img_norm imagenet --extractor_name UNI --batch_size 256 # job 25605396
# python aug_c_extract_features.py --dataset ouh_batch3_40x --img_norm imagenet --extractor_name prov-gigapath --batch_size 256 # job 25605400
# python aug_c_extract_features.py --dataset ouh_batch3_40x --img_norm imagenet --extractor_name owkin-phikon-v2 --batch_size 256 # job 25605403
# python aug_c_extract_features.py --dataset ouh_batch3_40x --img_norm imagenet --extractor_name virchow-v1-concat --batch_size 256 # job 25605710
# python aug_c_extract_features.py --dataset ouh_batch3_40x --img_norm resize_only --extractor_name simclr-tcga-lung_resnet18-10x --batch_size 256 # job 25606011
# python aug_c_extract_features.py --dataset ouh_batch3_40x --img_norm imagenet --extractor_name simclr-tcga-lung_resnet18-10x --batch_size 256 # job 25704781 (done)
# TODO:

# DART_001 for models: UNI, prov-gigapath, owkin-phikon-v2, virchow-v1-concat, simclr-tcga-lung_resnet18-10x
# Done:
# python aug_c_extract_features.py --dataset DART_001 --img_norm imagenet --extractor_name UNI --batch_size 256 # job 25606155
# python aug_c_extract_features.py --dataset DART_001 --img_norm imagenet --extractor_name prov-gigapath --batch_size 256 # job 25606177
# python aug_c_extract_features.py --dataset DART_001 --img_norm imagenet --extractor_name owkin-phikon-v2 --batch_size 256 # job 25606183
# python aug_c_extract_features.py --dataset DART_001 --img_norm imagenet --extractor_name virchow-v1-concat --batch_size 256 # job 25606184
# python aug_c_extract_features.py --dataset DART_001 --img_norm resize_only --extractor_name simclr-tcga-lung_resnet18-10x --batch_size 256 # 25606186
# python aug_c_extract_features.py --dataset DART_001 --img_norm imagenet --extractor_name simclr-tcga-lung_resnet18-10x --batch_size 256 # 25704805 (done)
# TODO:

# DART_002 for models: UNI, prov-gigapath, owkin-phikon-v2, virchow-v1-concat, simclr-tcga-lung_resnet18-10x
# Done:
# python aug_c_extract_features.py --dataset DART_002 --img_norm imagenet --extractor_name UNI --batch_size 256 # job 25606219
# python aug_c_extract_features.py --dataset DART_002 --img_norm imagenet --extractor_name prov-gigapath --batch_size 256 # job 25606222
# python aug_c_extract_features.py --dataset DART_002 --img_norm imagenet --extractor_name owkin-phikon-v2 --batch_size 256 # job 25606223
# python aug_c_extract_features.py --dataset DART_002 --img_norm imagenet --extractor_name virchow-v1-concat --batch_size 256 # job 25606227
# python aug_c_extract_features.py --dataset DART_002 --img_norm resize_only --extractor_name simclr-tcga-lung_resnet18-10x --batch_size 256 # job 25606230
# python aug_c_extract_features.py --dataset DART_002 --img_norm imagenet --extractor_name simclr-tcga-lung_resnet18-10x --batch_size 256 # 25704843 (done)
# TODO:

# DART_004 for models: UNI, prov-gigapath, owkin-phikon-v2, virchow-v1-concat, simclr-tcga-lung_resnet18-10x
# Done:
# python aug_c_extract_features.py --dataset DART_004 --img_norm imagenet --extractor_name UNI --batch_size 256 # job 25606239
# python aug_c_extract_features.py --dataset DART_004 --img_norm imagenet --extractor_name prov-gigapath --batch_size 256 # job 25606243
# python aug_c_extract_features.py --dataset DART_004 --img_norm imagenet --extractor_name owkin-phikon-v2 --batch_size 256 # job 25606244
# python aug_c_extract_features.py --dataset DART_004 --img_norm imagenet --extractor_name virchow-v1-concat --batch_size 256 # job 25606249
# python aug_c_extract_features.py --dataset DART_004 --img_norm resize_only --extractor_name simclr-tcga-lung_resnet18-10x --batch_size 256 # job 25606257
# python aug_c_extract_features.py --dataset DART_004 --img_norm imagenet --extractor_name simclr-tcga-lung_resnet18-10x --batch_size 256 # 25704882 (done)
# TODO:

# DART_003 for models: UNI, prov-gigapath, owkin-phikon-v2, virchow-v1-concat, simclr-tcga-lung_resnet18-10x
# Done:
# python aug_c_extract_features.py --dataset DART_003 --img_norm imagenet --extractor_name UNI --batch_size 256 # job 25606669 (failed), job 25615104
# python aug_c_extract_features.py --dataset DART_003 --img_norm imagenet --extractor_name prov-gigapath --batch_size 256 # job 25606870 (failed), job 25615863
# python aug_c_extract_features.py --dataset DART_003 --img_norm imagenet --extractor_name owkin-phikon-v2 --batch_size 256 # job 25606874 (failed), job 25615930
# python aug_c_extract_features.py --dataset DART_003 --img_norm imagenet --extractor_name virchow-v1-concat --batch_size 256 # job 25606876 (failed), job 25615993
# python aug_c_extract_features.py --dataset DART_003 --img_norm resize_only --extractor_name simclr-tcga-lung_resnet18-10x --batch_size 256 # job 25606913 (failed), job 25616028
# python aug_c_extract_features.py --dataset DART_003 --img_norm imagenet --extractor_name simclr-tcga-lung_resnet18-10x --batch_size 256 # 25704942 (done)
# TODO:

# CAMELYON16 for models: UNI, prov-gigapath, owkin-phikon-v2, virchow-v1-concat, simclr-camelyon16_resnet18-20x
# Done:
# python aug_c_extract_features.py --dataset CAMELYON16 --img_norm imagenet --extractor_name UNI --batch_size 256 # job (25635964 failed), 25641098
# python aug_c_extract_features.py --dataset CAMELYON16 --img_norm imagenet --extractor_name prov-gigapath --batch_size 256 # job (25635965 failed), 25641108
# python aug_c_extract_features.py --dataset CAMELYON16 --img_norm imagenet --extractor_name owkin-phikon-v2 --batch_size 256 # job (25635966 failed), 25641112
# python aug_c_extract_features.py --dataset CAMELYON16 --img_norm imagenet --extractor_name virchow-v1-concat --batch_size 256 # job (25635967 failed), 25641119
# CUDA_VISIBLE_DEVICES=0 python aug_c_extract_features.py --dataset CAMELYON16 --img_norm resize_only --extractor_name simclr-camelyon16_resnet18-20x --batch_size 256 # (done) in tmux comp-aug-camelyon16-features on compg023
# python aug_c_extract_features.py --dataset CAMELYON16 --img_norm imagenet --extractor_name simclr-camelyon16_resnet18-20x --batch_size 256 # 25705080 (done)
# TODO:

# -----------------------------------------------------------------------------------------------
# gpu_long
# -----------------------------------------------------------------------------------------------

# TCIA-CPTAC for models: UNI, prov-gigapath, owkin-phikon-v2, virchow-v1-concat, simclr-tcga-lung_resnet18-10x
# Done:
# python aug_c_extract_features.py --dataset TCIA-CPTAC --img_norm imagenet --extractor_name UNI --batch_size 256 # job 25606997 (canceled), job 25616155 (done)
# python aug_c_extract_features.py --dataset TCIA-CPTAC --img_norm imagenet --extractor_name prov-gigapath --batch_size 256 # job 25607064 (canceled), job 25616190 (done)
# python aug_c_extract_features.py --dataset TCIA-CPTAC --img_norm imagenet --extractor_name owkin-phikon-v2 --batch_size 256 # job 25607178 (canceled), job 25616225 (done)
# python aug_c_extract_features.py --dataset TCIA-CPTAC --img_norm imagenet --extractor_name virchow-v1-concat --batch_size 256 # job 25607280 (canceled), job 25616227 (done)
# python aug_c_extract_features.py --dataset TCIA-CPTAC --img_norm resize_only --extractor_name simclr-tcga-lung_resnet18-10x --batch_size 256 # job 25607342 (canceled), job 25616254 (done)
# python aug_c_extract_features.py --dataset TCIA-CPTAC --img_norm imagenet --extractor_name simclr-tcga-lung_resnet18-10x --batch_size 256 # 25705491 (done)
# TODO:

# TCGA-lung for models: UNI, prov-gigapath, owkin-phikon-v2, virchow-v1-concat, simclr-tcga-lung_resnet18-10x
# Done:
# python aug_c_extract_features.py --dataset TCGA-lung --img_norm imagenet --extractor_name UNI --batch_size 256 # job 25607398 (canceled), job 25616276 (done)
# python aug_c_extract_features.py --dataset TCGA-lung --img_norm imagenet --extractor_name prov-gigapath --batch_size 256 # job 25607438 (canceled), job 25616278 (done)
# python aug_c_extract_features.py --dataset TCGA-lung --img_norm imagenet --extractor_name owkin-phikon-v2 --batch_size 256 # job 25607515 (canceled), job 25616303 (done)
# python aug_c_extract_features.py --dataset TCGA-lung --img_norm imagenet --extractor_name virchow-v1-concat --batch_size 256 # job 25607573 (canceled), job 25616332 (done)
# python aug_c_extract_features.py --dataset TCGA-lung --img_norm resize_only --extractor_name simclr-tcga-lung_resnet18-10x --batch_size 256 # job 25607630 (canceled), job 25616333 (done)
# python aug_c_extract_features.py --dataset TCGA-lung --img_norm imagenet --extractor_name simclr-tcga-lung_resnet18-10x --batch_size 256 # 25705241 (done)
# TODO:

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name.")
    parser.add_argument("--img_norm", type=str, default='resize_only', help="Image normalization type.", choices=ALL_IMG_NORMS)
    parser.add_argument("--extractor_name", type=str, default='UNI', help="Feature extractor name.", choices=ALL_EXTRACTOR_MODELS)
    parser.add_argument("--device", type=str, default='cuda', help="Device to use. 'cpu' or 'cuda' or 'cuda:<INDEX>'.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for feature extraction.")
    parser.add_argument("--resume", action='store_true', default=False, help="Resume feature extraction by skipping already processed slides.")
    args = parser.parse_args()

    print('\n', "-" * 48, '\n')
    print("Arguments passed to the script:")
    print("\nargs - raw:")
    print(sys.argv)
    print("\nargs:")
    print(args)
    print('\n', "-" * 48, '\n')

    device = args.device if torch.cuda.is_available() else 'cpu'
    base_dir, feature_vectors_dir = prepare_directories(dataset=args.dataset, extractor_name=args.extractor_name, img_norm=args.img_norm)

    try:
        data_transform = get_data_transform(img_norm=args.img_norm)
    except KeyError as e:
        print(f"Key {e} not found in either constants_zoo of `data.get_norm_constants()` or data-specific transforms in {DATASET_SPECIFIC_NORMALIZATION_CONSTANTS_PATH}")
        raise

    feature_extractor = prepare_feature_extractor(extractor_name=args.extractor_name, device=device)

    slide_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir()])
    for idx, slide_dir in enumerate(slide_dirs, start=1):
        slide_id = slide_dir.name
        slide_feature_dir = feature_vectors_dir / slide_id
        features_save_paths = {
            'ids': slide_feature_dir / 'ids.npy',
            'ids_2_img_paths': slide_feature_dir / 'ids_2_img_paths.json',
            'features': slide_feature_dir / 'features.npy'
        }
        if args.resume and slide_feature_dir.exists() and all(path.exists() for path in features_save_paths.values()):
            print(f"{idx}/{len(slide_dirs)} Skipping slide: {slide_id} (already processed)")
            continue
        print(f"{idx}/{len(slide_dirs)} Processing slide: {slide_id}") # start enumerate with 1
        dataset = make_pytorch_dataset(img_dir=slide_dir, data_transform=data_transform)
        if len(dataset) == 0:
            print(f"Skipping {slide_id} as it has no images.\n")
            continue
        dataloader = make_pytorch_dataloader(dataset=dataset, batch_size=args.batch_size)
        
        try:
            print(f"Extracting features using device: {device} ...")
            features_and_info = extract_features(feature_extractor=feature_extractor, dataloader=dataloader, device=device)
        except ValueError as e:
            print(e, "\n")
            continue
        
        slide_feature_dir.mkdir(parents=True, exist_ok=True)
        save_features(contents=features_and_info, paths=features_save_paths)
        print("Files saved.\n")