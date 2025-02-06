import argparse
import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from pathlib import Path
from source.constants import FEATURE_VECTORS_SAVE_DIR, ALL_IMG_NORMS, ALL_EXTRACTOR_MODELS
from source.eval_utils import compute_connectivity_matrix, compute_clustering_metrics, precision_at_1, precision_at_k

def get_imgpaths_2_intids(ids_2_imgpaths):
    assert len(set(ids_2_imgpaths.values())) == len(list(ids_2_imgpaths.values())), "Can only reverse a bijective mapping, duplicate values found."
    imgpaths_2_intids = {v: int(k) for k, v in ids_2_imgpaths.items()}
    return imgpaths_2_intids

def extract_connectivity_vector(connectivity_matrix):
    connectivity_vector = connectivity_matrix[np.triu_indices(connectivity_matrix.shape[0], k=1)]
    return connectivity_vector

def get_true_connectivity(ids_2_imgpaths):
    imgpaths_2_intids = get_imgpaths_2_intids(ids_2_imgpaths=ids_2_imgpaths)
    true_clusters_dict = {}
    for img_path, int_id in imgpaths_2_intids.items():
        base_name = Path(img_path).stem
        coords = base_name.split('_')[0:2]
        coord_key = f"{coords[0]}_{coords[1]}"
        if coord_key not in true_clusters_dict:
            true_clusters_dict[coord_key] = []
        true_clusters_dict[coord_key].append(img_path)
    num_true_clusters = len(true_clusters_dict)
    num_total_images = sum([len(cluster) for cluster in true_clusters_dict.values()])
    true_cluster_labels = -1 * np.ones(num_total_images, dtype=int)
    for cluster_id, img_paths in enumerate(true_clusters_dict.values()):
        for img_path in img_paths:
            true_cluster_labels[imgpaths_2_intids[img_path]] = cluster_id
    assert np.all(true_cluster_labels != -1)
    true_connectivity_matrix = compute_connectivity_matrix(clusters_dict=true_clusters_dict, imgpaths_2_intids=imgpaths_2_intids)
    true_connectivity_vector = extract_connectivity_vector(true_connectivity_matrix)
    return true_connectivity_matrix, true_connectivity_vector, true_cluster_labels, num_true_clusters

def get_predicted_connectivity(predicted_cluster_labels, ids_2_imgpaths):
    imgpaths_2_intids = get_imgpaths_2_intids(ids_2_imgpaths=ids_2_imgpaths)
    predicted_clusters_dict = {}
    for i, predicted_cluster_label in enumerate(predicted_cluster_labels):
        if predicted_cluster_label not in predicted_clusters_dict:
            predicted_clusters_dict[predicted_cluster_label] = []
        predicted_clusters_dict[predicted_cluster_label].append(ids_2_imgpaths[str(i)])
    predicted_connectivity_matrix = compute_connectivity_matrix(clusters_dict=predicted_clusters_dict, imgpaths_2_intids=imgpaths_2_intids)
    predicted_connectivity_vector = extract_connectivity_vector(predicted_connectivity_matrix)
    return predicted_connectivity_vector

def evaluate_clustering(features_save_dir, verbose):
    features_npy_path = f'{features_save_dir}/features.npy'
    ids_2_imgpaths_json_path = f'{features_save_dir}/ids_2_img_paths.json'
    assert os.path.isfile(features_npy_path), f"File does not exist: \n\t{features_npy_path}"
    assert os.path.isfile(ids_2_imgpaths_json_path), f"File does not exist: \n\t{ids_2_imgpaths_json_path}"
    features = np.load(features_npy_path)
    features = features / np.linalg.norm(features, axis=1, keepdims=True)
    with open(ids_2_imgpaths_json_path, 'r') as f:
        ids_2_imgpaths = json.load(f)
    assert len(set(ids_2_imgpaths.values())) == len(ids_2_imgpaths.values())
    true_connectivity_matrix, true_connectivity_vector, true_cluster_labels, num_true_clusters = get_true_connectivity(ids_2_imgpaths=ids_2_imgpaths)
    if num_true_clusters == 1:
        print(f"Skipping {features_save_dir} - only 1 true cluster")
        return {}
    precision_at_1_value = precision_at_1(features, true_connectivity_matrix, metric='euclidean')
    precision_at_5_value = precision_at_k(features, true_connectivity_matrix, k=5, metric='euclidean')
    kmeans = KMeans(n_clusters=num_true_clusters, random_state=0).fit(features)
    predicted_cluster_labels = kmeans.labels_
    predicted_connectivity_vector = get_predicted_connectivity(predicted_cluster_labels=predicted_cluster_labels, ids_2_imgpaths=ids_2_imgpaths)
    assert predicted_connectivity_vector.shape == true_connectivity_vector.shape
    metrics = compute_clustering_metrics(true_connectivity_vector, predicted_connectivity_vector, true_cluster_labels, predicted_cluster_labels)
    metrics['precision@1'] = precision_at_1_value
    metrics['precision@5'] = precision_at_5_value
    if verbose:
        for metric, value in metrics.items():
            if isinstance(value, int):
                print(f"{metric}: {value}")
            elif isinstance(value, float):
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}:\n {value}")
        print()
    return metrics

# Commands to run:
# python aug_d_evaluate_clustering.py --dataset ouh_batch1_20x
# python aug_d_evaluate_clustering.py --dataset ouh_batch1_40x
# python aug_d_evaluate_clustering.py --dataset ouh_batch2_20x
# python aug_d_evaluate_clustering.py --dataset ouh_batch3_40x
# python aug_d_evaluate_clustering.py --dataset DART_001
# python aug_d_evaluate_clustering.py --dataset DART_002
# python aug_d_evaluate_clustering.py --dataset DART_003
# python aug_d_evaluate_clustering.py --dataset DART_004
# python aug_d_evaluate_clustering.py --dataset TCGA-lung
# python aug_d_evaluate_clustering.py --dataset TCIA-CPTAC
# python aug_d_evaluate_clustering.py --dataset CAMELYON16
def main():
    parser = argparse.ArgumentParser(description='Evaluate extractor-reduction-clustering pipeline.')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name.')
    parser.add_argument('--extractor_name', type=str, default='all', choices=list(ALL_EXTRACTOR_MODELS) + ['all'], help='Feature extractor name.')
    parser.add_argument('--img_norm', type=str, default='all', choices=list(ALL_IMG_NORMS) + ['all'])
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing results.')
    parser.add_argument('--verbose', action='store_true', help='Print metrics to stdout.')
    args = parser.parse_args()

    print('\n', "-" * 48, '\n')
    print("Arguments passed to the script:")
    print("\nargs - raw:")
    print(sys.argv)
    print("\nargs:")
    print(args)
    print('\n', "-" * 48, '\n')

    all_features_save_dir = f"{FEATURE_VECTORS_SAVE_DIR}/{args.dataset}"
    extractor_names = [args.extractor_name] if args.extractor_name != 'all' else os.listdir(all_features_save_dir)
    eval_results_dir = 'eval_results'
    filename_base = f"dataset={args.dataset}#extractor_name=all#img_norm=all#distance_metric=cosine#dimensionality_reduction=none#clustering=kmeans"
    try:
        with open(f"{eval_results_dir}/{filename_base}.json", 'r') as f:
            all_metrics = json.load(f)
        print(f"Will append to existing file {eval_results_dir}/{filename_base}.json", end='\n\n')
    except FileNotFoundError:
        print(f"File {eval_results_dir}/{filename_base}.json not found, initializing with empty dictionary.", end='\n\n')
        all_metrics = {}
    for extractor_name in extractor_names:
        img_norms = [args.img_norm] if args.img_norm != 'all' else os.listdir(f"{all_features_save_dir}/{extractor_name}")
        for img_norm in img_norms:
            features_save_dir = f"{all_features_save_dir}/{extractor_name}/{img_norm}"
            slide_dirs = sorted([d for d in Path(features_save_dir).iterdir() if d.is_dir()])
            for idx, slide_dir in enumerate(slide_dirs, start=1):
                slide_id = slide_dir.name
                slide_features_save_dir = f"{features_save_dir}/{slide_id}"
                combo_key = f"{extractor_name}#{img_norm}#{slide_id}"
                if (combo_key in all_metrics) and (not args.overwrite):
                    print(f"{idx}/{len(slide_dirs)} Skipping {combo_key} - already computed")
                    continue
                else:
                    try:
                        print(f"\n{idx}/{len(slide_dirs)} Computing {combo_key} ...\n")
                        current_metrics = evaluate_clustering(
                            features_save_dir=slide_features_save_dir,
                            verbose=args.verbose,
                        )
                        all_metrics[combo_key] = current_metrics
                        with open(f"{eval_results_dir}/{filename_base}.json", 'w') as f:
                            json.dump(all_metrics, f, indent=4)
                    except ValueError as e:
                        if "Input contains NaN" in str(e):
                            print(f"ValueError processing {slide_dir}: {e}")
                            continue
                        else:
                            raise e
    df = pd.DataFrame(all_metrics).T
    df.sort_index(inplace=True)
    df.to_csv(f"{eval_results_dir}/{filename_base}.csv")

if __name__ == "__main__":
    main()
