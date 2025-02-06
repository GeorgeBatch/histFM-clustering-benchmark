# Choosing a Promising Histopathology Foundation Model by Clustering Augmented Patches

Feature extration and clustering evaluation code in this repository is based on my code from [LC25000-clean](https://github.com/GeorgeBatch/LC25000-clean)

Patch extraction code uses TIAToolbox: https://github.com/TissueImageAnalytics/tiatoolbox

## Environment Setup

Run the commands detailed in the [environment-creation.md](./environment-creation.md) file to create a conda environment with the necessary dependencies.

## Datasets and Feature Extractors

* 5 datasets:
  * 2 private lung cancer datasets: OUH lung, DART lung
  * 2 public lung cancer datasets: TCGA lung, TCIC-CPTAC lung
  * 1 breast cancer dataset: CAMELYON16
* 6 feature extractors:
  * Lung cancer ResNet18 (used four lung cancer datasets) pre-trained by Bin Li for DS-MIL work: https://drive.google.com/drive/folders/1Rn_VpgM82VEfnjiVjDbObbBFHvs0V1OE
  * Breast cancer ResNet18 (used only on the CAMELYON16 dataset) pre-trained by Bin Li for DS-MIL work: https://drive.google.com/drive/folders/14pSKk2rnPJiJsGK2CQJXctP7fhRJZiyn
  * 4 general purpose foundation models: UNI, Prov-GigaPath, Phikon-v2, Virchow v1 concat.

# 1. Extract Patches

**Script**: [aug_a_extract_and_save_patches.py](./aug_a_extract_and_save_patches.py)

Example command:
```shell
python aug_a_extract_and_save_patches.py \
    --dataset TCGA-lung \
    --slide_format svs \
    --patch_size 448 \
    --num_patches 50 \
    --patch_resolution 0.5 \
    --patch_units mpp \
    --num_workers 24
```

# 2. Augment Patches

**Script**: [aug_b_augment_and_save_patches.py](./aug_b_augment_and_save_patches.py)

Example command:
```shell
python aug_b_augment_and_save_patches.py \
    --dataset TCGA-lung \
    --patch_size 224 \
    --num_patches 10 \
    --num_workers 24
```

## 3. Extract Features

**Script**: [aug_c_extract_features.py](./aug_c_extract_features.py)

Used a pre-traned models to extract features from each of the dataset slides (500 images per slide). For each slide, the features are then saved in a `features.npy` file. The mapping of the image index in the `.npy` file to the image path was also saved in `ids_2_img_paths.json`.

Example command:
```shell
python aug_c_extract_features.py \
    --dataset TCGA-lung \
    --img_norm resize_only \
    --extractor_name UNI \
    --device cuda \
    --batch_size 256
```

Notes:
* If file exists, the user will be asked whether to overwrite it.
* The script will also print the progress of the feature extraction.

## 4. Clustering Evaluation

Used scikit-learn KMeans clustering to cluster the features extracted from each of the models. The number of clusters was set to 50. The clustering was done on the features extracted from patches of each of the slides independently.

The features was evaluated by using the the ground truth (we know which augmented patches come from which original patches). The features were evaluated using the following metrics:

* Retrieval metrics to evaluate if the closest images in the feature space are from the same original image
    - precision@1
    - precision@5

* Binary connectivety metrics: two images are considered connected (label 1) if they are in the same ground truth cluster, and disconnected (label 0) otherwise.
    - Confusion Matrix
    - Accuracy
    - Precision
    - Recall
    - F1 Score
    - Specificity
    - Balanced Accuracy

* Clustering metrics: to evaluate the quality of the clustering against the manual clustering
    - Fowlkes-Mallows Index
    - Adjusted Rand Index (ARI)
    - Normalized Mutual Information (NMI)
    - Homogeneity
    - Completeness
    - V-Measure

**Script**: [aug_d_evaluate_clustering.py](./aug_d_evaluate_clustering.py)

Example command:
```shell
python aug_d_evaluate_clustering.py \
    --dataset TCGA-lung \
    --img_norm resize_only \
    --extractor_name UNI
``` 

Other arguments:
- `--overwrite`: Overwrite the existing evaluation results
- `--verbose`: Print the evaluation metrics


## 4. Analysis of the Evaluation Results

To reproduce plots, run the notebook [aug1-analyze-clustering-results.ipynb](./aug1-analyze-clustering-results.ipynb).


## Evaluating a New Model

To evaluate a new model on a new dataset, follow these steps:

1. Download the data.
2. Set up the environment as described in the previous section.
3. Prepare the model in the same format as the other models in [source/feature_extraction/get_model_with_transform.py](./source/feature_extraction/get_model_with_transform.py). The model should inherit from `torch.nn.Module` and have a `forward` method that takes an image tensor and returns a feature tensor. If the model is set-up in a different way, adjust it like shown in [source/feature_extraction/models/owkin_phikon.py](./source/feature_extraction/models/owkin_phikon.py).
4. Sample patches from the slides using the [aug_a_extract_and_save_patches.py](./aug_a_extract_and_save_patches.py) script.
5. Augment the patches using the [aug_b_augment_and_save_patches.py](./aug_b_augment_and_save_patches.py) script.
6. Extract features from augmented patches using a feature extractor model with the prefered normalisation method using the [aug_c_extract_features.py](./aug_c_extract_features.py) script.
7. Evaluate the clustering performance using [aug_d_evaluate_clustering.py](./aug_d_evaluate_clustering.py) script.
8. Analyze the evaluation results using the [aug1-analyze-clustering-results.ipynb](./aug1-analyze-clustering-results.ipynb) notebook.
