from os import path as osp

PROJECT_PATH = osp.abspath(
    osp.join(osp.dirname(osp.realpath(__file__)), '../'))

RANDOM_SEED = 42

DATASET_SPECIFIC_NORMALIZATION_CONSTANTS_PATH = osp.join(PROJECT_PATH, 'source/feature_extraction/img_normalisation_constants.json')

PATCHES_SAVE_DIR = osp.join(PROJECT_PATH, 'datasets')
AUG_PATCHES_SAVE_DIR = osp.join(PROJECT_PATH, 'aug_datasets')
FEATURE_VECTORS_SAVE_DIR = osp.join(PROJECT_PATH, 'aug_feature_vectors')

ALL_IMG_NORMS = (
    'imagenet',
    'openai_clip',
    'uniform',
    'resize_only',
)

ALL_EXTRACTOR_MODELS = (
    # Natural Images
    'imagenet_resnet18-last-layer',
    'imagenet_resnet50-clam-extractor',
    'dinov2_vits14',  # ViT sizes: small -> base -> large -> giant
    'dinov2_vitb14',
    # Pathology Images
    'UNI',
    'prov-gigapath',
    'owkin-phikon',
    'owkin-phikon-v2',
    "virchow-v1-cls_token",
    "virchow-v1-mean_patch_tokens",
    "virchow-v1-concat",
    "virchow-v2-cls_token",
    "virchow-v2-mean_patch_tokens",
    "virchow-v2-concat",
    'simclr-tcga-lung_resnet18-10x',
    'simclr-tcga-lung_resnet18-2.5x',
    'simclr-camelyon16_resnet18-20x',
    'simclr-camelyon16_resnet18-5x',
)

ORIGINAL_2_PRETTY_MODEL_NAMES = {
    'UNI': 'UNI',
    'prov-gigapath': 'Prov-GigaPath',
    'owkin-phikon': 'Phikon',
    'owkin-phikon-v2': 'Phikon-v2',
    'virchow-v1-cls_token': 'Virchow-v1-CLS',
    'virchow-v1-mean_patch_tokens': 'Virchow-v1-Mean',
    'virchow-v1-concat': 'Virchow-v1-Concat',
    'virchow-v2-cls_token': 'Virchow-v2-CLS',
    'virchow-v2-mean_patch_tokens': 'Virchow-v2-Mean',
    'virchow-v2-concat': 'Virchow-v2-Concat',
    'dinov2_vits14': 'DINOv2-ViT-S/14',
    'dinov2_vitb14': 'DINOv2-ViT-B/14',
    'simclr-tcga-lung_resnet18-10x': 'ResNet18-lung-10x',
    'simclr-tcga-lung_resnet18-2.5x': 'ResNet18-lung-2.5x',
    'simclr-camelyon16_resnet18-20x': 'ResNet18-camelyon16-20x',
    'simclr-camelyon16_resnet18-5x': 'ResNet18-camelyon16-5x',
    'imagenet_resnet50-clam-extractor': 'ResNet50-CLAM',
    'imagenet_resnet18-last-layer': 'ResNet18',
}

EXTRACTOR_NAMES_2_WEIGHTS_PATHS = {
    'simclr-tcga-lung_resnet18-10x': osp.join(
        PROJECT_PATH, 'weights/simclr-tcga-lung/weights-10x/model-v1.pth'),
    'simclr-tcga-lung_resnet18-2.5x': osp.join(
        PROJECT_PATH, 'weights/simclr-tcga-lung/weights-2.5x/model-v1.pth'),
    'simclr-camelyon16_resnet18-20x': osp.join(
        PROJECT_PATH, 'weights/simclr-camelyon16/weights-20x/model-v2.pth'),
    'simclr-camelyon16_resnet18-5x': osp.join(
        PROJECT_PATH, 'weights/simclr-camelyon16/weights-5x/model.pth'),
}
