import pytest
from torch import nn

from source.feature_extraction.get_model_with_transform import get_feature_extractor


@pytest.mark.parametrize("model_name", [
    # natural image models
    'imagenet_resnet18-last-layer',
    'imagenet_resnet50-clam-extractor',
    'dinov2_vits14',
    'dinov2_vitb14',
    # pathology-specific models
    'UNI',
    'prov-gigapath',
    'owkin-phikon',
    'owkin-phikon-v2',
    'virchow-v1-cls_token',
    'virchow-v1-mean_patch_tokens',
    'virchow-v1-concat',
    'virchow-v2-cls_token',
    'virchow-v2-mean_patch_tokens',
    'virchow-v2-concat',  
    'simclr-tcga-lung_resnet18-2.5x',
    'simclr-tcga-lung_resnet18-10x',
    'simclr-camelyon16_resnet18-5x',
    'simclr-camelyon16_resnet18-20x',
])
def test_get_feature_extractor(model_name):
    extractor = get_feature_extractor(model_name)
    assert isinstance(extractor, nn.Module)

    if model_name == 'imagenet_resnet18-last-layer':
        assert isinstance(extractor.fc, nn.Identity)
