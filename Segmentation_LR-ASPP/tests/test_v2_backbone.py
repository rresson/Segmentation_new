import importlib
import importlib.util
from pathlib import Path
import sys

import pytest
import torch

from segmentation import Config
from segmentation.models import build_model
from segmentation.checkpoints import export_backbone, load_backbone


def reference_modules():
    path = Path(__file__).resolve().parents[2] / 'segmentation_V2/segmentation'
    if not (path / '__init__.py').is_file():
        pytest.skip('cross-project check requires the sibling segmentation_V2 directory')
    if '_reference_v2' not in sys.modules:
        spec = importlib.util.spec_from_file_location('_reference_v2', path / '__init__.py',
                                                    submodule_search_locations=[str(path)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return (importlib.import_module('_reference_v2.config'),
            importlib.import_module('_reference_v2.models'),
            importlib.import_module('_reference_v2.checkpoints'))


@pytest.mark.parametrize('attention', ['original', 'multiscale', 'lepe'])
@pytest.mark.parametrize('squeeze', [False, True])
def test_same_backbone_features_weights_and_cross_project_checkpoints(tmp_path, attention, squeeze):
    torch.set_num_threads(2)
    config, models, checkpoints = reference_modules()
    reference_cfg = config.Config(stem_pretrained=False, attention_type=attention,
                                  use_learnable_squeeze=squeeze)
    reference = models.build_model(reference_cfg).eval()
    cfg = Config(stem_pretrained=False, seaformer_attention=attention,
                 seaformer_learnable_squeeze=squeeze)
    actual = build_model(cfg).eval()
    source = tmp_path / 'v2.pth'
    checkpoints.export_backbone(reference, reference_cfg, source)
    cfg.backbone_checkpoint = str(source)
    load_backbone(actual, cfg)
    assert actual.backbone.state_dict().keys() == reference.backbone.state_dict().keys()
    assert not hasattr(actual.backbone, 'c3_projection')
    assert [len(actual.backbone.context1), len(actual.backbone.context2), len(actual.backbone.context3)] == [2, 2, 2]
    x = torch.randn(2, 3, 65, 97)
    with torch.no_grad():
        old = reference.backbone(x)
        new = actual.backbone(x)
    for left, right in zip(old, new):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    inputs = []
    hook = actual.decode_head.register_forward_pre_hook(lambda module, args: inputs.append(args[0]))
    with torch.no_grad():
        actual(x)
    hook.remove()
    assert len(inputs[0]) == 2
    torch.testing.assert_close(inputs[0][0], new[0], rtol=0, atol=0)
    torch.testing.assert_close(inputs[0][1], new[2], rtol=0, atol=0)
    exported = tmp_path / 'v3.pth'
    export_backbone(actual, cfg, exported)
    reference_cfg.backbone_checkpoint = str(exported)
    checkpoints.load_backbone(reference, reference_cfg)
    for name, value in actual.backbone.state_dict().items():
        torch.testing.assert_close(value, reference.backbone.state_dict()[name], rtol=0, atol=0)

def test_defaults_match_latest_v2_backbone_config():
    from segmentation.models.seaformer import backbone_network_config
    source_config, _, _ = reference_modules()
    expected = source_config.network_kwargs(source_config.Config())
    actual = backbone_network_config()
    cfg = Config()
    assert cfg.seaformer_attention == expected['attention_type'] == 'original'
    assert cfg.seaformer_learnable_squeeze == expected['use_learnable_squeeze'] is True
    for key in ('dim', 'key_dim', 'num_heads', 'attn_ratio', 'pos_shape', 'use_learnable_squeeze',
                'ffn_ratio', 'expansion_ratio1', 'expansion_ratio2', 'attention_type', 'fusion_type'):
        assert actual[key] == expected[key], key
