from dataclasses import asdict, replace
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch import nn

from profile_model import profile_model
from segmentation import Config
from segmentation.config import prepare_config
from segmentation.checkpoints import export_backbone, load_backbone
from segmentation.models import build_model
from segmentation.models.mini_head import MiniC3Head
from segmentation.models.mobile_v3_blocks import InvertedResidualV3, SqueezeExcitation
from helpers import reference_modules


def test_default_is_exactly_one_ir_se_context_c3_only_and_under_one_million():
    model = build_model(Config(stem_pretrained=False)).eval()
    head = model.decode_head
    assert isinstance(head, MiniC3Head)
    assert len(head.context) == 1
    block = head.context[0]
    assert isinstance(block, InvertedResidualV3) and block.use_residual
    assert isinstance(block.se, SqueezeExcitation)
    assert block.expand[0].in_channels == block.project[0].out_channels == 128
    assert block.expand[0].out_channels == block.project[0].in_channels == 368
    assert block.depthwise[0].kernel_size == (5, 5)
    assert block.se.reduce.out_channels == 96
    assert set(dict(head.named_children())) == {'input_projection', 'context', 'lateral', 'refine', 'classifier'}
    stats = profile_model(model, torch.zeros(1,3,256,256))
    assert stats['parameters'] == 997178
    assert stats['backbone_parameters'] == 757860
    assert stats['head_parameters'] == 239318
    assert stats['total_counted_macs'] == 427811840
    assert stats['head_input_shapes'] == [[1,64,32,32]]
    assert stats['output_shape'] == [1,150,16,16]
    assert not any(m._forward_hooks for m in model.modules())
    output = model(torch.randn(2,3,65,97))
    assert output.shape == (2,150,5,7)
    output.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    for key in ('context.0.expand.0.weight','context.0.se.reduce.weight','context.0.se.expand.weight','context.0.project.0.weight'):
        assert dict(head.named_parameters())[key].grad.norm() > 0
    with pytest.raises(ValueError, match='one C3'):
        head([torch.zeros(1,64,8,8)] * 3)


@pytest.mark.parametrize('width,expanded,expected', [
    (64,72,796610), (64,96,801554), (64,112,806650), (64,128,810202),
    (64,512,993850), (64,520,1005490), (96,448,999338),
    (128,352,983890), (128,368,997178), (128,376,1001002),
])
def test_expansion_configuration_controls_real_parameter_count(width, expanded, expected):
    cfg = prepare_config(Config(stem_pretrained=False, head_width=width, context_expanded_channels=expanded))
    model = build_model(cfg)
    assert sum(p.numel() for p in model.parameters()) == expected
    assert model.decode_head.context[0].expand[0].out_channels == expanded


@pytest.mark.parametrize('overrides', [dict(context_expanded_channels=64), dict(context_expanded_channels=121),
                                        dict(head_width=65), dict(head_width=True), dict(context_expanded_channels=128),
                                        dict(head_name='mobile_c3'),
                                        dict(seaformer_depths=(1,2,2)), dict(backbone_name='mobile_v3')])
def test_mini_rejects_silent_architecture_changes(overrides):
    with pytest.raises(ValueError):
        prepare_config(Config(**overrides))
    with pytest.raises(ValueError):
        build_model(Config(**overrides), initialize_stem=False)


@pytest.mark.parametrize('attention', ['original', 'multiscale', 'lepe'])
@pytest.mark.parametrize('squeeze', [False, True])
def test_backbone_is_identical_and_transfers_with_both_projects(tmp_path, attention, squeeze):
    cfg = Config(stem_pretrained=False, seaformer_attention=attention, seaformer_learnable_squeeze=squeeze)
    mini = build_model(cfg).eval()
    for project in ('segmentation_V2','segmentation_V2_V3'):
        configs, models, checkpoints = reference_modules(project)
        options = (dict(attention_type=attention,use_learnable_squeeze=squeeze) if project == 'segmentation_V2'
                   else dict(seaformer_attention=attention,seaformer_learnable_squeeze=squeeze))
        source_cfg = configs.Config(stem_pretrained=False, **options)
        source = models.build_model(source_cfg).eval()
        weights = tmp_path / (project + '.pth')
        checkpoints.export_backbone(source, source_cfg, weights)
        load_backbone(mini, replace(cfg,backbone_checkpoint=str(weights)))
        images = torch.randn(2,3,65,97)
        with torch.no_grad():
            expected, actual = source.backbone(images), mini.backbone(images)
        for left, right in zip(expected, actual):
            torch.testing.assert_close(left,right,rtol=0,atol=0)
        exported = tmp_path / 'mini_backbone.pth'
        export_backbone(mini,cfg,exported)
        checkpoints.load_backbone(source, replace(source_cfg,backbone_checkpoint=str(exported)))
        for key,value in mini.backbone.state_dict().items():
            torch.testing.assert_close(value,source.backbone.state_dict()[key],rtol=0,atol=0)
        full = tmp_path / 'old_full.pth'
        torch.save({'config':asdict(source_cfg),'model':source.state_dict()},full)
        # Full old model is valid as a backbone source, but never as mini head initialization.
        load_backbone(mini,replace(cfg,backbone_checkpoint=str(full)))
        with pytest.raises(ValueError, match='head_name'):
            load_backbone(mini,replace(cfg,model_checkpoint=str(full)))


def test_no_torchvision_and_profile_cli_supports_expansion(tmp_path):
    project = Path(__file__).resolve().parents[1]
    script = "import sys; sys.modules['torchvision']=None; import main, pipeline, profile_model; from segmentation import Config; from segmentation.models import build_model; print(sum(p.numel() for p in build_model(Config(stem_pretrained=False)).parameters()))"
    env = {**os.environ, 'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','PYTHONDONTWRITEBYTECODE':'1'}
    imported = subprocess.run([sys.executable,'-c',script],cwd=project,text=True,capture_output=True,env=env,timeout=30)
    assert imported.returncode == 0, imported.stderr
    assert imported.stdout.strip().endswith('997178')
    output = tmp_path/'profile.json'
    result = subprocess.run([sys.executable,str(project/'profile_model.py'),'--size','256','--mode','head',
                             '--head-width','64','--context-expanded-channels','112','--output',str(output)],cwd=tmp_path,
                            text=True,capture_output=True,env=env,timeout=30)
    assert result.returncode == 0,result.stdout+result.stderr
    import json
    report=json.loads(output.read_text())
    assert report['parameters'] == 806650 and report['trainable_parameters'] == 48790

def test_defaults_match_latest_v2_backbone_config():
    from segmentation.models.seaformer import backbone_network_config
    source_config, _, _ = reference_modules('segmentation_V2')
    expected = source_config.network_kwargs(source_config.Config())
    actual = backbone_network_config()
    cfg = Config()
    assert cfg.seaformer_attention == expected['attention_type'] == 'original'
    assert cfg.seaformer_learnable_squeeze == expected['use_learnable_squeeze'] is True
    for key in ('dim', 'key_dim', 'num_heads', 'attn_ratio', 'pos_shape', 'use_learnable_squeeze',
                'ffn_ratio', 'expansion_ratio1', 'expansion_ratio2', 'attention_type', 'fusion_type'):
        assert actual[key] == expected[key], key
