from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

from segmentation import Config
from segmentation.checkpoints import load_backbone
from segmentation.config import prepare_config
from segmentation.models import build_model
from segmentation.run_paths import prepare_output_dir
from test_v2_backbone import reference_modules


def allocate(config):
    prepared = prepare_config(config)
    output = prepare_output_dir(prepared, explicit_output_dir=bool(config.output_dir))
    return prepared, output


def test_custom_from_scratch_and_explicit_output_never_overwrite(tmp_path):
    cfg = Config(output_root=str(tmp_path))
    assert allocate(cfg)[1] == tmp_path / 'run1/custom'
    assert allocate(cfg)[1] == tmp_path / 'run2/custom'
    explicit = replace(cfg, output_dir=str(tmp_path / 'legacy_custom'))
    first = allocate(explicit)[1]
    (first / 'keep.txt').write_text('keep old training')
    assert allocate(explicit)[1] == tmp_path / 'legacy_custom_2'
    assert (first / 'keep.txt').read_text() == 'keep old training'
    with pytest.raises(ValueError, match='run_name'):
        allocate(replace(cfg, run_name='../escape'))


def test_head_auto_uses_latest_pretrain_and_resume_uses_original_directory(tmp_path):
    source = tmp_path / 'run1/pretrain/backbone_best_miou.pth'
    source.parent.mkdir(parents=True)
    source.write_bytes(b'weights')
    cfg = Config(mode='head', output_root=str(tmp_path))
    prepared, output = allocate(cfg)
    assert output == tmp_path / 'run1/head1'
    assert Path(prepared.backbone_checkpoint) == source
    assert allocate(cfg)[1] == tmp_path / 'run1/head2'
    (tmp_path / 'run2/pretrain').mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match='run2'):
        allocate(cfg)
    # 续训依据原 checkpoint，不受尚未完成的新 run 影响。
    resume = output / 'last.pth'
    resume.write_bytes(b'state')
    assert allocate(replace(cfg, resume=str(resume)))[1] == output
    assert not (tmp_path / 'run1/head3').exists()


def test_full_model_initialization_is_strict_and_does_not_download(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    cfg = Config(stem_pretrained=False)
    original = build_model(cfg)
    source = tmp_path / 'complete.pth'
    payload = {'config': asdict(cfg), 'model': original.state_dict()}
    torch.save(payload, source)

    def forbidden_download(*args, **kwargs):
        raise AssertionError('A complete checkpoint must not download ImageNet weights')

    monkeypatch.setattr(torch.hub, 'load_state_dict_from_url', forbidden_download)
    initialized_cfg = replace(cfg, model_checkpoint=str(source), stem_pretrained=True)
    initialized = build_model(initialized_cfg)
    load_backbone(initialized, initialized_cfg)
    for name, value in original.state_dict().items():
        torch.testing.assert_close(value, initialized.state_dict()[name], rtol=0, atol=0)
    with pytest.raises(ValueError, match='seaformer_attention'):
        load_backbone(initialized, replace(initialized_cfg, seaformer_attention='lepe'))
    with pytest.raises(ValueError, match='head_width'):
        load_backbone(initialized, replace(initialized_cfg, head_width=16))
    payload['model'].pop('decode_head.classifier.weight')
    torch.save(payload, source)
    with pytest.raises(RuntimeError, match='Missing key'):
        load_backbone(initialized, initialized_cfg)


def test_v2_backbone_can_transfer_but_its_full_head_cannot(tmp_path):
    torch.set_num_threads(2)
    reference_config, models, checkpoints = reference_modules()
    v2_cfg = reference_config.Config(stem_pretrained=False)
    v2 = models.build_model(v2_cfg)
    complete = tmp_path / 'v2_full.pth'
    torch.save({'model': v2.state_dict(), 'config': asdict(v2_cfg)}, complete)
    cfg = Config(stem_pretrained=False, backbone_checkpoint=str(complete))
    target = build_model(cfg)
    load_backbone(target, cfg)
    for name, value in v2.backbone.state_dict().items():
        torch.testing.assert_close(value, target.backbone.state_dict()[name], rtol=0, atol=0)
    with pytest.raises(ValueError, match='Full checkpoint backbone_name'):
        load_backbone(target, replace(cfg, backbone_checkpoint='', model_checkpoint=str(complete)))

@pytest.mark.parametrize('flags,parameters,macs', [
    ([], 1850938, 512352256),
    (['--seaformer-attention', 'multiscale', '--no-learnable-squeeze'], 2040790, 561627136),
])
def test_profile_cli_selects_new_and_previous_backbone_settings(tmp_path, flags, parameters, macs):
    import json
    import subprocess
    import sys
    project = Path(__file__).resolve().parents[1]
    output = tmp_path / 'profile.json'
    result = subprocess.run([sys.executable, str(project / 'profile_model.py'), '--size', '256',
                             '--output', str(output), *flags], cwd=tmp_path,
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(output.read_text())
    assert report['parameters'] == parameters
    assert report['total_counted_macs'] == macs
