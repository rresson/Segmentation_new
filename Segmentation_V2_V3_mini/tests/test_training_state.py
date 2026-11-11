from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import torch
from PIL import Image

from segmentation import Config, run_experiment
from segmentation.checkpoints import load_backbone
from segmentation.models import build_model
from segmentation.pipeline import checkpoint_run, run_pipeline
from helpers import make_dataset


def test_real_resume_and_inference_keep_expansion_and_frozen_backbone(tmp_path):
    data = tmp_path/'data'
    make_dataset(data)
    cfg = Config(stem_pretrained=False, data_root=str(data), output_root=str(tmp_path/'runs'),
                 head_width=64, context_expanded_channels=112, epochs=2, save_every=1, batch_size=2,
                 crop_size=32, val_short_side=32, val_max_side=40,
                 scale_range=(1.,1.), device='cpu', amp=False)
    pretrain = run_experiment(cfg)
    folder = Path(pretrain['output_dir'])
    assert pretrain['parameters'] == 806650
    source = folder/'backbone_best_miou.pth'
    head_cfg = replace(cfg, mode='head', backbone_checkpoint=str(source))
    report = run_experiment(head_cfg)
    output = Path(report['output_dir'])
    saved = torch.load(output/'last.pth',weights_only=True)
    assert saved['config']['head_name'] == 'mobile_c3_mini'
    assert saved['config']['context_expanded_channels'] == 112
    assert saved['config']['head_width'] == 64
    assert report['trainable_parameters'] == 48790
    for key,value in torch.load(source,weights_only=True)['state_dict'].items():
        torch.testing.assert_close(value,saved['model']['backbone.'+key],rtol=0,atol=0)
    # Replay epoch 2 from epoch 1, with restored optimizer/scheduler and the same RNG policy.
    resumed = tmp_path/'resume'
    resumed.mkdir()
    shutil.copy2(output/'epoch_001.pth',resumed/'epoch_001.pth')
    resume_cfg = replace(head_cfg,resume=str(resumed/'epoch_001.pth'),output_dir=str(resumed))
    run_experiment(resume_cfg)
    restored = torch.load(resumed/'last.pth',weights_only=True)
    assert restored['epoch'] == 2 and restored['optimizer']['state']
    for key,value in saved['model'].items():
        torch.testing.assert_close(value,restored['model'][key],rtol=0,atol=0)
    with pytest.raises(ValueError,match='context_expanded_channels'):
        run_experiment(replace(resume_cfg,context_expanded_channels=120))
    # Inference reconstructs W64/E112 instead of using the new W128/E368 defaults.
    project = Path(__file__).resolve().parents[1]
    prediction = tmp_path/'prediction.png'
    env = {**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','PYTHONDONTWRITEBYTECODE':'1'}
    result = subprocess.run([sys.executable,str(project/'load_checkpoint.py'),str(output/'best_miou.pth'),
                             '--device','cpu','--image',str(data/'images/validation/0.jpg'),'--output',str(prediction)],
                            cwd=tmp_path,env=env,text=True,capture_output=True,timeout=60)
    assert result.returncode == 0,result.stdout+result.stderr
    with Image.open(prediction) as image:
        assert image.size == (40,32)
        assert image.getextrema()[1] < 150


def test_full_model_initialization_checks_expansion_metadata_and_skips_download(tmp_path,monkeypatch):
    cfg = Config(stem_pretrained=False)
    original=build_model(cfg)
    source=tmp_path/'full.pth'
    payload={'config':asdict(cfg),'model':original.state_dict()}
    torch.save(payload,source)
    def forbidden_download(*args,**kwargs):
        raise AssertionError('Full initialization must not download stem weights')
    monkeypatch.setattr(torch.hub,'load_state_dict_from_url',forbidden_download)
    initialized_cfg=replace(cfg,stem_pretrained=True,model_checkpoint=str(source))
    initialized=build_model(initialized_cfg)
    load_backbone(initialized,initialized_cfg)
    for key,value in original.state_dict().items():
        torch.testing.assert_close(value,initialized.state_dict()[key],rtol=0,atol=0)
    with pytest.raises(ValueError,match='head_width'):
        load_backbone(initialized,replace(initialized_cfg,head_width=64))
    with pytest.raises(ValueError,match='context_expanded_channels'):
        load_backbone(initialized,replace(initialized_cfg,context_expanded_channels=352))
    payload['config'].pop('context_expanded_channels')
    torch.save(payload,source)
    with pytest.raises(ValueError,match='missing context_expanded_channels'):
        load_backbone(initialized,initialized_cfg)


def test_other_projects_run_directories_are_read_only_sources(tmp_path):
    source=tmp_path/'original/runs/run9/pretrain/backbone_best_miou.pth'
    source.parent.mkdir(parents=True)
    source.write_bytes(b'weights')
    root=tmp_path/'mini_runs'
    assert checkpoint_run(source,root) == root/'run1'
    assert checkpoint_run(source,root) == root/'run1'
    assert list(source.parent.parent.iterdir()) == [source.parent]
    assert json.loads((root/'run1/backbone_source.json').read_text())['checkpoint'] == str(source)


def test_backbone_only_rejects_old_head_before_allocating_run(tmp_path):
    cfg=Config(stem_pretrained=False,output_root=str(tmp_path/'runs'))
    model=build_model(cfg)
    old_config=asdict(cfg)
    old_config['head_name']='mobile_c3'
    source=tmp_path/'old_full.pth'
    torch.save({'config':old_config,'model':model.state_dict()},source)
    with pytest.raises(ValueError,match='head_name'):
        run_pipeline(cfg,3,checkpoint=str(source))
    assert not Path(cfg.output_root).exists()

@pytest.mark.parametrize('flags,parameters,macs', [
    ([], 997178, 427811840),
    (['--head-width', '64', '--context-expanded-channels', '120',
      '--seaformer-attention', 'multiscale', '--no-learnable-squeeze'], 998278, 445306368),
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
