from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from segmentation import Config, run_experiment
from segmentation.config import prepare_config
from segmentation.run_paths import prepare_output_dir


def allocate(cfg):
    prepared = prepare_config(cfg)
    output = prepare_output_dir(prepared, explicit_output_dir=bool(cfg.output_dir))
    return prepared, output


def saved_pretrain(root, name):
    folder = root / name / 'pretrain'
    folder.mkdir(parents=True)
    checkpoint = folder / 'backbone_best_miou.pth'
    checkpoint.write_bytes(name.encode())
    return checkpoint


def test_custom_binds_checkpoint_and_repeated_output_to_same_run(tmp_path):
    saved_pretrain(tmp_path, 'run2')
    newest = saved_pretrain(tmp_path, 'run10')
    cfg = Config(mode='custom', output_root=str(tmp_path))
    first, output = allocate(cfg)
    assert output == tmp_path / 'run10/custom'
    assert first.run_name == 'run10'
    assert Path(first.backbone_checkpoint) == newest
    assert not first.allow_random_backbone
    (output / 'keep.txt').write_text('preserve')
    second, repeated = allocate(cfg)
    assert repeated == tmp_path / 'run10/custom_2'
    assert second.backbone_checkpoint == first.backbone_checkpoint
    assert (output / 'keep.txt').read_text() == 'preserve'
    assert not (tmp_path / 'run11').exists()
    assert cfg.run_name == cfg.backbone_checkpoint == ''


def test_explicit_run_and_explicit_output_bind_source(tmp_path):
    older = saved_pretrain(tmp_path, 'run1')
    saved_pretrain(tmp_path, 'run2')
    cfg = Config(mode='custom', output_root=str(tmp_path), run_name='run1')
    selected, output = allocate(cfg)
    assert Path(selected.backbone_checkpoint) == older
    assert output == tmp_path / 'run1/custom'
    selected, output = allocate(replace(cfg, run_name='', output_dir=str(tmp_path / 'run1/custom')))
    assert Path(selected.backbone_checkpoint) == older
    assert output == tmp_path / 'run1/custom_2'
    with pytest.raises(ValueError, match='different runs'):
        allocate(replace(cfg, output_dir=str(tmp_path / 'run2/custom')))


def test_latest_unfinished_pretrain_does_not_silently_load_old_run(tmp_path):
    saved_pretrain(tmp_path, 'run1')
    (tmp_path / 'run2/pretrain').mkdir(parents=True)
    cfg = Config(mode='custom', output_root=str(tmp_path))
    with pytest.raises(FileNotFoundError, match='run2'):
        allocate(cfg)
    assert not (tmp_path / 'run2/custom').exists()
    with pytest.raises(FileNotFoundError, match='run3'):
        allocate(replace(cfg, run_name='run3'))
    assert not (tmp_path / 'run3').exists()


def test_manual_checkpoints_override_automatic_lookup(tmp_path):
    checkpoint = tmp_path / 'external.pth'
    checkpoint.write_bytes(b'external')
    cfg = Config(mode='custom', output_root=str(tmp_path / 'runs'), model_checkpoint=str(checkpoint))
    prepared, output = allocate(cfg)
    assert prepared.backbone_checkpoint == ''
    assert Path(prepared.model_checkpoint) == checkpoint
    assert output == tmp_path / 'runs/run1/custom'
    prepared, output = allocate(replace(cfg, model_checkpoint='', backbone_checkpoint=str(checkpoint)))
    assert Path(prepared.backbone_checkpoint) == checkpoint
    assert output == tmp_path / 'runs/run2/custom'
    with pytest.raises(FileNotFoundError):
        allocate(replace(cfg, model_checkpoint=str(tmp_path / 'missing.pth')))
    assert not (tmp_path / 'runs/run3').exists()


def test_resume_uses_its_own_run_even_if_a_newer_pretrain_exists(tmp_path):
    saved_pretrain(tmp_path, 'run1')
    old = tmp_path / 'run1/custom'
    old.mkdir()
    checkpoint = old / 'last.pth'
    checkpoint.write_bytes(b'resume')
    saved_pretrain(tmp_path, 'run2')
    cfg = Config(mode='custom', output_root=str(tmp_path), resume=str(checkpoint))
    prepared, output = allocate(cfg)
    assert output == old
    assert prepared.run_name == 'run1'
    assert prepared.backbone_checkpoint == ''
    assert not (tmp_path / 'run1/custom_2').exists()
    with pytest.raises(ValueError, match='run_name'):
        allocate(replace(cfg, run_name='run2'))


def test_pretrain_numbering_profile_purity_and_concurrent_custom(tmp_path):
    root = tmp_path / 'runs'
    cfg = Config(output_root=str(root))
    prepared = prepare_config(cfg)
    assert prepared.backbone_checkpoint == ''
    assert not root.exists()
    assert allocate(cfg)[1] == root / 'run1/pretrain'
    assert allocate(cfg)[1] == root / 'run2/pretrain'
    (root / 'run2/pretrain/backbone_best_miou.pth').write_bytes(b'weights')
    custom = replace(cfg, mode='custom')
    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(pool.map(lambda _: allocate(custom)[1], range(8)))
    assert len(set(outputs)) == 8
    assert all(path.parent == root / 'run2' for path in outputs)
    assert not (root / 'run3').exists()
    with pytest.raises(FileExistsError, match='Explicit run'):
        allocate(replace(cfg, run_name='run2'))


def test_no_pretrain_rejected_without_creating_custom_output(tmp_path):
    root = tmp_path / 'absent'
    with pytest.raises(FileNotFoundError, match='No pretrain run'):
        allocate(Config(mode='custom', output_root=str(root)))
    assert not root.exists()


def test_real_pretrain_then_custom_loads_backbone_and_resumes(tmp_path):
    torch.set_num_threads(2)
    data = tmp_path / 'data'
    rng = np.random.default_rng(18)
    for split, count in (('training', 2), ('validation', 1)):
        images = data / 'images' / split
        masks = data / 'annotations' / split
        images.mkdir(parents=True)
        masks.mkdir(parents=True)
        for index in range(count):
            Image.fromarray(rng.integers(0, 256, (64, 80, 3), dtype=np.uint8)).save(images / f'{index}.jpg')
            Image.fromarray(rng.integers(1, 151, (64, 80), dtype=np.uint8)).save(masks / f'{index}.png')
    root = tmp_path / 'runs'
    cfg = Config(output_root=str(root), data_root=str(data), stem_pretrained=False,
                 epochs=2, save_every=1, batch_size=2, crop_size=64,
                 val_short_side=64, val_max_side=80, scale_range=(1.0, 1.0),
                 device='cpu', amp=False)
    run_experiment(cfg)
    source = root / 'run1/pretrain/backbone_best_miou.pth'
    before_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    custom = replace(cfg, mode='custom', trainable_prefixes=('decode_head',))
    run_experiment(custom)
    output = root / 'run1/custom'
    saved_cfg = json.loads((output / 'config.json').read_text())
    assert saved_cfg['run_name'] == 'run1'
    assert saved_cfg['backbone_checkpoint'] == str(source)
    assert saved_cfg['output_dir'] == str(output)
    pretrained = torch.load(source, weights_only=True)['state_dict']
    trained = torch.load(output / 'last.pth', weights_only=True)
    for name, value in pretrained.items():
        torch.testing.assert_close(value, trained['model']['backbone.' + name], rtol=0, atol=0)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before_hash
    assert not (root / 'run2').exists()
    # 恢复同一 custom 实验，且兼容尚无 run_name 字段的旧格式配置。
    checkpoint = torch.load(output / 'epoch_001.pth', weights_only=True)
    checkpoint['config'].pop('run_name')
    torch.save(checkpoint, output / 'legacy_resume.pth')
    run_experiment(replace(custom, resume=str(output / 'legacy_resume.pth')))
    restored = torch.load(output / 'last.pth', weights_only=True)
    assert restored['epoch'] == 2
    for name, value in trained['model'].items():
        torch.testing.assert_close(value, restored['model'][name], rtol=0, atol=0)
    assert not (root / 'run1/custom_2').exists()
