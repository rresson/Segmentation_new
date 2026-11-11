from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from segmentation import Config
from segmentation.pipeline import backbone_training_checkpoint, checkpoint_run, run_pipeline
from segmentation.run_paths import reserve_head_directory
from helpers import make_dataset as make_data


@pytest.fixture(autouse=True)
def threads():
    before = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(before)


def small_config(tmp_path):
    data = tmp_path / 'data'
    make_data(data)
    return Config(data_root=str(data), output_root=str(tmp_path / 'runs'),
                  stem_pretrained=False, epochs=1, save_every=1, batch_size=2,
                  crop_size=64, val_short_side=64, val_max_side=80,
                  scale_range=(1.0, 1.0), device='cpu', amp=False)


def assert_frozen_backbone(source, output):
    weights = torch.load(source, weights_only=True)['state_dict']
    model = torch.load(output / 'last.pth', weights_only=True)['model']
    for name, value in weights.items():
        torch.testing.assert_close(value, model['backbone.' + name], rtol=0, atol=0)


def test_all_three_choices_keep_lineage_and_exact_frozen_states(tmp_path):
    cfg = small_config(tmp_path)
    root = Path(cfg.output_root)
    joint = run_pipeline(cfg, 1)
    assert Path(joint['pretrain']['output_dir']) == root / 'run1/pretrain'
    assert Path(joint['head']['output_dir']) == root / 'run1/head1'
    source = root / 'run1/pretrain/backbone_best_miou.pth'
    first_head = root / 'run1/head1'
    assert_frozen_backbone(source, first_head)
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in first_head.iterdir() if p.is_file()}
    extra = run_pipeline(cfg, 2, checkpoint=str(source))
    assert Path(extra['head']['output_dir']) == root / 'run1/head2'
    assert_frozen_backbone(source, root / 'run1/head2')
    assert before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in first_head.iterdir() if p.is_file()}
    assert not (root / 'run2').exists()

    initial = torch.load(source.with_name('best_miou.pth'), weights_only=True)['model']
    custom = run_pipeline(cfg, 3, checkpoint=str(source))
    output = root / 'run2/custom'
    assert Path(custom['backbone']['output_dir']) == output
    trained = torch.load(output / 'last.pth', weights_only=True)
    assert all(name.startswith('backbone.') for name in trained['trainable_names'])
    assert any(not torch.equal(value, trained['model'][name]) for name, value in initial.items()
               if name.startswith('backbone.') and name.endswith('weight'))
    for name, value in initial.items():
        if name.startswith('decode_head.'):
            torch.testing.assert_close(value, trained['model'][name], rtol=0, atol=0)
    assert custom['backbone']['components']['decode_head']['trainable_parameters'] == 0
    saved = json.loads((output / 'config.json').read_text())
    assert saved['trainable_prefixes'] == ['backbone']
    assert saved['model_checkpoint'] == str(source.with_name('best_miou.pth'))
    for folder in (root / 'run1/pretrain', first_head, root / 'run1/head2', output):
        assert (folder / 'config.json').is_file()
        assert (folder / 'model_profile.json').is_file()
        assert (folder / 'pipeline.json').is_file()

    updated_source = output / 'backbone_best_miou.pth'
    updated_head = run_pipeline(cfg, 2, checkpoint=str(updated_source))
    assert Path(updated_head['head']['output_dir']) == root / 'run2/head1'
    assert_frozen_backbone(updated_source, root / 'run2/head1')


@pytest.mark.parametrize('entrypoint', ['main.py', 'pipeline.py'])
def test_interactive_pipeline_and_restarted_process_keep_same_run(tmp_path, entrypoint):
    cfg = small_config(tmp_path)
    project = Path(__file__).resolve().parents[1]
    root = tmp_path / 'runs with spaces'
    common = [sys.executable, str(project / entrypoint), '--data-root', cfg.data_root,
              '--output-root', str(root), '--epochs', '1', '--batch-size', '2',
              '--crop-size', '64', '--val-short-side', '64', '--val-max-side', '80',
              '--device', 'cpu', '--no-pretrained-stem', '--head-width', '64', '--context-expanded-channels', '112']
    env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}
    first = subprocess.run(common, input='1\n', text=True, capture_output=True, cwd=tmp_path,
                           env=env, timeout=120)
    assert first.returncode == 0, first.stdout + first.stderr
    assert '请选择' in first.stdout
    assert first.stdout.count('pre-training detection:') == 2
    source = root / 'run1/pretrain/backbone_best_miou.pth'
    second = subprocess.run(common, input=f'2\n"{source}"\n', text=True, capture_output=True,
                            cwd=tmp_path, env=env, timeout=120)
    assert second.returncode == 0, second.stdout + second.stderr
    assert (root / 'run1/head1/last.pth').is_file()
    assert (root / 'run1/head2/last.pth').is_file()
    assert not (root / 'run2').exists()
    assert_frozen_backbone(source, root / 'run1/head2')
    assert second.stdout.count('pre-training detection:') == 1

    # 非交互命令也从同一入口进入 pipeline；custom 仅更新 backbone。
    third = subprocess.run(common + ['--choice', '3', '--checkpoint', str(source)],
                           text=True, capture_output=True, cwd=tmp_path, env=env, timeout=120)
    assert third.returncode == 0, third.stdout + third.stderr
    assert '请选择' not in third.stdout
    custom = root / 'run2/custom'
    initial = torch.load(source.with_name('best_miou.pth'), weights_only=True)['model']
    trained = torch.load(custom / 'last.pth', weights_only=True)
    assert all(name.startswith('backbone.') for name in trained['trainable_names'])
    for name, value in initial.items():
        if name.startswith('decode_head.'):
            torch.testing.assert_close(value, trained['model'][name], rtol=0, atol=0)
    assert json.loads((custom / 'pipeline.json').read_text())['choice'] == 3

    history = (custom / 'history.csv').read_bytes()
    resume_command = [sys.executable, str(project / 'main.py'), *common[2:]]
    resumed = subprocess.run(resume_command + ['--mode', 'custom', '--trainable-prefixes', 'backbone',
                                       '--resume', str(custom / 'last.pth')],
                             text=True, capture_output=True, cwd=tmp_path, env=env, timeout=120)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert '请选择' not in resumed.stdout
    assert (custom / 'history.csv').read_bytes() == history
    assert not (root / 'run3').exists()



def test_main_explicit_mode_and_resume_skip_pipeline(tmp_path):
    cfg = small_config(tmp_path)
    project = Path(__file__).resolve().parents[1]
    root = tmp_path / 'single stage runs'
    common = [sys.executable, str(project / 'main.py'), '--data-root', cfg.data_root,
              '--output-root', str(root), '--epochs', '2', '--batch-size', '2',
              '--crop-size', '64', '--val-short-side', '64', '--val-max-side', '80',
              '--device', 'cpu', '--no-pretrained-stem',
              '--seaformer-attention', 'multiscale', '--no-learnable-squeeze',
              '--head-width', '64', '--context-expanded-channels', '120']
    env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}
    first = subprocess.run(common + ['--mode=pretrain'], stdin=subprocess.DEVNULL,
                           text=True, capture_output=True, cwd=tmp_path, env=env, timeout=120)
    assert first.returncode == 0, first.stdout + first.stderr
    output = root / 'run1/pretrain'
    assert '请选择' not in first.stdout
    assert not (output / 'pipeline.json').exists()
    assert not (root / 'run1/head1').exists()
    # 已完成 checkpoint 的续训应直接恢复原目录，不再弹出菜单或创建 run2。
    before = (output / 'history.csv').read_bytes()
    resumed = subprocess.run(common + ['--mode', 'pretrain', '--resume', str(output / 'last.pth')],
                             stdin=subprocess.DEVNULL, text=True, capture_output=True,
                             cwd=tmp_path, env=env, timeout=120)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert '请选择' not in resumed.stdout
    assert '恢复第 2 轮' in resumed.stdout
    assert before == (output / 'history.csv').read_bytes()
    assert not (root / 'run2').exists()


@pytest.mark.parametrize('arguments', [
    ['--mode', 'pretrain', '--choice', '1'],
    ['--resume', 'missing.pth', '--head-epochs', '2'],
    ['--backbone-checkpoint', 'missing.pth'],
    ['--choice', '2'],
])
def test_main_rejects_ambiguous_or_incomplete_cli_before_training(tmp_path, arguments):
    project = Path(__file__).resolve().parents[1]
    output = tmp_path / 'runs'
    result = subprocess.run([sys.executable, str(project / 'main.py'),
                             '--output-root', str(output), *arguments],
                            stdin=subprocess.DEVNULL, text=True, capture_output=True,
                            cwd=tmp_path, timeout=30)
    assert result.returncode == 2, result.stdout + result.stderr
    assert not output.exists()


def test_external_checkpoint_binding_is_persistent_and_concurrent(tmp_path):
    source = tmp_path / 'external.pth'
    source.write_bytes(b'fixed checkpoint content')
    root = tmp_path / 'runs'
    with ThreadPoolExecutor(max_workers=4) as pool:
        owners = list(pool.map(lambda _: checkpoint_run(source, root), range(8)))
    assert set(owners) == {root / 'run1'}
    assert checkpoint_run(source, root) == root / 'run1'
    source.write_bytes(b'different backbone at the same path')
    assert checkpoint_run(source, root) == root / 'run2'
    assert not (root / 'run3').exists()


def test_external_checkpoint_is_bound_across_processes(tmp_path):
    source = tmp_path / 'external.pth'
    source.write_bytes(b'fixed checkpoint content')
    root = tmp_path / 'runs'
    assert checkpoint_run(source, root) == root / 'run1'
    project = Path(__file__).resolve().parents[1]
    script = ('from pathlib import Path; import sys; '
              'from segmentation.pipeline import checkpoint_run; '
              'print(checkpoint_run(Path(sys.argv[1]), Path(sys.argv[2])))')
    result = subprocess.run([sys.executable, '-c', script, str(source), str(root)],
                            cwd=project, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(root / 'run1')


def test_head_number_allocation_preserves_old_and_concurrent_outputs(tmp_path):
    (tmp_path / 'head').mkdir()
    (tmp_path / 'head_2').mkdir()
    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(lambda _: reserve_head_directory(tmp_path), range(8)))
    assert {path.name for path in paths} == {f'head{i}' for i in range(3, 11)}
    assert all(path.is_dir() for path in paths)


def test_backbone_only_requires_matching_trained_head(tmp_path):
    head_only = tmp_path / 'head.pth'
    torch.save({'model': {'decode_head.segment_head.weight': torch.ones(2)}}, head_only)
    with pytest.raises(ValueError, match='只有 head'):
        backbone_training_checkpoint(head_only)
    source = tmp_path / 'backbone_best_miou.pth'
    torch.save({'state_dict': {'stem.weight': torch.ones(2)}}, source)
    with pytest.raises(ValueError, match='已有 head'):
        backbone_training_checkpoint(source)
    paired = tmp_path / 'best_miou.pth'
    torch.save({'model': {'backbone.stem.weight': torch.zeros(2),
                          'decode_head.segment_head.weight': torch.ones(2)}}, paired)
    with pytest.raises(ValueError, match='不匹配'):
        backbone_training_checkpoint(source)
    torch.save({'model': {'backbone.stem.weight': torch.ones(2),
                          'decode_head.segment_head.weight': torch.ones(2)}}, paired)
    assert backbone_training_checkpoint(source) == paired
    assert backbone_training_checkpoint(paired) == paired


def test_single_stage_requires_user_checkpoint_and_joint_rejects_resume(tmp_path):
    cfg = Config(output_root=str(tmp_path / 'runs'))
    for choice in (2, 3):
        with pytest.raises(FileNotFoundError, match='手动'):
            run_pipeline(cfg, choice)
    assert not (tmp_path / 'runs').exists()
    with pytest.raises(ValueError, match='续训'):
        run_pipeline(replace(cfg, resume='last.pth'), 1)
    with pytest.raises(ValueError, match='手动权重'):
        run_pipeline(cfg, 1, checkpoint='some.pth')
