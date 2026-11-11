from dataclasses import replace
from pathlib import Path
import json
import shutil
from unittest.mock import patch

import numpy as np
from PIL import Image
import pytest
import torch
from torch.nn import functional as F

from segmentation.models.network import main_network
from profile_model import profile_model
from segmentation import Config, run_experiment
from segmentation.config import network_kwargs, prepare_config
from segmentation.models import build_model
from segmentation.checkpoints import load_full_model, load_backbone
from segmentation.freezing import configure_trainable, set_training_mode


@pytest.fixture(autouse=True)
def threads():
    before = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(before)


@pytest.mark.parametrize("attention", ["original", "multiscale", "lepe"])
@pytest.mark.parametrize("fusion", ["original", "enhanced"])
@pytest.mark.parametrize("head", ["original", "multilevel"])
def test_exact_uploaded_network_equivalence(attention, fusion, head):
    cfg = Config(stem_pretrained=False, attention_type=attention, fusion_type=fusion, head_name=head)
    source = main_network(**network_kwargs(cfg), stem_pretrained=False)
    adapted = build_model(cfg)
    # 模拟用户上传 load_checkpoint.py 所使用的 checkpoint 格式。
    checkpoint = dict(model_state_dict=source.state_dict(), model_config=network_kwargs(cfg), num_classes=150)
    load_full_model(adapted, checkpoint, cfg)
    source.eval()
    adapted.eval()
    x = torch.randn(2, 3, 65, 81)
    with torch.no_grad():
        torch.testing.assert_close(adapted(x), source(x), rtol=0, atol=0)
    source.train()
    adapted.train()
    torch.manual_seed(15)
    old = source(x)
    torch.manual_seed(15)
    new = adapted(x)
    assert isinstance(new, tuple) and len(new) == 2
    for actual, expected in zip(new, old):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    sum(y.mean() for y in new).backward()
    assert adapted.backbone.stem.backbone[0][0].weight.grad is not None


@pytest.mark.parametrize("squeeze", [False, True])
def test_profile_counts_actual_attention_matmuls_and_aux_without_mutation(squeeze):
    cfg = Config(stem_pretrained=False, use_learnable_squeeze=squeeze)
    model = build_model(cfg)
    configure_trainable(model, ("decode_head",))
    set_training_mode(model, ("decode_head",))
    before = {k: v.clone() for k, v in model.state_dict().items()}
    modes = {k: m.training for k, m in model.named_modules()}
    rng_before = torch.get_rng_state().clone()
    actual_macs = []
    multiply = torch.matmul

    def record(left, right, *args, **kwargs):
        result = multiply(left, right, *args, **kwargs)
        actual_macs.append(result.numel() * left.shape[-1])
        return result

    with patch("torch.matmul", side_effect=record):
        report = profile_model(model, torch.zeros(2, 3, 65, 81))
    assert report["attention_macs"] == sum(actual_macs) > 0
    assert len(actual_macs) == 6 * 4  # 原网络六个 context block
    assert report["includes_auxiliary_head"] and report["auxiliary_macs"] > 0
    assert report["auxiliary_output_shape"] == report["output_shape"] == [2, 150, 65, 81]
    assert report["components"]["backbone"]["trainable_parameters"] == 0
    assert report["total_counted_macs"] == sum(report[key] for key in
                                             ("convolution_macs", "linear_macs", "attention_macs"))
    assert modes == {k: m.training for k, m in model.named_modules()}
    assert torch.equal(rng_before, torch.get_rng_state())
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
    assert not any(m._forward_hooks for m in model.modules())


def test_full_weights_without_aux_and_backbone_from_uploaded_checkpoint(tmp_path):
    cfg = Config(stem_pretrained=False, use_aux=False, head_name="multilevel")
    source = main_network(**network_kwargs(cfg), stem_pretrained=False)
    checkpoint = dict(model_state_dict=source.state_dict(), model_config=network_kwargs(cfg))
    path = tmp_path / "uploaded.pth"
    torch.save(checkpoint, path)
    actual = build_model(cfg)
    load_backbone(actual, replace(cfg, backbone_checkpoint=str(path)))
    for key, value in actual.backbone.state_dict().items():
        torch.testing.assert_close(value, source.state_dict()[key], rtol=0, atol=0)
    load_full_model(actual, checkpoint, cfg)
    report = profile_model(actual, torch.zeros(1, 3, 64, 64))
    assert report["auxiliary_macs"] == 0 and not report["includes_auxiliary_head"]
    with pytest.raises(RuntimeError, match="Missing key"):
        load_full_model(build_model(replace(cfg, use_aux=True)), checkpoint, replace(cfg, use_aux=True))
    with pytest.raises(ValueError, match="segment_type"):
        load_full_model(actual, checkpoint, replace(cfg, head_name="original"))


def make_data(root):
    rng = np.random.default_rng(22)
    for split, count in (("training", 2), ("validation", 1)):
        (root / "images" / split).mkdir(parents=True)
        (root / "annotations" / split).mkdir(parents=True)
        for i in range(count):
            Image.fromarray(rng.integers(0, 256, (64, 80, 3), dtype=np.uint8)).save(root / "images" / split / f"{i}.jpg")
            mask = rng.integers(1, 151, (64, 80), dtype=np.uint8)
            mask[:2] = 0
            Image.fromarray(mask).save(root / "annotations" / split / f"{i}.png")


@pytest.mark.parametrize("aux", [False, True])
def test_pretrain_frozen_head_and_real_resume(tmp_path, aux, capsys):
    data = tmp_path / "data"
    make_data(data)
    cfg = Config(stem_pretrained=False, use_aux=aux, data_root=str(data),
                 output_dir=str(tmp_path / "pretrain"), mode="pretrain", epochs=2,
                 save_every=1, batch_size=2, crop_size=64, val_short_side=64, val_max_side=80,
                 scale_range=(1.0, 1.0), device="cpu", amp=False)
    report = run_experiment(cfg)
    log = capsys.readouterr().out
    assert log.index("Attention matrix MACs") < log.index("Epoch 001")
    assert report["includes_auxiliary_head"] is aux
    weights = Path(cfg.output_dir) / "backbone_last.pth"
    exported = torch.load(weights, weights_only=True)
    head_cfg = replace(cfg, mode="head", backbone_checkpoint=str(weights), output_dir=str(tmp_path / "head"))
    run_experiment(head_cfg)
    head_path = Path(head_cfg.output_dir)
    last = torch.load(head_path / "last.pth", weights_only=True)
    first = torch.load(head_path / "epoch_001.pth", weights_only=True)
    for key, value in exported["state_dict"].items():
        torch.testing.assert_close(value, last["model"]["backbone." + key], rtol=0, atol=0)
    for prefix in ("decode_head.segment_head.",) + (("decode_head.auxiliary_head.",) if aux else ()):
        assert any(not torch.equal(value, last["model"][key]) for key, value in first["model"].items()
                   if key.startswith(prefix) and key.endswith("weight"))
    resumed = tmp_path / "resume"
    resumed.mkdir()
    shutil.copy2(head_path / "epoch_001.pth", resumed / "epoch_001.pth")
    resume_cfg = replace(head_cfg, resume=str(resumed / "epoch_001.pth"), output_dir=str(resumed))
    run_experiment(resume_cfg)
    restored = torch.load(resumed / "last.pth", weights_only=True)
    assert restored["epoch"] == 2
    assert all(torch.equal(v, restored["model"][k]) for k, v in last["model"].items())
    with pytest.raises(ValueError, match="aux_loss_weight"):
        run_experiment(replace(resume_cfg, aux_loss_weight=0.7))


def test_auxiliary_loss_has_configured_weight():
    from segmentation.engine import train_one_epoch
    from torch import nn
    class FixedOutputs(nn.Module):
        def __init__(self):
            super().__init__()
            self.decode_head = nn.ParameterDict({
                "main": nn.Parameter(torch.randn(1, 150, 2, 2)),
                "aux": nn.Parameter(torch.randn(1, 150, 2, 2)),
            })
        def forward(self, images):
            return self.decode_head["main"], self.decode_head["aux"]
    model = FixedOutputs()
    labels = torch.tensor([[[0, 1], [255, 3]]])
    expected = F.cross_entropy(model.decode_head["main"], labels, ignore_index=255)
    expected += 0.4 * F.cross_entropy(model.decode_head["aux"], labels, ignore_index=255)
    cfg = Config(use_aux=True, aux_loss_weight=0.4, trainable_prefixes=("decode_head",), amp=False)
    loss = train_one_epoch(model, [(torch.zeros(1, 3, 2, 2), labels)],
                          torch.optim.SGD(model.parameters(), lr=0),
                          torch.amp.GradScaler("cuda", enabled=False), cfg, torch.device("cpu"))
    assert loss == pytest.approx(expected.item())


def test_model_checkpoint_replaces_head_mode_backbone_requirement():
    cfg = prepare_config(Config(mode="head", model_checkpoint="uploaded.pth"))
    assert cfg.backbone_checkpoint == "" and cfg.trainable_prefixes == ("decode_head",)
    with pytest.raises(ValueError, match="dim must be 64"):
        prepare_config(Config(dim=128))
