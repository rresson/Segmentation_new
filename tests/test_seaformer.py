"""无需网络/真实 ADE20K 的结构、冻结、统计与完整训练回归测试。"""
from dataclasses import replace
import csv
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import pytest
import torch
from torch.nn import functional as F
from torchvision.models import mobilenet_v2

from profile_model import profile_model
from segmentation import Config, run_experiment
from segmentation.config import prepare_config
from segmentation.freezing import configure_trainable, set_training_mode
from segmentation.models import build_model
from segmentation.models.mobile_v3_head import MobileV3C3Head, MobileV3FPNHead
from segmentation.models.seaformer_attention import SeaAttention
from segmentation.models.seaformer_stem import stem_block


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_stem_is_original_mobilenet_prefix():
    torch.manual_seed(7)
    original = mobilenet_v2(weights=None)
    original.features[7].conv[1][0].stride = (1, 1)
    original = original.features[:8].eval()
    torch.manual_seed(7)
    stem = stem_block(pretrained=False).eval()
    assert stem.backbone[7].conv[1][0].stride == (1, 1)
    assert original.state_dict().keys() == stem.backbone.state_dict().keys()
    for name, value in original.state_dict().items():
        torch.testing.assert_close(value, stem.backbone.state_dict()[name], rtol=0, atol=0)
    images = torch.randn(2, 3, 65, 97)
    with torch.no_grad():
        torch.testing.assert_close(original(images), stem(images), rtol=0, atol=0)
    assert stem(images).shape == (2, 64, 9, 13)


def test_head_is_exactly_the_original_mobile_c3_branch():
    reference = MobileV3FPNHead((32, 64, 128), width=16)
    head = MobileV3C3Head(128, width=16)
    head.context.load_state_dict(reference.context.state_dict())
    head.lateral.load_state_dict(reference.lateral[2].state_dict())
    head.refine.load_state_dict(reference.refine[2].state_dict())
    head.classifier.load_state_dict(reference.classifier.state_dict())
    c3 = torch.randn(2, 128, 7, 11)
    expected = reference.classifier(reference.refine[2](reference.lateral[2](reference.context(c3))))
    torch.testing.assert_close(head([c3]), expected)
    assert set(dict(head.named_children())) == {"context", "lateral", "refine", "classifier"}
    for invalid in ([c3, c3, c3], c3):
        with pytest.raises(ValueError, match="exactly one"):
            head(invalid)


@pytest.mark.parametrize("variant", ["original", "multiscale", "lepe"])
@pytest.mark.parametrize("squeeze", [False, True])
def test_attention_variants_backward_and_exact_matrix_macs(variant, squeeze):
    attention = SeaAttention(variant=variant, use_learnable_squeeze=squeeze)
    counted = []
    hook = attention.matmul.register_forward_hook(
        lambda module, args, output: counted.append(module.count_macs(args, output)))
    x = torch.randn(2, 64, 5, 7, requires_grad=True)
    y = attention(x)
    y.square().mean().backward()
    hook.remove()
    assert y.shape == x.shape and torch.isfinite(y).all()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert len(counted) == 4
    assert sum(counted) == 2 * 4 * (16 + 32) * (5**2 + 7**2)


@pytest.mark.parametrize("variant", ["original", "multiscale", "lepe"])
def test_network_odd_input_and_backprop(variant):
    cfg = Config(stem_pretrained=False, seaformer_attention=variant)
    model = build_model(cfg)
    configure_trainable(model, ("backbone", "decode_head"))
    x = torch.randn(2, 3, 65, 97)
    logits = model(x)
    assert logits.shape == (2, 150, 5, 7)
    logits.square().mean().backward()
    assert model.backbone.stem.backbone[0][0].weight.grad is not None
    assert model.decode_head.classifier.weight.grad is not None
    assert torch.isfinite(logits).all()


def test_frozen_backbone_and_profiler_preserve_weights_bn_and_modes():
    model = build_model(Config(stem_pretrained=False))
    names = configure_trainable(model, ("decode_head",))
    assert names and all(name.startswith("decode_head.") for name in names)
    set_training_mode(model, ("decode_head",))
    before = {name: value.clone() for name, value in model.backbone.state_dict().items()}
    head_before = model.decode_head.classifier.weight.detach().clone()
    modes = {name: m.training for name, m in model.named_modules()}
    x = torch.randn(2, 3, 64, 96)
    report = profile_model(model, x)
    assert {name: m.training for name, m in model.named_modules()} == modes
    assert not any(m._forward_hooks for m in model.modules())
    assert report["components"]["backbone"]["trainable_parameters"] == 0
    assert report["attention_macs"] > 0
    assert report["total_counted_macs"] == sum(report[key] for key in
                                             ("convolution_macs", "linear_macs", "attention_macs"))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001)
    logits = model(x)
    F.cross_entropy(logits, torch.randint(150, (2, *logits.shape[-2:]))).backward()
    optimizer.step()
    assert all(p.grad is None for p in model.backbone.parameters())
    for name, value in before.items():
        torch.testing.assert_close(value, model.backbone.state_dict()[name], rtol=0, atol=0)
    assert not torch.equal(head_before, model.decode_head.classifier.weight)


def make_dataset(root):
    rng = np.random.default_rng(4)
    for split, count in (("training", 2), ("validation", 1)):
        images = root / "images" / split
        masks = root / "annotations" / split
        images.mkdir(parents=True)
        masks.mkdir(parents=True)
        for i in range(count):
            Image.fromarray(rng.integers(0, 256, (32, 40, 3), dtype=np.uint8)).save(images / f"{i}.jpg")
            mask = rng.integers(1, 151, (32, 40), dtype=np.uint8)
            mask[:2] = 0
            Image.fromarray(mask).save(masks / f"{i}.png")


def test_pretrain_head_and_actual_checkpoint_resume(tmp_path, capsys):
    data = tmp_path / "data"
    make_dataset(data)
    cfg = Config(mode="pretrain", stem_pretrained=False, data_root=str(data),
                 output_dir=str(tmp_path / "pretrain"), epochs=2, save_every=1,
                 batch_size=2, crop_size=32, val_short_side=32, val_max_side=40,
                 scale_range=(1.0, 1.0), amp=False, device="cpu", head_width=16)
    report = run_experiment(cfg)
    log = capsys.readouterr().out
    assert log.index("Attention matrix MACs") < log.index("Epoch 001")
    assert report["attention_macs"] > 0
    pretrain = Path(cfg.output_dir)
    exported = torch.load(pretrain / "backbone_last.pth", weights_only=True)
    assert exported["out_channels"] == (128,)
    assert exported["architecture"] == "SeaFormerSingleFusion_v1"
    head_cfg = replace(cfg, mode="head", backbone_checkpoint=str(pretrain / "backbone_last.pth"),
                       output_dir=str(tmp_path / "head"))
    head_report = run_experiment(head_cfg)
    output = Path(head_cfg.output_dir)
    last = torch.load(output / "last.pth", weights_only=True)
    assert head_report["components"]["backbone"]["trainable_parameters"] == 0
    assert all(name.startswith("decode_head.") for name in last["trainable_names"])
    for name, value in exported["state_dict"].items():
        torch.testing.assert_close(last["model"]["backbone." + name], value, rtol=0, atol=0)
    assert json.loads((output / "model_profile.json").read_text())["attention_macs"] > 0
    rows = list(csv.DictReader((output / "history.csv").open()))
    assert len(rows) == 2 and all(np.isfinite(float(row["loss"])) for row in rows)

    resumed = tmp_path / "resume"
    resumed.mkdir()
    shutil.copy2(output / "epoch_001.pth", resumed / "epoch_001.pth")
    resume_cfg = replace(head_cfg, output_dir=str(resumed), resume=str(resumed / "epoch_001.pth"))
    run_experiment(resume_cfg)
    restored = torch.load(resumed / "last.pth", weights_only=True)
    assert restored["epoch"] == 2 and restored["optimizer"]["state"]
    for name, value in last["model"].items():
        torch.testing.assert_close(restored["model"][name], value, rtol=0, atol=0)

    with pytest.raises(ValueError, match="seaformer_depths"):
        run_experiment(replace(resume_cfg, seaformer_depths=(2, 2, 2)))


def test_seaformer_never_selects_another_head():
    assert prepare_config(Config(mode="pretrain")).head_name == "mobile_c3"
    with pytest.raises(ValueError, match="mobile_c3"):
        prepare_config(Config(head_name="mobile_fpn"))
    with pytest.raises(ValueError, match="depths"):
        prepare_config(Config(seaformer_depths=(0, 2, 2)))


@pytest.mark.parametrize("backbone,head", [("demo", "simple"), ("small_cnn", "fpn"), ("mobile_v3", "mobile_fpn")])
def test_existing_models_still_work(backbone, head):
    model = build_model(Config(backbone_name=backbone, head_name=head)).eval()
    report = profile_model(model, torch.randn(1, 3, 64, 80))
    assert report["output_shape"][1] == 150
    assert report["parameters"] > 0 and report["attention_macs"] == 0
