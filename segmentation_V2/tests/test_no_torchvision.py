"""移除 torchvision 后的独立进程和预训练权重加载回归。"""
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from segmentation.models.Stem_block import IMAGENET_WEIGHTS_URL, stem_block


def test_pretrained_loader_keeps_official_keys_and_requires_complete_weights(monkeypatch):
    original = stem_block(pretrained=False)
    state = {"features." + name: value.clone()
             for name, value in original.backbone.state_dict().items()}
    state["features.8.conv.0.0.weight"] = torch.zeros(1)
    state["classifier.1.weight"] = torch.zeros(1)
    calls = []

    def load(url, **kwargs):
        calls.append((url, kwargs))
        return state

    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", load)
    loaded = stem_block(pretrained=True)
    for name, value in original.state_dict().items():
        torch.testing.assert_close(value, loaded.state_dict()[name], rtol=0, atol=0)
    assert calls[0][0] == IMAGENET_WEIGHTS_URL
    assert calls[0][1]["check_hash"] is True
    assert calls[0][1]["weights_only"] is True
    state.pop("features.0.0.weight")
    with pytest.raises(RuntimeError, match="Missing key"):
        stem_block(pretrained=True)


def test_entrypoints_and_inference_do_not_import_torchvision():
    script = r'''
import importlib.abc
import sys
class BlockVision(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torchvision" or fullname.startswith("torchvision."):
            raise AssertionError("Unexpected torchvision import: " + fullname)
sys.meta_path.insert(0, BlockVision())
import main
import load_checkpoint
import profile_model
import torch
from PIL import Image
from segmentation import Config
from segmentation.models import build_model
torch.set_num_threads(2)
model = build_model(Config(stem_pretrained=False, use_aux=False)).eval()
with torch.no_grad():
    assert model(torch.zeros(1, 3, 65, 81)).shape == (1, 150, 65, 81)
report = profile_model.profile_model(model, torch.zeros(1, 3, 64, 64))
assert report["parameters"] == 1029654
assert report["attention_macs"] > 0
assert not any(n == "torchvision" or n.startswith("torchvision.") for n in sys.modules)
print("All entrypoints and model inference work without torchvision")
'''
    result = subprocess.run([sys.executable, "-B", "-c", script],
                            cwd=Path(__file__).resolve().parents[1],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
