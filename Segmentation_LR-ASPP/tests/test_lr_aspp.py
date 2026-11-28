import pytest
import torch
from torch.nn import functional as F

from segmentation.models import LRASPPHead


@pytest.mark.parametrize('feature_hw,gate_hw', [((32, 32), (1, 1)), ((129, 177), (2, 3)),
                                                ((13, 177), (1, 3))])
def test_figure_10_gate_and_logit_fusion(feature_hw, gate_hw):
    torch.manual_seed(7)
    head = LRASPPHead(4, num_classes=3, width=8).eval()
    low, fused = (torch.randn(1, 4, *feature_hw, requires_grad=True) for _ in range(2))
    output = head((low, fused))

    # 独立按图 10 取大窗口的平均值，检查池化步幅、门控和两路分类的相加顺序。
    high = F.avg_pool2d(fused, 2, 2, ceil_mode=True, count_include_pad=False)
    h, w = high.shape[-2:]
    kh, kw = min(49, h), min(49, w)
    pooled = torch.stack([
        torch.stack([high[:, :, i:i + kh, j:j + kw].mean(dim=(-2, -1))
                     for j in range(0, w - kw + 1, 20)], dim=-1)
        for i in range(0, h - kh + 1, 16)
    ], dim=-2)
    assert pooled.shape[-2:] == gate_hw
    gate = torch.sigmoid(head.gate[0](pooled))
    gated = head.project(high) * F.interpolate(gate, size=(h, w), mode='bilinear', align_corners=False)
    expected = head.low_classifier(low) + head.high_classifier(
        F.interpolate(gated, size=feature_hw, mode='bilinear', align_corners=False))
    torch.testing.assert_close(output, expected)
    output.square().mean().backward()
    for grad in (low.grad, fused.grad, head.gate[0].weight.grad):
        assert torch.isfinite(grad).all() and grad.abs().sum() > 0
