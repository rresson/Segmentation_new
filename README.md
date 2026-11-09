# Segmentation_new

默认模型为 **原始 MobileNetV2 stem + 优化 SeaFormer backbone + MobileV3FPNHead 的 C3 单路分割头**。

完整结构、安装、两阶段训练、断点恢复及实测统计见 [SEA_FORMER.md](SEA_FORMER.md)。

```bash
python main.py --mode pretrain --data-root /path/to/ADEChallengeData2016
python main.py --mode head --data-root /path/to/ADEChallengeData2016
python profile_model.py --mode head --size 256
python -m pytest -q tests
```

先完成新 backbone 的预训练，再冻结 backbone 训练 head。仅有 stem 的 ImageNet 权重不足以把整个新 backbone 当作已训练模型。

`instruction.md` 是旧 SmallCNN/FPN 的配置说明；当前 SeaFormer 使用 `SEA_FORMER.md`。
