# 按最新 V2 backbone 调整 mini head

依据：GitHub `main` 的 V2 配置提交 `b8036a4`（2026-11-11）。
`attention_type="original"`、`use_learnable_squeeze=True`；其余 backbone 设置为 enhanced fusion、dim=64、key_dim=16、heads=4、attn_ratio=2、pos_shape=16、ffn_ratio=2、两种 expansion_ratio=2、三阶段各两个 context block。原 MobileNetV2 stem 不变。

V2 的 `head_name`、`use_aux` 和 Windows 数据路径不属于 backbone 设置，不复制到 mini/V3 的 head 或数据配置。V2 配置本身未改动。

## 默认结构与选择依据

```text
V2 backbone f3: [B,64,H/8,W/8]
  → DW3×3 stride2, BN, ReLU6, PW64→128, BN
  → 一个 IR：PW128→368, GN, Hardswish
             DW5×5, GN, Hardswish
             SE368→96→368
             PW368→128, GN, residual
  → lateral 1×1, 128→128
  → SepConv：DW3×3 + PW1×1，均含 GN/Hardswish
  → classifier 1×1, 128→150
```

head 只有 C3 一路、一个 IR/SE context 和一个 refine SepConv；没有 C1/C2 的 lateral、refine、top-down、拼接或融合模块。输出步长为 16。

head 预算为 `1,000,000 - 757,860 = 242,140`；严格小于 1M 时至多使用 242,139 个参数。

| 主通道 C | 扩展通道 E | Head Params | 全网 Params | 全网 MACs，256² |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 120 | 50,566 | 808,426 | 396,031,488 |
| 64 | 512 | 235,990 | 993,850 | 411,508,736 |
| 96 | 432 | 234,342 | 992,202 | 419,867,136 |
| 96 | 448 | 241,478 | 999,338 | 420,759,552 |
| 128 | 352 | 226,030 | 983,890 | 426,652,160 |
| **128** | **368** | **239,318** | **997,178** | **427,811,840** |
| 128 | 376 | 243,142 | 1,001,002 | 428,388,864 |

默认采用 C128/E368，使 context 输出、空间特征细化和分类层均获得更宽的特征。C64/E512 的 SE 使用 131,712 个参数，占该 head 约 56%；C128/E368 的 SE 使用 71,120 个参数，占该 head 约 30%，更多预算用于空间特征处理。代价是后者计算量略大。

在当前结构、C128、扩展通道按 8 对齐的约束下，E368 是预算内最大的扩展通道；下一个 E376 即超预算。此结论不意味着它是所有可能结构中精度最优的模型。希望留更多参数余量时，可用 C128/E352。

## Params 逐项计算

使用 `sum(p.numel() for p in model.parameters())`；包括 bias、BN/GN 仿射参数和被冻结参数，不计 BN running statistics 等 buffers。

| 模块 | 计算项 | Params |
| --- | --- | ---: |
| 输入适配 | DW64×9 + BN128 + PW64×128 + BN256 | 9,152 |
| IR expand | PW128×368 + GN736 | 47,840 |
| IR depthwise | DW368×25 + GN736 | 9,936 |
| IR SE | 368×96 + 96 + 96×368 + 368 | 71,120 |
| IR project | PW368×128 + GN256 | 47,360 |
| lateral | 128×128 + 128 | 16,512 |
| refine | DW128×9 + GN256 + PW128×128 + GN256 | 18,048 |
| classifier | 128×150 + 150 | 19,350 |
| **head 合计** | **一个 context 合计 176,256** | **239,318** |
| backbone | 三个工程相同配置，实际模型参数求和 | 757,860 |
| **全网** | **757,860 + 239,318** | **997,178** |

通用公式（150 类、E>C、C/E 均按 8 对齐）：

```text
S = 8 × ceil(E / 32)
P_context = (2C + 30)E + 2ES + S + 2C
P_head = 854 + 230C + 2C² + P_context
P_total = 757860 + P_head
```

新默认值距离 1M 为 2,822 参数。此前完整 mini 默认为 multiscale / squeeze=False、C64/E120，共 998,278 参数；现在 backbone 节省 189,852，head 增加 188,752，全网减少 1,100。

## MACs 计算与复现

输入 `[1,3,256,256]`，1 次乘加记为 1 MAC。Conv2d 按 `输出元素数 × (输入通道/groups) × kernel_H × kernel_W`；Linear 按 `输出元素数 × 输入维度`。SeaFormer QKᵀ/AV 按实现的行/列注意力另行计数，不与投影卷积重复。

| 部分 | MACs |
| --- | ---: |
| backbone | 385,425,408 |
| 输入适配 | 2,244,608 |
| 一个 IR/SE | 26,543,104 |
| lateral | 4,194,304 |
| refine | 4,489,216 |
| classifier | 4,915,200 |
| **head 合计** | **42,386,432** |
| **全网** | **427,811,840** |

不计 bias 加法、归一化、激活、池化、softmax、插值、逐元素门控/加权 squeeze、loss、反向和优化器。MACs 不是运行时延。注意力以本实现的 `B × heads × (key_dim + value_dim) × (H² + W²)` 计 QKᵀ/AV。

```bash
cd Segmentation_V2_V3_mini
python profile_model.py --size 256 --mode head --output model_profile.json
python profile_model.py --size 256 --head-width 64 --context-expanded-channels 512
python profile_model.py --size 256 --head-width 128 --context-expanded-channels 352
python -m pytest -q tests
```

`model_profile.json` 包含逐层卷积、线性、注意力统计及 backbone/head 分项。head-only 模式的可训练参数为 239,318，全网 Params 仍是 997,178。

## 验证范围与训练预期

CPU 测试：mini 46 项通过，V2_V3 44 项通过。覆盖单 C3/单 IR/单 SE、非方形输入、前向与有限梯度、冻结模块参数及 BN buffers 不变、三选项 pipeline、重复调用 head 编号、断点续训、推理和旧宽度配置。V2、V2_V3、mini 的 backbone 严格交换权重后，四张特征图逐元素完全一致。

独立执行了新默认模型的联合训练、head-only、backbone-only 三种范围的单步反向与优化，以及完整权重严格保存/恢复检查。所有应训练参数都有有限梯度。

相对 C64/E120，增加通道缓解了 head 的容量约束；相对原 V3 双 context head，mini 仍减少一个 IR 和扩展通道，表达能力可能下降。现有少量合成/训练数据不足以判断 ADE20K mIoU，也无法证明本方案优于 C64/E512 或 C96/E448。应在相同 backbone 初始化、数据、训练预算和多随机种子下比较精度。

V2_V3 同步使用相同的新 backbone 默认值，原 head 保留不变：全网 1,850,938 Params、512,352,256 MACs。Mini 的新宽度需训练新的 head；旧 backbone 仅在 attention/squeeze 等配置相同时复用，旧 mini 的完整模型可通过显式设置原宽度继续使用，见 README。
