# Segmentation_V2_V3_mini

独立的 mini 训练工程，与 `segmentation_V2`、`segmentation_V2_V3` 并列。默认模型为已核验的 **一个 128→368→128 IR＋SE context、C3 单路 head**，全网 **997,178 Params**。

## 模型

- Backbone 与另外两个工程的相同配置一致：原始 MobileNetV2 stem、六个 SeaFormer context block、三次 enhanced fusion。
- 仅把最终 fusion 的 f3 送入 head，不建立 C1/C2 分支。
- Head 默认主通道 128；context 数量固定 1，DW5×5、SE368→96→368、线性投影和残差连接均保留。
- 输入适配为 DW3×3 stride2 + PW64→128；后接 context、lateral128、SepConv128、classifier150。输出步长 16。
- 图像读取与增强使用 Pillow/NumPy/PyTorch，不依赖 torchvision。

默认与 V2 提交 `b8036a4` 一致的 `original` attention、`seaformer_learnable_squeeze=True`、150 类，输入 `[1,3,256,256]`：

| 部分 | Params | MACs/图 |
| --- | ---: | ---: |
| backbone | 757,860 | 385,425,408 |
| mini head | 239,318 | 42,386,432 |
| **全网** | **997,178** | **427,811,840** |

Params 包含冻结参数和归一化仿射参数，不包含 buffers。MACs 计卷积、线性和 attention QKᵀ/AV；不计插值、归一化、池化、逐元素运算、loss 和反向。

## 启动

使用 Python 3.11 或兼容环境，在本工程目录运行：

```bash
cd Segmentation_V2_V3_mini
python -m pip install -r requirements-dev.txt
python main.py
```

在 `segmentation/config.py` 设置 `data_root`，或使用 `--data-root`。ADE20K 数据布局：

```text
ADEChallengeData2016/
├── images/
│   ├── training/*.jpg
│   └── validation/*.jpg
└── annotations/
    ├── training/*.png
    └── validation/*.png
```

标注使用原始 ADE20K 编码：0 为忽略，1..150 为类别。默认训练 5 epochs、训练子集 1000 张、验证 200 张；完整训练请设置实际预算，并把两个 `max_*_samples` 设为 0。

`main.py` 默认显示三选项菜单；PyCharm 直接运行即可。`pipeline.py` 是相同菜单的兼容入口。所有相对配置路径按本工程目录解析，不受 IDE 工作目录影响。

| 选项 | 流程 | 保存位置 |
| --- | --- | --- |
| 1 | pretrain 联合训练 backbone + mini head，然后加载本次最佳 backbone，冻结它并训练新 mini head | 同一 `runN/pretrain`、`runN/head1` |
| 2 | 用户选择 checkpoint，提取 backbone 并冻结，训练新 mini head | 该来源对应 run 的下一个 `headN` |
| 3 | 用户选择完整 mini 模型，冻结已训练 head，仅训练 backbone（custom） | 新建 `runN/custom` |

选项 2/3 必须手动提供权重路径。选项 3 若提供独立 `backbone_best_miou.pth` / `backbone_last.pth`，同目录须保留对应 `best_miou.pth` / `last.pth`；程序先核对 backbone 张量完全匹配，再加载完整 mini 模型。

也可使用非交互命令：

```bash
python main.py --choice 1 --data-root /path/to/ADEChallengeData2016 --pretrain-epochs 50 --head-epochs 20
python main.py --choice 2 --checkpoint runs/ade20k_v2_v3_mini/run1/pretrain/backbone_best_miou.pth
python main.py --choice 3 --checkpoint runs/ade20k_v2_v3_mini/run1/head1/best_miou.pth
```

未指定阶段轮数时，两阶段各用 `Config.epochs`。第一阶段失败或未生成最佳 backbone 时，不启动 head 阶段。
默认 stem 加载 ImageNet 权重；离线测试可加 `--no-pretrained-stem`。加载已有完整模型、backbone 或续训时不会下载 stem 权重。

## 输出与权重归属

```text
runs/ade20k_v2_v3_mini/
├── run1/
│   ├── pretrain/
│   ├── head1/
│   └── head2/
└── run2/
    └── custom/
```

编号从磁盘读取，程序重启后再次调用同一 backbone 继续分配 head2、head3。每个阶段保存：

- `config.json`、`model_profile.json`、`pipeline.json`。
- `trainable_parameters.txt`、`samples.json`、`history.csv`。
- `best_miou.pth`、`best_accuracy.pth`、`last.pth`、定期 `epoch_XXX.pth`：完整模型与训练状态，包含 head。
- `backbone_best_miou.pth`、`backbone_last.pth`：可单独复用的 backbone。

训练前按真实预处理后的 batch 打印并保存 Params/MACs。head-only 每轮检查 backbone 参数及 BN buffers 不变；backbone-only 对 head 做相同检查。

本输出根目录内的标准 checkpoint 归入原 run。**外部 checkpoint，包括另外两个工程的 runN 权重，在 mini 输出根目录独立绑定**，不把 head 结果写回原工程。绑定使用路径与内容指纹，保存在 `.pipeline_sources.sqlite3` 和 `runN/backbone_source.json` 中。

## 复用 V2/V3 backbone

```bash
python main.py --choice 2 --checkpoint ../segmentation_V2_V3/runs/ade20k_v2_v3/run1/pretrain/backbone_best_miou.pth
```

backbone 结构与预处理配置相同即可严格加载；attention 类型或 learnable squeeze 不同会明确拒绝。
旧工程的完整模型也可作为选项 2 的来源，只提取 backbone。

**旧版完整 head 不能作为 mini 的完整模型初始化、续训或选项 3 的冻结 head。** 请先用旧 backbone 训练 mini head，再使用产生的完整 mini checkpoint。Mini checkpoint 保存 `head_name="mobile_c3_mini"`、`head_width` 和 `context_expanded_channels`，完整加载与续训均校验这些字段。

## 单阶段和续训

```bash
python main.py --mode pretrain --data-root /path/to/ADEChallengeData2016
python main.py --mode head --backbone-checkpoint runs/ade20k_v2_v3_mini/run1/pretrain/backbone_best_miou.pth
python main.py --mode head --resume runs/ade20k_v2_v3_mini/run1/head1/last.pth
python main.py --mode custom --trainable-prefixes backbone --resume runs/ade20k_v2_v3_mini/run2/custom/last.pth
```

`--mode` / `--resume` 选择单阶段流程，不与 `--choice`、`--checkpoint` 或阶段轮数混用。
`--model-checkpoint` 初始化整个 mini 模型，`--backbone-checkpoint` 只初始化 backbone，两者不能同时使用。
续训从原 checkpoint 目录继续，不创建新编号；保持原来的训练范围、数据、网络、总轮数和超参数。旧实验已完成时不会重复 epoch。
显式 `output_dir` 已存在时，新实验自动追加 `_2`、`_3`；指定 resume 时则恢复原目录。

## Head 参数预算与配置

默认 `head_width=128`、`context_expanded_channels=368`。一个 IR 的路径为 128→368→128；head 主干、lateral、refine 与 classifier 的输入宽度一起由 `head_width` 控制。SE 中间通道按原规则向上对齐为 96。

```bash
python main.py --choice 1 --head-width 128 --context-expanded-channels 368
python profile_model.py --size 256 --mode head --head-width 128 --context-expanded-channels 368
```

主通道须为正整数且为 8 的倍数；扩展通道须大于主通道且为 8 的倍数，保留真实 PW 扩展。改变任一宽度需新建 head 实验，不能直接续训不同尺寸的完整 head。同配置的 backbone 可以复用。

当前 757,860 Params backbone、150 类下的实测比较：

| 主通道 / 扩展通道 | Head Params | 全网 Params | 全网 MACs/图（256²） |
| --- | ---: | ---: | ---: |
| 64 / 120（原 mini head） | 50,566 | 808,426 | 396,031,488 |
| 64 / 512 | 235,990 | 993,850 | 411,508,736 |
| 96 / 448 | 241,478 | 999,338 | 420,759,552 |
| 128 / 352 | 226,030 | 983,890 | 426,652,160 |
| **128 / 368（默认）** | **239,318** | **997,178** | **427,811,840** |
| 128 / 376（超预算） | 243,142 | 1,001,002 | 428,388,864 |

选择 128/368 是为了同时增加 context、空间特征细化与分类层的容量。它是预算内的工程选择，尚无 ADE20K mIoU 证据证明它优于表中其他结构。完整参数推导见 [HEAD_BUDGET.md](HEAD_BUDGET.md)。

只有上述默认设置保证全网小于 1M。更换 backbone attention/squeeze 或继续增加宽度需重新统计；程序允许超预算实验并报告实际值。

### 旧配置兼容

原 mini 的默认值是 W64/E120、multiscale attention、不使用 learnable squeeze。旧完整 checkpoint 的训练和续训需显式恢复这些设置（并保持原训练超参数）：

```bash
python main.py --mode head --resume /path/to/old/head1/last.pth --head-width 64 --context-expanded-channels 120 --seaformer-attention multiscale --no-learnable-squeeze
python profile_model.py --size 256 --head-width 64 --context-expanded-channels 120 --seaformer-attention multiscale --no-learnable-squeeze
```

第二条命令仍统计为 998,278 Params / 445,306,368 MACs。仅复用旧 backbone 时可独立选择新 head 宽度，但仍须匹配来源 backbone 的 attention/squeeze；该组合未必低于 1M。推理入口从完整 checkpoint 读取两种宽度和 backbone 设置，不需要手动转换权重。

## 独立统计与推理

```bash
python profile_model.py --size 256 --mode head --output model_profile.json
python load_checkpoint.py runs/ade20k_v2_v3_mini/run1/head1/best_miou.pth --image /path/to/image.jpg --output prediction.png
```

推理从 checkpoint 恢复主通道、扩展通道与网络配置，无需手动重新填写宽度。输出 PNG 为 0..149 类别索引，不是彩色可视化标签。

## 验证与目录

```bash
python -m pytest -q tests
```

测试覆盖默认参数/MACs、单个 IR/SE/C3 路径、扩展宽度、无 torchvision 导入、两工程 backbone 严格互换、三选项 pipeline、重启 head 编号、外部来源隔离、完整 checkpoint 校验、真实 epoch 续训和图像推理。
测试使用 CPU 与合成 ADE20K 格式数据，只验证工程与训练流程；没有实际 ADE20K 精度保证。
交叉工程核验在仓库内运行；单独复制 mini 目录时，缺少兄弟工程的交叉核验会跳过，训练和其他测试不依赖它们。

```text
Segmentation_V2_V3_mini/
├── main.py / pipeline.py
├── profile_model.py / load_checkpoint.py
├── requirements.txt / requirements-dev.txt
├── segmentation/
│   ├── config.py / cli.py / pipeline.py / run_paths.py
│   ├── trainer.py / checkpoints.py / freezing.py
│   ├── dataset.py / engine.py / metrics.py / utils.py
│   └── models/
│       ├── segmentor.py / mini_head.py / mobile_v3_blocks.py
│       ├── seaformer.py / uploaded_backbone.py
│       ├── network.py / Stem_block.py / fusion_block.py
│       └── seaformer_attention*.py
└── tests/
```
