> 历史设计资料：本文件保留迁移前内容。当前实现与验证结果请以 README.md 为准。

# SeaFormer + MobileV3 C3 单路头

本实现使用用户确认的 `MobileV3FPNHead`，仅保留 **C3 context → lateral[2] → refine[2] → classifier**。默认 150 类、head_width=128。没有 C1/C2 分割分支、多尺度 top-down/cat/fuse、原上传 segment_block 或辅助分割头。

## 结构与优化

```text
image [B,3,H,W]
  │
  └─ backbone.stem：原 MobileNetV2 features[0:8]
       features[7] depthwise stride=1，默认 ImageNet 初始化
       spatial [B,64,ceil(H/8),ceil(W/8)] ──────────────────┐
          │                                               │
          └─ context1 ×1  (1/8)                          │
               └─ down1 → context2 ×2 (1/16)              │
                    └─ down2 → context3 ×2 (1/32)         │
                         └─ fusion3：一次 enhanced 融合 ←─┘
                              └─ c3_projection：DW stride2 + PW 64→128
                                   C3 [B,128,ceil(H/16),ceil(W/16)]
                                      │
                                      └─ decode_head: MobileV3C3Head
                                           context: 2 × InvertedResidualV3
                                                    128→768→128, kernel=5, GN
                                           lateral: Conv1×1 128→head_width
                                           refine: 原 SeparableConv(DW + PW, GN)
                                           classifier: Conv1×1 head_width→150
                                              │
                                              └─ logits [B,150,ceil(H/16),ceil(W/16)]
                                                   engine 对齐原始标签尺寸后计算 loss/指标
```

相较上传网络，将最高分辨率 context 从 2 块减为 1 块，将 3 次空间融合减为 1 次最终融合。C3 接口适配放在 backbone 末端，既保持上传 stem 的原始结构，也保留原 MobileV3 C3 头的两个完整 context 块。默认 depths=(1,2,2)，attention="multiscale"；还支持 "original"、"lepe" 和可学习 squeeze。三个上传注意力实现的重复代码合并，数学计算与对应上传变体一致。

head 模式中，stem、所有 context、fusion3 和 c3_projection 全部冻结；decode_head.context/lateral/refine/classifier 全部训练。沿用工程的 `requires_grad`、`no_grad` 和冻结模块 eval 控制，并在每轮结束精确比较 backbone 参数及 BatchNorm buffers。

## 安装

已验证 Python 3.12、torch 2.9.1+cpu、torchvision 0.24.1+cpu、NumPy 2.5.4、Pillow 13.0.0。CPU 环境示例：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-dev.txt
```

CUDA 环境需从 PyTorch 官方安装匹配的 torch/torchvision 构建。本次未验证 GPU。

本云实例可以直接使用：

```bash
source /workspace/.venvs/segmentation-new/bin/activate
export TORCH_HOME=/workspace/.cache/torch
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
cd /workspace/Segmentation_new
```

首次从零训练默认从 `download.pytorch.org` 加载 ImageNet MobileNetV2 权重。断点恢复、加载完整 backbone 和独立结构统计不需要重新下载。离线调试可传 `--no-pretrained-stem`；该选项只改变初始化，不改变 stem 结构。

## 训练入口

数据结构沿用 ADE20K：`images/training`、`images/validation`、`annotations/training`、`annotations/validation`，同名 JPG/PNG 配对。PNG 是原始单通道标签：0 忽略，1..150 对应类别。默认 data_root 为项目根目录下的 ADEChallengeData2016，也可以传绝对路径。

没有新 backbone 的完整权重时，先联合训练；SeaFormer 的 pretrain 模式也强制使用同一个 MobileV3C3Head，不会切换成 SimpleHead：

```bash
python main.py --mode pretrain \
  --data-root /path/to/ADEChallengeData2016 \
  --output-dir runs/ade20k_seaformer/pretrain
```

然后冻结整个 backbone，仅训练新建的 C3 头：

```bash
python main.py --mode head \
  --data-root /path/to/ADEChallengeData2016 \
  --backbone-checkpoint runs/ade20k_seaformer/pretrain/backbone_best_miou.pth \
  --output-dir runs/ade20k_seaformer/head
```

`mode=head` 强制 `trainable_prefixes=("decode_head",)`，并要求真实完整 backbone 权重。只有 ImageNet stem 预训练时，其他 context/fusion/projection 仍未训练，不应直接冻结它们开展正式实验。

head 模式会重新初始化分割头，与原工程行为一致。继续同一个实验中的已训练 head 请用 `--resume`，并保持原实验的 epochs、输入尺寸、batch size、种子、数据目录、attention/depths 等设置：

```bash
python main.py --mode head \
  --data-root /path/to/ADEChallengeData2016 \
  --output-dir runs/ade20k_seaformer/head \
  --resume runs/ade20k_seaformer/head/last.pth
```

上述命令使用 Config 默认 epochs=5、batch_size=4、crop_size=256。epochs 是原计划总轮数，续训时不要改为剩余轮数。非默认参数通过相同 CLI 选项或 Config 重现。配置仍集中于 `segmentation/config.py`，PyCharm 可以直接运行 main.py。

上传的 `load_checkpoint.py` 读取旧网络的 `model_state_dict/model_config`。新网络更改了 context 深度、融合拓扑和 C3 适配，**该旧完整 checkpoint 不能直接作为新网络的完整 backbone 或 resume 断点**。这里保留原始 stem 初始化，并使用本工程导出的新架构 backbone checkpoint。没有声称旧完整权重可无损迁移。

## Params / MACs

每次训练开始前，`profile_training()` 从真实训练 DataLoader 取一个预处理后的 batch，打印 backbone/head/全网总参数、可训练参数、MACs/图和 MACs/batch，保存到本次 output_dir 的 `model_profile.json`。统计结束恢复随机状态与模块 train/eval 状态，不更新 BN。`run_experiment()` 在训练完成后也返回该统计字典。

独立统计不依赖数据或权重；`--mode head` 可以查看冻结后的可训练参数：

```bash
python profile_model.py --mode head --size 256 --batch-size 1 --output profile-256.json
python profile_model.py --mode head --size 512 --batch-size 1 --output profile-512.json
```

默认 multiscale、depths=(1,2,2)、head_width=128 的实测值：

| 部分 | Params | 仅 head 模式可训练 Params | 256×256 MACs/图 | 512×512 MACs/图 |
| --- | ---: | ---: | ---: | ---: |
| backbone | 796,704 | 0 | 329,715,712 | 1,313,435,648 |
| MobileV3 C3 head | 1,083,926 | 1,083,926 | 124,682,240 | 496,959,488 |
| 全网 | 1,880,630 | 1,083,926 | 454,397,952 | 1,810,395,136 |

MACs 包含 Conv2d/Linear 和 SeaFormer 的四次轴向 QKᵀ/AV 矩阵乘法。1 次乘加计 1 MAC。默认模型 256 输入的注意力矩阵部分为 638,976 MACs；512 输入为 2,555,904 MACs。未统计插值、softmax、归一化、池化、逐元素门控/加权 squeeze、损失及反向传播；因此这些数字不是全部算子的 FLOPs，也不是时延测量。

与上传网络 `multiscale + enhanced fusion + original segment head + use_aux=False` 对比，同口径 MACs 在 256 输入从 516,620,288 降至 454,397,952（约 12.04%），512 输入约下降 12.07%。由于按要求保留了 MobileV3 C3 头两个较大的 context 块，总参数从原上传组合的 1,029,654 增至 1,880,630。此方案优化计算量，并未同时降低该对照的参数量。对照原网络在内部上采样，新模型在现有 engine 中上采样；两者插值均不计入上述 MACs。真实 ADE20K 精度和设备时延尚未验证。

## 验证

```bash
python -m pytest -q tests
```

17 项测试通过：原始 stem 结构/计算、MobileV3 C3 分支逐项等价、三种 attention 和两种 squeeze 的梯度/矩阵 MACs、非方形奇数输入、冻结参数与 BN buffers、统计无副作用、两阶段训练、严格权重导出/加载、实际断点续训及旧三种 backbone 回归。

另使用原上传文件做了数值对照：三种 attention × 两种 squeeze 在加载相同 state_dict 后输出逐项相同；新 stem 的 ImageNet 权重和输出与上传 stem 逐项相同。CLI 的 pretrain 和 head 均在合成 ADE20K 格式样本上完成一个 epoch，包含训练前统计、验证与保存。合成数据用于功能检查，不代表真实分割效果。

## 项目结构

```text
Segmentation_new/
├── main.py                         # 训练入口，Config + CLI 覆盖
├── profile_model.py                # Params/MACs，包含轴向注意力矩阵乘法
├── requirements.txt
├── requirements-dev.txt
├── README.md
├── SEA_FORMER.md                    # 当前网络和训练说明
├── instruction.md                  # 旧 SmallCNN/FPN 配置说明
├── segmentation/
│   ├── __init__.py
│   ├── config.py                   # 默认 seaformer + mobile_c3
│   ├── trainer.py                  # 训练前统计、训练、评估、保存/续训
│   ├── engine.py                   # logits 尺寸对齐、loss、验证
│   ├── dataset.py
│   ├── metrics.py
│   ├── checkpoints.py              # 新架构标识和配置一致性检查
│   ├── freezing.py                 # 权重和 BN buffers 冻结
│   ├── utils.py
│   └── models/
│       ├── __init__.py
│       ├── segmentor.py            # backbone/decode_head 组装
│       ├── seaformer_stem.py       # 保留上传 MobileNetV2 stem
│       ├── seaformer_attention.py  # original/multiscale/lepe 共用实现
│       ├── seaformer.py            # 优化 backbone + C3 适配
│       ├── mobile_v3_head.py       # 新 MobileV3C3Head，保留旧完整头
│       ├── mobile_v3_cnn.py
│       ├── heads.py                # 原 FPNHead/SimpleHead
│       └── cnn.py
└── tests/
    └── test_seaformer.py
```

输出目录会包含 config.json、samples.json、trainable_parameters.txt、model_profile.json、history.csv、完整模型断点和独立 backbone 权重。运行生成文件与数据集不包含在交付代码包中。
