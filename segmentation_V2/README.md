# 上传 MobileNetV2-stem 网络的训练器适配

此文件夹是独立项目。先从仓库根目录执行 `cd segmentation_V2`，再运行本文中的安装、训练、统计和测试命令。

目标是用户上传的 `network.py: main_network`：MobileNetV2 stem、六个 context block、三次 fusion、原始 segment_block/segment_block2 和可选 auxiliary_head。保留其模块和 forward 计算顺序，不使用 MobileV3Backbone、MobileV3FPNHead 或上一版优化 SeaFormer。

适配以 GitHub `rresson/Segmentation_new` 提交 `867c59a` 的训练框架为基准。完整项目包可直接解压；替换包中的文件均是完整文件，不是 diff。上传的网络文件放在项目根目录，保留 `Stem_block.py` 的大小写。

## 可以直接复用

| 来源 | 文件 | 说明 |
| --- | --- | --- |
| GitHub | segmentation/trainer.py | 原封不动复用统计、训练、验证、保存、续训循环 |
| GitHub | segmentation/dataset.py | ADE20K 图片、原始标签映射、增强与采样 |
| GitHub | segmentation/metrics.py | mIoU、像素准确率 |
| GitHub | segmentation/freezing.py | requires_grad、模块 eval、冻结参数及 BN buffers 检查 |
| GitHub | segmentation/utils.py | 随机种子、项目根目录路径解析 |
| GitHub | segmentation/__init__.py | Config/run_experiment 公共入口 |
| GitHub | requirements-dev.txt、.gitignore | 开发依赖和忽略规则 |
| 上传文件 | fusion_block.py | 两种 fusion、两种原始主分割头 |
| 上传文件 | seaformer_attention.py | original attention |
| 上传文件 | seaformer_attention_multiscale.py | multiscale attention |
| 上传文件 | seaformer_attention_lepe.py | lepe attention |

上表文件已做字节比对，交付包与对应来源一致。

## 要替换的完整文件

| 文件 | 修改目的 |
| --- | --- |
| main.py | 选择原网络 attention/fusion/segment，支持主/辅助头开关和三种权重路径 |
| profile_model.py | 统计原始 SeaAttention 的 QKᵀ/AV；开启 aux 时统计主头和辅助头 |
| segmentation/config.py | 对应原 main_network 的完整配置；pretrain 保留原主分割头 |
| segmentation/models/segmentor.py | 将原模型挂接为 backbone 和 decode_head，保持计算图与权重张量不变 |
| segmentation/models/__init__.py | 导出适配器，停止引用旧 MobileV3 模型组件 |
| segmentation/checkpoints.py | backbone/完整权重初始化、原 model_state_dict 格式映射、严格续训检查 |
| segmentation/engine.py | 支持训练时 (main_logits, aux_logits)，计算加权辅助损失 |
| requirements.txt | 增加原上传模块依赖的 torchinfo |
| Stem_block.py | 仅增加 pretrained=True 开关；保留原 features[0:8]、权重默认值和 stride 修改 |
| network.py | 仅增加 stem_pretrained=True 并转交 stem；不改变网络结构与 forward |
| load_checkpoint.py | 替换本机硬编码路径，支持旧/新完整权重格式和图片推理 |

新增验证文件：`tests/test_mobile_v2.py`。旧 `tests/test_seaformer.py` 针对上一版优化 MobileV3 C3 网络，不适用于此专用适配项目；请运行下面指定的新测试文件。旧 `segmentation/models/cnn.py`、`heads.py`、`mobile_v3_*`、`seaformer*` 不参与新适配器构建，可以不复制到新项目。

## 模块划分

```text
backbone
  stem                       原上传 MobileNetV2 stem
  context1/context2/context3  每级两个 context block
  context_down1/context_down2
  fusion1/fusion2/fusion3

decode_head
  segment_head               原 segment_block 或 segment_block2
  auxiliary_head             use_aux=True 时存在
```

mode=head 冻结整个 backbone，包括 stem、attention、FFN、downscaling、fusion 的参数和 BatchNorm buffers。启用辅助头时，head 训练会同时更新主头和辅助头；只训练主头请设置 `use_aux=False` 或 `--no-aux`。custom 模式允许例如 `( "decode_head", "backbone.context3" )` 这样的范围。

pretrain 模式训练整个原网络，**不会替换成 SimpleHead**。pretrain 和 head 使用同一选定的主分割头结构。原 stem 的 ImageNet 权重仅覆盖 stem，其余 backbone 部分仍需训练。

## 安装和运行

已验证 Python 3.12、PyTorch 2.9.1+cpu、torchvision 0.24.1+cpu、torchinfo 1.8.0。CPU 示例：

```bash
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-dev.txt
python -m pytest -q tests/test_mobile_v2.py
```

GPU 环境请使用匹配的 PyTorch/torchvision CUDA 构建。首次新实验默认加载 ImageNet MobileNetV2 权重，使用系统正常证书验证。`--no-pretrained-stem` 可用于离线/随机初始化实验；恢复完整权重和独立统计会自动跳过 ImageNet 初始化。

在 `segmentation/config.py` 修改默认值，PyCharm 直接运行 main.py；也可以从项目根目录使用命令行覆盖。默认 attention=multiscale、fusion=enhanced、主头=original、use_aux=True、aux_loss_weight=0.4、epochs=5、batch_size=4。dim 固定 64，对应原 stem 输出，保持原始网络结构。

准备 ADEChallengeData2016 的 images/training、images/validation、annotations/training、annotations/validation。标签为原始单通道 PNG（0 忽略，1..150 为类别）。

1. 没有已训练 backbone，先联合训练：

```bash
python main.py --mode pretrain --data-root /path/to/ADEChallengeData2016
```

2. 加载上述实验导出的 backbone，冻结后训练新主头/辅助头：

```bash
python main.py --mode head --data-root /path/to/ADEChallengeData2016 \
  --backbone-checkpoint runs/ade20k_mobile_v2/pretrain/backbone_best_miou.pth
```

3. 如果已有原上传网络训练出的完整 checkpoint，可以保留其中的主头权重，再冻结 backbone 继续训练 head（新建优化器）：

```bash
python main.py --mode head --data-root /path/to/ADEChallengeData2016 \
  --model-checkpoint /path/to/original_model.pth \
  --output-dir runs/ade20k_mobile_v2/head_from_original
```

这里要求 Config 与原 checkpoint 的 model_config 一致。比如原模型是 original attention、enhanced fusion、multilevel 主头、learnable squeeze、无辅助头，应额外指定：

```bash
--attention-type original --fusion-type enhanced --segment-type multilevel --learnable-squeeze --no-aux
```

同样的网络配置必须用于它的后续实验/恢复，不能把示例选项当作任意 checkpoint 的通用参数。如果原权重没有 auxiliary_head 而 Config.use_aux=True，严格加载会报错；设置 `--no-aux` 才能按其实际结构加载。

4. 本训练器生成的同一实验断点续训：

```bash
python main.py --mode head --data-root /path/to/ADEChallengeData2016 \
  --output-dir runs/ade20k_mobile_v2/head \
  --resume runs/ade20k_mobile_v2/head/last.pth
```

保留原实验总 epochs、学习率、batch size、尺寸、采样限制、数据目录、模式、attention/fusion/head/aux 设置；output_dir 必须是 resume 文件所在目录。原上传 checkpoint 支持**权重初始化**，但不能假定它具有本训练器兼容的 optimizer/scheduler/scaler 状态，因此不直接用作 resume。

| 配置字段 | 加载范围 | 是否恢复优化器和轮次 |
| --- | --- | --- |
| backbone_checkpoint | 独立 backbone，或从完整权重中提取 backbone；新建 head | 否 |
| model_checkpoint | 原/新完整网络权重，含原主头及存在的辅助头 | 否 |
| resume | 本训练器同一实验的完整状态 | 是 |

前两项不能同时配置。原始权重名 `stem.* / context* / fusion*` 映射为 `backbone.*`，`segment_head.* / auxiliary_head.*` 映射为 `decode_head.*`；不裁剪权重、不静默忽略主网络缺失参数，随后 strict=True 加载。

## 训练前 Params/MACs

trainer.py 仍在第一个 epoch 前调用 profile_training，使用真实训练 batch，打印并保存 model_profile.json。统计会暂时切换 eval、恢复模块模式和随机状态，避免更新 BN 或消耗训练的随机数。辅助头在 eval 下显式执行，以统计其前向开销。

```bash
python profile_model.py --mode head --size 256 --output profile.json
python profile_model.py --mode head --size 256 --no-aux --output profile-no-aux.json
```

默认 multiscale/enhanced/original 主头，输入 [1,3,256,256]：

| 设置 | 全网 Params | 冻结 backbone 后可训练 Params | MACs/图 |
| --- | ---: | ---: | ---: |
| use_aux=True | 1,122,988 | 175,276 | 540,409,856 |
| use_aux=False | 1,029,654 | 81,942 | 516,620,288 |

MACs 包括 Conv2d、Linear 和每个 SeaAttention 的 row/col QKᵀ、AV。这里注意力矩阵 MACs 为 1,032,192；辅助头额外为 23,789,568。1 次乘加计 1 MAC；不计插值、归一化、softmax、池化、逐元素门控/加权 squeeze、loss 和反向传播。这是说明口径的部分算子 MACs，不是全部 FLOPs 或运行时延。

use_aux=True 时训练 loss = 主交叉熵 + aux_loss_weight × 辅助交叉熵；验证 mIoU/accuracy 只使用主 logits。history.csv 中的 loss 为该加权总损失。

## 推理

```bash
python load_checkpoint.py /path/to/full_model.pth --size 256 --device cpu
python load_checkpoint.py /path/to/full_model.pth --image /path/to/image.jpg --output prediction.png
```

接受原 `model_state_dict + model_config` 或新 `model + config` 格式。输出 PNG 为 0..149 的类别索引，不是彩色可视化；独立 backbone 权重不能用于完整模型推理。

## 验证范围

19 项测试通过：12 种 attention/fusion/主头组合与原 main_network 在相同权重下训练及推理输出逐项一致；冻结参数/BN、辅助损失权重、实际矩阵乘法 MACs、统计无副作用、原权重格式加载、两阶段训练和断点续训均已验证。另已运行 main.py 的 pretrain/head、两种 checkpoint 格式的推理和含/不含 aux 的 profile CLI。

这些验证使用合成 ADE20K 格式数据和构造的同结构原格式 checkpoint；没有真实 ADE20K 精度、GPU 或用户真实权重文件的验证结论。

## 项目结构

```text
segmentation_V2/
├── main.py
├── profile_model.py
├── load_checkpoint.py
├── network.py
├── Stem_block.py
├── fusion_block.py
├── seaformer_attention.py
├── seaformer_attention_multiscale.py
├── seaformer_attention_lepe.py
├── requirements.txt
├── requirements-dev.txt
├── README.md
├── segmentation/
│   ├── __init__.py
│   ├── config.py
│   ├── trainer.py
│   ├── engine.py
│   ├── checkpoints.py
│   ├── dataset.py
│   ├── metrics.py
│   ├── freezing.py
│   ├── utils.py
│   └── models/
│       ├── __init__.py
│       └── segmentor.py
└── tests/
    └── test_mobile_v2.py
```
