# MobileNetV2-stem 分割网络与三选项训练 pipeline

此文件夹是独立项目；仓库根目录的原项目不参与构建。网络保留上传实现的 MobileNetV2 stem、六个 context block、三次 fusion、原始 segment_block/segment_block2 和可选 auxiliary_head。

## 启动

```bash
cd segmentation_V2
python -m pip install -r requirements-dev.txt
python main.py
```

在 `segmentation/config.py` 设置 `data_root`、训练轮数和数据/模型参数，也可用命令行覆盖。默认训练数据限制为 1000 张、验证 200 张；完整数据集训练可将 `max_train_samples`、`max_val_samples` 设为 0。相对路径统一从 `segmentation_V2` 工程目录解析，PyCharm 可直接运行 `main.py`。

`main.py` 默认启动 pipeline。旧的 `python pipeline.py` 入口仍可使用，两者共享菜单和参数解析。指定 `--mode` 或 `--resume` 时，`main.py` 才进入单阶段训练；不要与 pipeline 的 `--choice` 等选项混用。

程序提供三个选项：

| 选项 | 训练流程 | 初始化和冻结 | 保存位置 |
| --- | --- | --- | --- |
| 1 | pretrain 联合训练，然后运行 head 阶段 | 第一步训练 backbone + head；第二步加载本次最佳 backbone，新建 head 并冻结 backbone | 同一 `runN/pretrain`、`runN/head1` |
| 2 | 单独训练 head | 用户手动选择 backbone 或完整模型 checkpoint；提取 backbone，冻结它，训练新 head | 所选 backbone 所属 run 的下一个 `headN` |
| 3 | 单独训练 backbone，使用 custom 模式 | 用户手动选择权重；加载完整模型，冻结已训练 head，仅更新 backbone | 新的 `runN/custom`，记录原权重路径 |

选项 2/3 必须手动输入 checkpoint 路径，或通过 `--checkpoint` 指定；pipeline 不自动猜测最近一次预训练实验。它会按所选流程设置模式、初始化路径和可训练范围；数据、网络及优化超参数沿用 Config。pipeline 用于新训练；断点续训使用后面的 `main.py --resume`。

### 选项 1：同一 run 内完成两个阶段

```bash
python main.py --choice 1 --data-root /path/to/ADEChallengeData2016
```

默认两个阶段各使用 `Config.epochs`。可分别设置：

```bash
python main.py --choice 1 --pretrain-epochs 50 --head-epochs 20
```

第一阶段结束后，只读取它实际输出目录中的 `backbone_best_miou.pth`；第二阶段保存到该实验的 `head1`。第一阶段失败或没有生成该权重时，head 阶段不会启动。

### 选项 2：再次调用同一 backbone

```bash
python main.py --choice 2 --checkpoint runs/ade20k/run1/pretrain/backbone_best_miou.pth
```

如 `run1/head1` 已存在，这次创建 `run1/head2`，下次创建 `run1/head3`。编号来自磁盘，关闭程序再启动、切换工作目录后都能保持同一归属。历史的 `head`、`head_2` 命名也会计入编号；创建目录时原子占用，避免并发覆盖。

标准布局中的 checkpoint 归入它所在的 `runN`，即使该 run 位于另一个输出根目录。普通外部 checkpoint 首次使用时在 `output_root` 下分配 run，并用内容指纹和绝对路径记录归属。该绑定保存在输出根目录的 `.pipeline_sources.sqlite3`，实验目录保存 `backbone_source.json`；后续重启继续复用。外部文件内容改变会建立新的归属。

### 选项 3：固定已训练 head，只训练 backbone

```bash
python main.py --choice 3 --checkpoint runs/ade20k/run1/pretrain/backbone_best_miou.pth
```

分割任务仍需通过 head 计算损失。此模式会读取同目录的配套 `best_miou.pth`，验证其中的 backbone 与所选导出文件完全匹配，再冻结 head 参数、BatchNorm buffers 和 Dropout，只更新 backbone。

`backbone_last.pth` 对应 `last.pth`。也可以直接选择完整 `best_miou.pth` 或 `last.pth`。只有独立 backbone 且没有配套完整模型时，会要求提供含已训练 head 的完整 checkpoint，避免固定随机 head 训练。

更新后的 backbone 属于新的实验，例如 `run2/custom`。以后选它的 `backbone_best_miou.pth` 训练 head，将保存到 `run2/head1`。

## 输出结构

```text
runs/ade20k/
├── run1/
│   ├── pretrain/               # 选项 1 的联合训练阶段
│   ├── head1/                  # 选项 1 的冻结 backbone 阶段
│   ├── head2/                  # 选项 2 再次调用 run1 的 backbone
│   └── head3/
└── run2/
    ├── custom/                 # 选项 3 训练得到新的 backbone
    └── head1/                  # 选项 2 选择 run2/custom 的 backbone
```

每个阶段目录保存：

- `config.json`：实际模式、run、输出路径、初始化权重和训练范围。
- `model_profile.json`：训练前 Params/MACs 和实际输入尺寸。
- `pipeline.json`：用户选项、选择的文件、实际加载的文件及输出目录。
- `history.csv`、`samples.json`、`trainable_parameters.txt`。
- `best_miou.pth`、`best_accuracy.pth`、`last.pth`、定期 `epoch_XXX.pth`：完整模型及训练状态，包含 head 权重。
- `backbone_best_miou.pth`、`backbone_last.pth`：独立 backbone 权重。

`headN` 内的完整模型 checkpoint 包含该次训练的 head。训练 head 时逐 epoch 检查整个 backbone 的参数和 BN buffers 不变；训练 backbone 时对整个 head 作相同检查。

## 原始单阶段入口与续训

`main.py --mode pretrain/head/custom` 显式选择单阶段训练；不带这些选项时显示 pipeline 菜单。custom 的训练范围由 `Config.trainable_prefixes` 决定；pipeline 选项 3 会明确设置为 `("backbone",)`。

```bash
python main.py --mode head --model-checkpoint runs/ade20k/run1/pretrain/best_miou.pth --run-name run1
```

此示例继承完整模型的 head，再冻结 backbone 训练；pipeline 选项 2 则新建 head。

```bash
python main.py --mode head --resume runs/ade20k/run1/head1/last.pth
```

续训时保持原实验的总 epochs、学习率、数据、网络配置及可训练范围。未提供 `output_dir` 时从 resume 文件推导原目录；显式提供时必须与 resume 文件同目录。custom backbone-only 的续训还需在 Config 中保留 `trainable_prefixes=("backbone",)`。`run_name` 留空或与该 checkpoint 所属 run 一致。resume 不创建新编号。

`backbone_checkpoint` 仅初始化 backbone，`model_checkpoint` 初始化完整网络；二者不会恢复优化器。`resume` 恢复本训练器的完整状态。手动 checkpoint 应与 Config 的 attention/fusion/head/aux 等结构选项一致，加载时会严格校验。

## 网络、数据与依赖

stem 使用纯 `torch.nn` 实现原 MobileNetV2 `features[0:8]`，保留末级 depthwise stride=1，输出 64 通道、1/8 分辨率。预训练初始化通过 `torch.hub` 加载相同的官方 ImageNet 权重。项目无需 torchvision，图像读取使用 PIL.Image/NumPy。

默认：attention=multiscale、fusion=enhanced、主头=original、use_aux=True、aux_loss_weight=0.4。backbone 包括 stem、三个 context stage、两次 downscaling 和三次 fusion；decode_head 包括主头及启用的辅助头。启用辅助头时，head-only 会训练两个头，backbone-only 会冻结两个头。

数据目录为 ADEChallengeData2016 的 `images/training`、`images/validation`、`annotations/training`、`annotations/validation`。原标签 0 忽略，1..150 映射为类别。设置 `--no-pretrained-stem` 可离线测试新实验。

```bash
python profile_model.py --mode head --size 256
python load_checkpoint.py runs/ade20k/run1/head1/best_miou.pth --image /path/to/image.jpg --output prediction.png
```

默认配置、输入 `[1,3,256,256]`：

| 设置 | 全网 Params | head-only 可训练 Params | MACs/图 |
| --- | ---: | ---: | ---: |
| use_aux=True | 1,122,988 | 175,276 | 540,409,856 |
| use_aux=False | 1,029,654 | 81,942 | 516,620,288 |

MACs 包含 Conv2d、Linear 和 SeaAttention 的 QKᵀ/AV，1 次乘加记为 1 MAC；启用 aux 时计入辅助头。不统计插值、归一化、softmax、池化、逐元素操作、loss 和反向传播。每个训练阶段都在首个 epoch 前用真实训练 batch 统计并保存。

## 验证

```bash
python -m pytest -q tests
```

测试覆盖原网络等价性、无 torchvision 导入、Params/MACs、权重加载、冻结状态、断点续训、三种 pipeline 选项、跨进程重启后的 head 归属、外部 checkpoint 持久绑定和并发目录编号。端到端训练使用合成 ADE20K 格式数据和 CPU；测试不代表真实 ADE20K 精度、用户数据或 Windows/CUDA 环境已经验证。

## 相关代码

```text
main.py                        默认 pipeline；--mode/--resume 单阶段训练
pipeline.py                    独立 pipeline 兼容入口
segmentation/cli.py            两个入口共享的菜单与命令行解析
segmentation/pipeline.py       三种流程、checkpoint 配对及归属
segmentation/run_paths.py      run/head 编号与续训目录
segmentation/config.py         网络、训练和路径配置
segmentation/trainer.py        统计、训练、验证、检查与保存
segmentation/freezing.py       梯度、BN/Dropout 与冻结状态检查
segmentation/checkpoints.py    权重初始化及续训兼容
tests/test_pipeline.py         pipeline 与跨进程回归验证
```

所有模型结构统一放在 `segmentation/models/`，包内使用相对导入。训练、统计和推理入口仍在工程根目录：

```text
segmentation_V2/
├── main.py
├── pipeline.py
├── profile_model.py
├── load_checkpoint.py
└── segmentation/
    ├── config.py
    ├── trainer.py
    └── models/
        ├── __init__.py
        ├── segmentor.py
        ├── network.py
        ├── Stem_block.py
        ├── fusion_block.py
        ├── seaformer_attention.py
        ├── seaformer_attention_multiscale.py
        └── seaformer_attention_lepe.py
```

自定义脚本请使用 `from segmentation.models.network import main_network` 或
`from segmentation.models import build_model`。原 checkpoint 的 state_dict 键名保持一致，
可继续通过 `load_checkpoint.py` 和训练器加载。
