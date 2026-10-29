# ADE20K：分割头与指定 CNN 层联合训练配置说明

适用工程：当前 `ade20k_modular`，默认网络为 `SmallCNNBackbone + FPNHead`。
修改位置：`segmentation/config.py` 中的 `Config` 类。运行入口始终为项目根目录 `main.py`。
本说明对应现有代码能力，不需要改动 CNN 或分割头结构。

## 1. 你当前需要的配置

如果希望训练整个分割头，同时训练 CNN 的最后两个 stage，修改 Config 中的对应字段：

```python
# 必须使用 custom；head 模式会覆盖你写入的 trainable_prefixes。
mode: Literal["pretrain", "head", "custom"] = "custom"

data_root: str = "ADEChallengeData2016"
backbone_name: Literal["small_cnn", "demo"] = "small_cnn"
head_name: Literal["fpn", "simple"] = "fpn"
head_width: int = 128

# 此处要求你已完成 CNN 初始训练，且文件确实存在。
backbone_checkpoint: str = "runs/ade20k/pretrain/backbone_best_miou.pth"
checkpoint_prefix: str = ""
allow_random_backbone: bool = False
resume: str = ""  # 新实验，不恢复旧 optimizer 或旧分割头。
output_dir: str = "runs/ade20k/head_stage2_stage3"

# 列出的范围会参与优化；未列出的参数被冻结。
trainable_prefixes: tuple[str, ...] = (
    "decode_head",
    "backbone.stage2",
    "backbone.stage3",
)
```

其效果如下：

| 模块 | 可训练参数 | 训练期间的模块状态 |
| --- | --- | --- |
| backbone.stem | 冻结 | eval |
| backbone.stage1 | 冻结 | eval |
| backbone.stage2 | 更新 | train |
| backbone.stage3 | 更新 | train |
| decode_head | 更新 | train |

这里“更新”表示这些参数有资格参与梯度更新，并不保证每个参数在每个 batch 上都具有非零梯度。
冻结区域的 BatchNorm 统计量保持不变；选中的 stage 内 BatchNorm 统计量会随训练更新。

**初始化含义：**这个配置加载已有 CNN，但 FPNHead 从随机初始化开始训练。若你希望保留已经训练好的 FPNHead，再改变 CNN 的解冻范围，请先阅读第 8 节；现有 config 尚不能直接完成这种阶段切换。

## 2. 配置填写规则

以下代码块是 Config 类中对应字段的替换内容，粘贴时保持类内缩进。替换原字段，不要在类末尾重复声明同名字段。
`Literal` 已在 config.py 中导入，无需再次添加。

| mode | 代码的实际行为 | 能否自行指定训练范围 |
| --- | --- | --- |
| pretrain | 强制 SimpleHead，强制训练 backbone 和 decode_head | 否 |
| head | 强制只训练 decode_head，必须加载 CNN 权重 | 否 |
| custom | 保留 head_name 与 trainable_prefixes 的设置 | 是 |

只修改 trainable_prefixes、却保留 mode="head"，CNN 指定层不会被解冻，因为 prepare_config() 会将其改回 `( "decode_head", )`。
实际生效的配置可查看输出目录中的 `config.json`。

路径相对于项目根目录，即 main.py 所在目录。当前数据目录应为：

| 用途 | 相对路径 |
| --- | --- |
| 训练图片 | ADEChallengeData2016/images/training |
| 训练标签 | ADEChallengeData2016/annotations/training |
| 验证图片 | ADEChallengeData2016/images/validation |
| 验证标签 | ADEChallengeData2016/annotations/validation |

data_root 应填写 `ADEChallengeData2016`，不再加 images。

## 3. 网络中哪些名称可以选择

### 3.1 默认 SmallCNNBackbone 的模块

| 前缀 | 对应范围 |
| --- | --- |
| backbone | 整个 CNN：stem、stage1、stage2、stage3 |
| backbone.stem | 最前面的卷积块 |
| backbone.stage1 | 第一级特征提取块 |
| backbone.stage2 | 第二级特征提取块 |
| backbone.stage3 | 第三级特征提取块 |
| decode_head | 整个分割头 |

这里的 stage 是一个卷积块，包含多个子层，并非单个卷积。

每个 SmallCNNBackbone 卷积块的内部编号如下，stem 与 stage1/2/3 都使用此结构：

| 相对编号 | 子层 | 是否有可训练参数 |
| --- | --- | --- |
| 0 | 第一个 Conv2d | 有，weight；无 bias |
| 1 | 第一个 BatchNorm2d | 有，weight 与 bias；另有运行统计量 buffers |
| 2 | 第一个 ReLU | 无 |
| 3 | 第二个 Conv2d | 有，weight；无 bias |
| 4 | 第二个 BatchNorm2d | 有，weight 与 bias；另有运行统计量 buffers |
| 5 | 第二个 ReLU | 无 |

例如：`backbone.stage3.0` 选择最后一个卷积块中的第一个卷积；`backbone.stage3.3` 选择其中第二个卷积。

以上内部编号针对 small_cnn。原版 DemoBackbone 没有 stem，且每个 stage 只有编号 0 的卷积、1 的 BN、2 的 ReLU；切换为 demo 时不能照搬 `.3` 等名称。

### 3.2 名称匹配规则

代码按完整名称或以点号分隔的前缀匹配，不支持通配符、正则表达式或排除表达式。

| 写法 | 是否支持 |
| --- | --- |
| `backbone.stage3` | 支持，选中该模块所有参数 |
| `backbone.stage3.0` | 支持，选中这个卷积层的参数 |
| `backbone.stage3.0.weight` | 支持，精确选择一个参数 |
| `stage3` | 不支持，缺少 backbone 前缀 |
| `backbone.stage*` | 不支持，没有通配符展开 |
| `!backbone.stage1` | 不支持，没有排除语法 |
| `backbone.stage3.2` | 不支持单独选择，ReLU 没有参数 |

多个前缀取并集。例如同时填写 backbone 和 backbone.stage3，仍然会训练整个 CNN，并不会只训练 stage3。
单元素元组末尾要有逗号：`("decode_head",)`，不要写成字符串括号 `("decode_head")`。

## 4. 常用训练范围组合

以下组合统一使用第 1 节的 custom 配置作为基础，只替换 trainable_prefixes，并为不同实验设置不同的 output_dir。
它们使用同一份预训练 CNN，默认会新建并初始化 FPNHead。

| 目标 | trainable_prefixes 的值 | output_dir 示例 |
| --- | --- | --- |
| 只训练分割头 | `("decode_head",)` | runs/ade20k/head_only_custom |
| 分割头 + CNN 最后一级 | `("decode_head", "backbone.stage3")` | runs/ade20k/head_stage3 |
| 分割头 + CNN 最后两级 | `("decode_head", "backbone.stage2", "backbone.stage3")` | runs/ade20k/head_stage2_stage3 |
| 分割头 + 三个 stage，冻结 stem | `("decode_head", "backbone.stage1", "backbone.stage2", "backbone.stage3")` | runs/ade20k/head_all_stages |
| 分割头 + 不连续的 stage1、stage3 | `("decode_head", "backbone.stage1", "backbone.stage3")` | runs/ade20k/head_stage1_stage3 |
| 分割头 + 整个 CNN | `("decode_head", "backbone")` | runs/ade20k/head_full_cnn |
| 分割头 + 最后块的第二个卷积 | `("decode_head", "backbone.stage3.3")` | runs/ade20k/head_stage3_conv2 |
| 分割头 + 最后块的两个卷积，冻结其 BN | `("decode_head", "backbone.stage3.0", "backbone.stage3.3")` | runs/ade20k/head_stage3_convs |
| 分割头 + 某个具体卷积权重 | `("decode_head", "backbone.stage3.0.weight")` | runs/ade20k/head_one_weight |

最后两类细粒度选择中，不被选中的 BN 模块保持 eval，参数与运行统计量不更新。
如果你选中的是 `backbone.stage3.1.weight`，则该 BN 模块会进入 train；它的 bias 可以保持冻结，但运行均值/方差仍会更新。这是当前实现的行为，不能将“只训练 BN 的 weight”理解为“BN 的其余状态全部不变”。

对于不连续层，例如 stage1 + stage3，冻结中间 stage2 的权重不会阻断梯度传回 stage1：只要 CNN 存在可训练参数，Segmentor 就不会对整个 CNN 使用 no_grad。

## 5. 其他常用模式组合

### 5.1 尚无 CNN 权重：从零训练 CNN + SimpleHead

```python
mode: Literal["pretrain", "head", "custom"] = "pretrain"
backbone_name: Literal["small_cnn", "demo"] = "small_cnn"
backbone_checkpoint: str = ""
checkpoint_prefix: str = ""
resume: str = ""
output_dir: str = "runs/ade20k/pretrain"
```

pretrain 自动选择 simple 头、允许随机 CNN，并训练整个 backbone 和 decode_head。
完成后可使用 `runs/ade20k/pretrain/backbone_best_miou.pth` 进行第 1 节的实验。

### 5.2 使用默认 head 模式，只训练 FPNHead

```python
mode: Literal["pretrain", "head", "custom"] = "head"
backbone_name: Literal["small_cnn", "demo"] = "small_cnn"
head_name: Literal["fpn", "simple"] = "fpn"
head_width: int = 128
backbone_checkpoint: str = "runs/ade20k/pretrain/backbone_best_miou.pth"
checkpoint_prefix: str = ""
resume: str = ""
output_dir: str = "runs/ade20k/head_only"
```

也可以将 backbone_checkpoint 留空，此时自动读取 output_root 下的 `pretrain/backbone_best_miou.pth`。显式填写路径更方便管理多个实验。
若第一阶段使用了不同 CNN 或不同输出目录，需要同步修改这里的架构选择或权重路径。

### 5.3 从零联合训练整个 CNN + FPNHead

```python
mode: Literal["pretrain", "head", "custom"] = "custom"
backbone_name: Literal["small_cnn", "demo"] = "small_cnn"
head_name: Literal["fpn", "simple"] = "fpn"
head_width: int = 128
backbone_checkpoint: str = ""
checkpoint_prefix: str = ""
allow_random_backbone: bool = True
trainable_prefixes: tuple[str, ...] = ("backbone", "decode_head")
resume: str = ""
output_dir: str = "runs/ade20k/full_fpn_from_scratch"
```

此组合直接训练完整 FPN 分割网络，与 pretrain 的 CNN + SimpleHead 不同。不要误用 pretrain 来训练 FPNHead，因为 pretrain 会自动替换为 SimpleHead。
程序会打印随机 CNN 提示；在全部参数都参与训练的此场景中，随机初始化是有意的。不要把这与冻结随机 CNN 的接口调试混淆。

### 5.4 固定 CNN，训练 SimpleHead 作为对照

```python
mode: Literal["pretrain", "head", "custom"] = "custom"
backbone_name: Literal["small_cnn", "demo"] = "small_cnn"
head_name: Literal["fpn", "simple"] = "simple"
backbone_checkpoint: str = "runs/ade20k/pretrain/backbone_best_miou.pth"
checkpoint_prefix: str = ""
allow_random_backbone: bool = False
trainable_prefixes: tuple[str, ...] = ("decode_head",)
resume: str = ""
output_dir: str = "runs/ade20k/simple_head_control"
```

这个 SimpleHead 也会重新初始化。与 FPNHead 做对照时，固定相同的 CNN 权重、数据划分和评估方法，并明确训练预算；不要直接将第一阶段的联合训练结果当成冻结 CNN 下的对照结果。

## 6. CNN 权重文件的组合

| 你拥有的权重 | backbone_checkpoint | checkpoint_prefix |
| --- | --- | --- |
| 本工程单独导出的 CNN | 对应 backbone_best_miou.pth / backbone_last.pth | 空字符串 |
| 同架构 CNN 的纯 state_dict | 对应文件路径 | 空字符串 |
| 完整模型 checkpoint，CNN 参数名以 backbone. 开头 | 对应 best_miou.pth / last.pth | `backbone.` |
| 原上传脚本的 DemoBackbone 权重 | 对应文件路径，并设置 backbone_name="demo" | 取决于文件保存的是独立 CNN 还是完整模型 |

从完整模型里只取 CNN 的例子：

```python
backbone_checkpoint: str = "runs/ade20k/head_only/best_miou.pth"
checkpoint_prefix: str = "backbone."
resume: str = ""
```

**该写法只读取 CNN，不会加载文件中的已训练分割头。** 即使文件来自 head 训练阶段，新建的分割头也会重新初始化。
DataParallel 的 module. 前缀由加载器自动移除，不要因此随意更改架构名称。
SmallCNNBackbone 与 DemoBackbone 的权重不能互换，输入 mean/std 也应与 CNN 训练时一致。

## 7. 原实验断点续训

如果实验原本就在训练“分割头 + stage3”，只是中断后继续，保持原配置和训练范围，仅填写原目录及 last.pth：

```python
mode: Literal["pretrain", "head", "custom"] = "custom"
trainable_prefixes: tuple[str, ...] = ("decode_head", "backbone.stage3")
output_dir: str = "runs/ade20k/head_stage3"
resume: str = "runs/ade20k/head_stage3/last.pth"
```

这个片段必须应用在与该断点匹配的原配置上，不是跨任意实验通用的恢复配置。
resume 会恢复整个模型、optimizer、scheduler、AMP scaler、已完成轮次和最佳指标，并跳过单独的 CNN 初始化。

当前续训会核对的配置包括：

- 实际可训练参数名称列表。
- CNN 类型、分割头类型和宽度。
- epochs、学习率、最小学习率、weight_decay。
- batch_size、AMP、seed、num_workers。
- 类别数、忽略值、mean/std、训练缩放裁剪及验证尺寸。
- 训练/验证样本上限；新工程断点还会核对 data_root。

output_dir 必须是 resume 文件所在目录。epochs 表示原计划总轮数，例如原计划 20 轮、中断于第 8 轮，续训仍填写 20，不是 12。
已完成所有计划轮次时，再用同样配置 resume 不会额外训练；不能简单把 epochs 加大来绕过配置一致性检查。

## 8. 哪些需求不能只改当前 config

| 需求 | 当前能力 |
| --- | --- |
| 新建实验，加载 CNN，训练新分割头 + 指定 CNN 层 | 支持，使用 custom |
| 在不改变训练范围的情况下恢复完整训练断点 | 支持，使用 resume |
| 保留已训练的分割头，再从“仅头训练”切换到“头 + CNN 部分层” | 不支持仅改配置；resume 会拒绝训练参数范围变化 |
| CNN 用较小学习率，分割头用较大学习率 | 尚未支持配置分组；当前优化器所有选中参数共用一个 learning_rate |
| 同一次运行中，到第 N 轮自动解冻更多层 | 没有该配置，需修改训练管理逻辑 |
| 选择整个 stage，但强制该 stage 的 BN 仍为 eval | 没有独立开关；可选择卷积子层来保持未选中的 BN 冻结 |
| 只训练分割头的 classifier，并复用一个已训练头的其他层 | 范围选择可以写 decode_head.classifier，但现有新实验初始化只加载 CNN，无法仅靠配置复用完整已训练头 |

如果你的真实目标是“现有训练好的 FPNHead 继续训练，同时新解冻几个 CNN 层”，需要增加“完整模型初始化、重建优化器”的入口。它不同于 resume，也不同于 backbone_checkpoint；当前不能用这两个字段假装实现同样效果。

不要直接删除续训检查来改变训练范围，旧 optimizer 的参数组可能与新的可训练参数列表不一致。

## 9. 数据规模、训练长度和学习率组合

以下是独立于训练范围的配置选项，按实验目的与显存情况调整。数值是示例，不是已经验证的最优超参数。

| 使用场景 | max_train_samples | max_val_samples | epochs | batch_size |
| --- | ---: | ---: | ---: | ---: |
| 当前默认流程检查 | 1000 | 200 | 5 | 4 |
| 极小数据快速排查 | 20 | 10 | 2 | 2 |
| 完整数据实验示例 | 0 | 0 | 50 | 4 |

使用完整数据时，对应设置为：

```python
max_train_samples: int = 0
max_val_samples: int = 0
```

训练时 crop_size=256 表示随机裁剪尺寸；验证使用 val_short_side/val_max_side 对输入等比例缩放，标签保留原始尺寸。
想减少显存占用，可以在新实验开始前减小 batch_size；正在恢复旧实验时，不能静默改变受检查的配置。

学习率示例：

```python
# 这是用于试验的数值，并不保证适合所有解冻范围。
learning_rate: float = 1e-4
min_lr: float = 1e-6
```

这会同时作用于所有选中的 CNN 参数和分割头参数，必须满足 0 <= min_lr <= learning_rate。
从冻结 CNN 改为微调 CNN 后，应结合 loss 和验证 mIoU 判断是否需要调整学习率；改变学习率的对照实验要使用新的输出目录。

保存与设备配置：

```python
save_every: int = 5  # 周期 checkpoint；last.pth 仍每轮保存，最佳指标仍单独保存。
device: str = "auto"
amp: bool = True     # 仅 CUDA 实际启用，CPU 自动关闭。
num_workers: int = 0 # Windows/PyCharm 初次运行保持 0。
```

## 10. 如何确认选层配置生效

1. 查看控制台的“可训练参数”范围。例如 head + stage3 应包含 decode_head 和 backbone.stage3。
2. 打开本实验输出目录的 trainable_parameters.txt，确认没有不希望训练的模块参数。
3. 打开 config.json，确认实际 mode、head_name、权重路径和 trainable_prefixes。
4. 使用 samples.json 检查不同实验是否使用相同训练/验证子集。
5. 观察 loss 与验证 mIoU/像素 accuracy；选择了参数不等于已经取得良好分割效果。

全部 CNN 冻结时，每轮会打印 `CNN冻结检查=通过`。
只要选中了 CNN 中的任何参数，就不会打印这一整网不变性检查结果；这是预期行为，不能因为该提示消失就认定其他层没有冻结。
当前自动精确比较覆盖“整个 CNN 冻结”的场景；部分微调时，如需逐项证明未选中参数和 BN buffers 完全不变，需要另行比较训练前后 state_dict。

参数名单示例中可以出现 backbone.stage3.0.weight、backbone.stage3.1.weight、backbone.stage3.1.bias 等。
BatchNorm 的 running_mean、running_var 等是 buffers，不会出现在可训练参数名单中；它们由模块 train/eval 状态控制。

## 11. 开始实验前的最短步骤

1. 没有 CNN 权重时，先运行 pretrain。
2. 选择第 1 节的 custom 配置，或从第 4 节选取你需要的层组合。
3. 给新实验填写独立 output_dir，并清空 resume。
4. 确认 backbone_checkpoint 的网络架构和输入归一化匹配。
5. 在 PyCharm 中运行根目录 main.py，检查 trainable_parameters.txt。
6. 同一个实验中断后，保持配置不变，填写其 last.pth 继续。

本文件是配置说明，未自动替换你正在使用的 config，也未启动真实 ADE20K 训练。
