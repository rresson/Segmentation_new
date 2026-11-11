# MobileNetV2 backbone + MobileV3FPNHead C3 单路

本目录由原仓库根目录工程迁入，与 `segmentation_V2` 并列。根目录 README 提供两套工程入口。请在本目录运行命令，避免两个同名 `segmentation` 包混用。

## 本次核对和修正

原版本不是 V2 的相同 backbone：context 深度为 `(1,2,2)`，只执行一次最终 fusion，且在 backbone 内增加了 64→128 的 stride-2 投影。

当前默认 `backbone_name="seaformer"` 已修正为 V2 原始结构：

- 相同纯 PyTorch MobileNetV2 stem，保留末级 stride 修改。
- 三个 context stage，每个包含两个 block；两次 context downscaling。
- fusion1、fusion2、fusion3 按原 V2 前向顺序全部执行。
- 原 attention/fusion/network/stem 源文件与 V2 对应文件一致，返回 `(f1,f2,f3,c2)`。
- 默认 enhanced fusion；dim=64、key_dim=16、num_heads=4、attn_ratio=2、pos_shape=16、ffn_ratio=2、expansion_ratio=2。与 V2 比较时需使用相同 attention 与 learnable squeeze 配置。

`seaformer_depths` 固定 `(2,2,2)`，避免误切回此前优化结构。其他遗留 demo/small_cnn/mobile_v3 模式仍保留，使用它们不属于“相同 V2 backbone”的实验。

## 分割头

保留 `mobile_v3_head.py` 中原 `MobileV3FPNHead`。实际默认使用其 C3 单路：

```text
V2 backbone 的 f3（64 通道，1/8）
  → head 内 DW stride-2 + PW 投影（128 通道，1/16）
  → 原 MobileV3 C3 context（两个 128→768→128 块）
  → lateral[2] 对应的 1×1 卷积
  → refine[2] 对应的可分离卷积
  → classifier
```

只给 decode_head 一张 f3 特征图，不使用 FPN 的 C1/C2 lateral、top-down、cat 或 fuse，也不使用 V2 的原主头/辅助头。64→128 接口适配属于 **decode_head**，不会改变 backbone 的 state_dict 或冻结边界。

## 运行

```bash
cd segmentation_V2_V3
python -m pip install -r requirements-dev.txt
python main.py --mode pretrain --data-root /path/to/ADEChallengeData2016
python main.py --mode head --backbone-checkpoint runs/ade20k_v2_v3/pretrain/backbone_best_miou.pth
python profile_model.py --size 256
```

该工程保留单阶段 main.py 启动器；三选项 pipeline 位于 `segmentation_V2`。训练前会输出并保存 Params/MACs。head 模式冻结完整 V2 backbone，仅训练接口适配和 C3 head。

默认输出根目录为 `runs/ade20k_v2_v3`。重复新实验请指定新的 `--output-dir`，续训使用 `--resume` 并指定 checkpoint 所在的 `--output-dir`，保持原训练超参数一致。

可直接加载 V2 导出的 backbone：

```bash
python main.py --mode head --backbone-checkpoint ../segmentation_V2/runs/ade20k/run1/pretrain/backbone_best_miou.pth --output-dir runs/ade20k_v2_v3/from_v2
```

两套工程的 backbone 导出权重支持双向严格加载，前提是结构和预处理参数一致。分割头不同，完整模型 checkpoint 不可互换；此前单次 fusion 版本的 checkpoint 也不能当作当前完整模型续训。

## 验证

```bash
python -m pytest -q tests
```

测试包括原 MobileV3FPNHead C3 运算路径等价性、六种 attention/squeeze 组合与 V2 的四张 backbone 输出逐元素一致、双向 backbone 权重加载、仅一路特征进入 head、冻结参数/BN、统计无副作用、训练及断点续训。使用 CPU 和合成 ADE20K 格式数据，不代表真实数据精度或 Windows/CUDA 实机结果。

默认 multiscale/enhanced、输入 `[1,3,256,256]` 的实测统计：

| 部分 | Params | MACs/图 |
| --- | ---: | ---: |
| backbone（与 V2 一致） | 947,712 | 434,700,288 |
| C3 head（包含 64→128 接口适配） | 1,093,078 | 126,926,848 |
| 整个网络 | 2,040,790 | 561,627,136 |

统计输出分别列出 backbone 的四张输出和实际进入 head 的一路 `[1,64,32,32]`。
MACs 包含卷积、线性层和 attention QKᵀ/AV；不含归一化、插值、池化、逐元素运算和反向传播。

历史单次 fusion attention 的实现保存在 `legacy_seaformer_attention.py`，仅供历史单元测试，当前默认 backbone 不使用它。`SEA_FORMER.md` 和 `instruction.md` 为历史资料。
