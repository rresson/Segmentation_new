# 分割实验工程

两个独立工程并列存放，运行命令前进入对应目录：

| 工程 | Backbone | 分割头 |
| --- | --- | --- |
| [segmentation_V2](segmentation_V2/README.md) | 上传的 MobileNetV2 stem + 六个 context block + 三次 fusion | 上传的原始分割头，可选辅助头 |
| [segmentation_V2_V3](segmentation_V2_V3/README.md) | 与 V2 相同的 backbone | MobileV3FPNHead 的 C3 单路，head 内做接口适配 |

两个工程均使用名为 `segmentation` 的包，请分别在各自目录运行，不要混用 PYTHONPATH。
