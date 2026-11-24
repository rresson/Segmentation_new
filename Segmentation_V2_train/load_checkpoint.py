"""加载原上传网络或本适配训练器的完整模型权重；不会自动启动训练。"""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from segmentation import Config
from segmentation.models import build_model
from segmentation.checkpoints import checkpoint_state, load_full_model


def config_from_checkpoint(checkpoint):
    if "config" in checkpoint:
        return Config(**checkpoint["config"])
    if "model_config" not in checkpoint:
        raise ValueError("Inference requires config/model_config metadata and complete model weights")
    values = dict(checkpoint["model_config"])
    mapping = {"hidden_channel": "head_width", "num_class": "num_classes", "segment_type": "head_name"}
    values = {mapping.get(k, k): v for k, v in values.items()}
    values.setdefault("attention_type", "original")
    values.setdefault("fusion_type", "original")
    values.setdefault("head_name", "original")
    values["num_classes"] = checkpoint.get("num_classes", values.get("num_classes", 150))
    # 原上传的 loader 会关闭不存在的辅助头；这里根据实际权重决定。
    state = checkpoint_state(checkpoint)
    values["use_aux"] = any(k.startswith(("auxiliary_head.", "decode_head.auxiliary_head.")) for k in state)
    return Config(**values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--output", type=Path, help="Save predicted indices 0..149 as PNG")
    parser.add_argument("--size", type=int, default=256, help="Synthetic input size without --image")
    args = parser.parse_args()
    if args.size < 32:
        parser.error("size must be at least 32")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    cfg = config_from_checkpoint(checkpoint)
    model = build_model(cfg, initialize_stem=False)
    load_full_model(model, checkpoint, cfg)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    model.to(device).eval()
    if args.image:
        with Image.open(args.image) as source:
            image = source.convert("RGB")
        width, height = image.size
        scale = min(cfg.val_short_side / min(width, height), cfg.val_max_side / max(width, height))
        image = image.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.BILINEAR)
        x = torch.from_numpy(np.array(image, dtype=np.float32) / 255).permute(2, 0, 1)
        x = ((x - torch.tensor(cfg.mean)[:, None, None]) / torch.tensor(cfg.std)[:, None, None]).unsqueeze(0)
        target_size = (height, width)
    else:
        x = torch.zeros(1, 3, args.size, args.size)
        target_size = (args.size, args.size)
    with torch.inference_mode():
        logits = F.interpolate(model(x.to(device)), target_size, mode="bilinear", align_corners=False)
        prediction = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(prediction).save(args.output)
    print("Loaded epoch:", checkpoint.get("epoch", "unknown"))
    print("Output shape:", tuple(logits.shape))
    print("Prediction shape:", prediction.shape)


if __name__ == "__main__":
    main()
