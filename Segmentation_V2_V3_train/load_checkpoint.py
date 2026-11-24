"""根据 mini checkpoint 的配置恢复模型，可对图像输出类别索引 PNG。"""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from segmentation import Config
from segmentation.models import build_model
from segmentation.checkpoints import load_full_model


def config_from_checkpoint(checkpoint):
    if "config" not in checkpoint:
        raise ValueError("Inference requires a complete mini checkpoint with config metadata")
    values = checkpoint["config"]
    if values.get("head_name") != "mobile_c3_mini" or "context_expanded_channels" not in values:
        raise ValueError("Expected a mini checkpoint; old V2/V3 head weights cannot be loaded")
    from segmentation.config import prepare_config
    return prepare_config(Config(**values))


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
