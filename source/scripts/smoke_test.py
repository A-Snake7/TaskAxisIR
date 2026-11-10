import argparse
import glob
import os

import torch
import torchvision.transforms.functional as TF
from PIL import Image

from utils.checkpoint import load_model_for_inference
from utils.inference import tiled_inference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", default=".")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = os.path.join(
        args.project_root,
        "checkpoints",
        "promamba_legacy_epoch_0130.pth",
    )
    model, checkpoint, config = load_model_for_inference(checkpoint_path, device)
    image_path = sorted(
        glob.glob(
            os.path.join(
                args.project_root,
                "data",
                "test",
                "derain",
                "input",
                "*",
            )
        )
    )[0]
    with Image.open(image_path) as image:
        tensor = TF.to_tensor(image.convert("RGB")).unsqueeze(0).to(device)
    prompt = torch.tensor([0], dtype=torch.long, device=device)
    output = tiled_inference(
        model,
        tensor,
        prompt,
        tile_size=256,
        overlap=32,
        amp_enabled=device.type == "cuda",
    )
    assert output.shape == tensor.shape
    assert torch.isfinite(output).all()
    print(
        "SMOKE_TEST_OK shape={} epoch={} config={}".format(
            tuple(output.shape),
            checkpoint.get("epoch"),
            config,
        )
    )


if __name__ == "__main__":
    main()
