import argparse
import os

import torch
import torchvision.transforms.functional as TF
from PIL import Image

from utils.checkpoint import load_model_for_inference
from utils.inference import tiled_inference
from utils.modes import BLIND_MODES


def parse_args():
    parser = argparse.ArgumentParser(description="Restore one image with ProMamba")
    parser.add_argument(
        "--weights",
        default="./checkpoints/promamba_legacy_epoch_0130.pth",
    )
    parser.add_argument("--img_path", required=True)
    parser.add_argument("--save_path", default="./result.png")
    parser.add_argument(
        "--task",
        type=int,
        default=None,
        choices=(0, 1, 2),
        help="Only required by oracle checkpoints or --oracle_routing.",
    )
    parser.add_argument(
        "--oracle_routing",
        action="store_true",
        help="Force a blind checkpoint to use the supplied task as an ablation.",
    )
    parser.add_argument("--tile_size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument(
        "--local_blind_routing",
        action="store_true",
        help="Predict routing independently for every tile instead of once per image.",
    )
    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument("--amp", dest="amp", action="store_true")
    amp_group.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.isfile(args.img_path):
        raise FileNotFoundError(args.img_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.amp is None:
        amp_enabled = bool(device.type == "cuda" and args.tile_size >= 512)
    else:
        amp_enabled = bool(args.amp and device.type == "cuda")
    model, checkpoint, model_config = load_model_for_inference(args.weights, device)
    is_blind = model_config.get("condition_mode", "oracle") in BLIND_MODES
    if not is_blind and args.task is None:
        raise ValueError("This oracle checkpoint requires --task")
    if args.oracle_routing and args.task is None:
        raise ValueError("--oracle_routing requires --task")
    use_oracle_routing = not is_blind or args.oracle_routing

    with Image.open(args.img_path) as image:
        input_tensor = TF.to_tensor(image.convert("RGB")).unsqueeze(0).to(device)
    task_prompt = (
        torch.tensor([args.task], dtype=torch.long, device=device)
        if use_oracle_routing
        else None
    )
    inference_output = tiled_inference(
        model,
        input_tensor,
        task_prompt,
        tile_size=args.tile_size,
        overlap=args.overlap,
        amp_enabled=amp_enabled,
        teacher_forcing=1.0 if is_blind and use_oracle_routing else 0.0,
        global_blind_routing=not args.local_blind_routing,
        return_routing=is_blind,
    )
    if is_blind:
        restored, routing = inference_output
    else:
        restored = inference_output
        routing = None
    restored = restored.clamp(0, 1)

    output_directory = os.path.dirname(os.path.abspath(args.save_path))
    os.makedirs(output_directory, exist_ok=True)
    output_image = TF.to_pil_image(restored.squeeze(0).cpu())
    output_image.save(args.save_path)
    if routing is not None and routing["task_probs"] is not None:
        probabilities = routing["task_probs"][0].detach().float().cpu().tolist()
        print(
            "Predicted task probabilities: derain={:.4f} dehaze={:.4f} "
            "denoise={:.4f}".format(*probabilities)
        )
    print(
        "Saved {} | device={} | routing={} | checkpoint_epoch={} | config={}".format(
            args.save_path,
            device,
            "oracle" if use_oracle_routing else model_config["condition_mode"],
            checkpoint.get("epoch"),
            model_config,
        )
    )


if __name__ == "__main__":
    main()
