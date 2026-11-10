import argparse
import csv
import glob
import json
import os
import sys
import zlib

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from utils.checkpoint import load_model_for_inference
from utils.common import AverageMeter, calc_psnr, calc_ssim
from utils.inference import center_crop_pair_to_common, tiled_inference
from utils.modes import BLIND_MODES


TASK_IDS = {"derain": 0, "dehaze": 1, "denoise": 2}
VALID_EXTENSIONS = (".png", ".jpg", ".jpeg")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ProMamba on full images")
    parser.add_argument(
        "--weights",
        default="./checkpoints/promamba_legacy_epoch_0130.pth",
    )
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--target_dir", default="")
    parser.add_argument("--save_dir", default="./results")
    parser.add_argument(
        "--task",
        choices=("auto",) + tuple(sorted(TASK_IDS)),
        default="auto",
        help=(
            "Dataset/metric task. Blind checkpoints do not receive this label; "
            "oracle checkpoints require a non-auto value."
        ),
    )
    parser.add_argument(
        "--oracle_routing",
        action="store_true",
        help="Force a blind checkpoint to use the supplied task as an oracle ablation.",
    )
    parser.add_argument("--sigma", type=int, default=15)
    parser.add_argument("--tile_size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument(
        "--local_blind_routing",
        action="store_true",
        help="Predict routing independently for every tile instead of once per image.",
    )
    parser.add_argument("--seed", type=int, default=42)
    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument("--amp", dest="amp", action="store_true")
    amp_group.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=None)
    return parser.parse_args()


def build_target_index(target_dir):
    if not target_dir or not os.path.isdir(target_dir):
        return {}
    return {
        os.path.splitext(filename)[0]: os.path.join(target_dir, filename)
        for filename in sorted(os.listdir(target_dir))
        if filename.lower().endswith(VALID_EXTENSIONS)
    }


def find_target(input_filename, target_index):
    stem = os.path.splitext(input_filename)[0]
    candidates = [stem]
    sidd_stem = stem.replace("_NOISY_", "_GT_")
    if sidd_stem != stem:
        candidates.append(sidd_stem)
    if "_" in stem:
        candidates.append(stem.split("_", 1)[0])
    if stem.startswith("rain-"):
        candidates.append(stem.replace("rain-", "norain-", 1))
    elif stem.startswith("rain"):
        candidates.append(stem.replace("rain", "norain", 1))
    return next(
        (target_index[candidate] for candidate in candidates if candidate in target_index),
        None,
    )


def load_rgb_tensor(path, device):
    with Image.open(path) as image:
        tensor = TF.to_tensor(image.convert("RGB")).unsqueeze(0)
    return tensor.to(device)


def save_rgb_tensor(tensor, path):
    image = TF.to_pil_image(tensor.squeeze(0).cpu().clamp(0, 1))
    image.save(path, format="PNG")


def main():
    args = parse_args()
    if not os.path.isdir(args.input_dir):
        raise FileNotFoundError("Input directory not found: {}".format(args.input_dir))
    os.makedirs(args.save_dir, exist_ok=True)
    output_dir = os.path.join(args.save_dir, "images")
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.amp is None:
        amp_enabled = bool(device.type == "cuda" and args.tile_size >= 512)
        precision_source = "auto"
    else:
        amp_enabled = bool(args.amp and device.type == "cuda")
        precision_source = "explicit"
    model, checkpoint, model_config = load_model_for_inference(args.weights, device)
    print("Loaded {} on {}".format(args.weights, device))
    print("Model config: {}".format(model_config))
    is_blind = model_config.get("condition_mode", "oracle") in BLIND_MODES
    if not is_blind and args.task == "auto":
        raise ValueError("This oracle checkpoint requires --task")
    if args.oracle_routing and args.task == "auto":
        raise ValueError("--oracle_routing requires a non-auto --task")
    use_oracle_routing = not is_blind or args.oracle_routing

    input_paths = sorted(
        path
        for path in glob.glob(os.path.join(args.input_dir, "*"))
        if path.lower().endswith(VALID_EXTENSIONS)
    )
    if not input_paths:
        raise RuntimeError("No supported images found in {}".format(args.input_dir))

    target_index = build_target_index(args.target_dir)
    task_prompt = (
        torch.tensor(
            [TASK_IDS[args.task]],
            dtype=torch.long,
            device=device,
        )
        if use_oracle_routing
        else None
    )
    teacher_forcing = 1.0 if is_blind and use_oracle_routing else 0.0
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    rows = []
    missing_targets = 0
    failures = []
    routing_accuracy = AverageMeter()
    routing_confusion = [[0 for _ in TASK_IDS] for _ in TASK_IDS]

    for input_path in tqdm(input_paths, desc=args.task):
        filename = os.path.basename(input_path)
        stem = os.path.splitext(filename)[0]
        try:
            degraded = load_rgb_tensor(input_path, device)
            target = None
            if args.task == "denoise" and not args.target_dir:
                target = degraded.clone()
                generator = torch.Generator(device=device)
                image_seed = args.seed + zlib.crc32(filename.encode("utf-8"))
                generator.manual_seed(image_seed)
                noise = torch.randn(
                    target.shape,
                    dtype=target.dtype,
                    device=device,
                    generator=generator,
                )
                degraded = torch.clamp(
                    target + noise * (args.sigma / 255.0),
                    0,
                    1,
                )
            elif target_index:
                target_path = find_target(filename, target_index)
                if target_path is not None:
                    target = load_rgb_tensor(target_path, device)
                else:
                    missing_targets += 1

            inference_output = tiled_inference(
                model,
                degraded,
                task_prompt,
                tile_size=args.tile_size,
                overlap=args.overlap,
                amp_enabled=amp_enabled,
                teacher_forcing=teacher_forcing,
                global_blind_routing=not args.local_blind_routing,
                return_routing=is_blind,
            )
            if is_blind:
                restored, routing = inference_output
            else:
                restored = inference_output
                routing = None
            restored = restored.clamp(0, 1)
            save_rgb_tensor(
                restored,
                os.path.join(output_dir, stem + ".png"),
            )

            row = {
                "image": filename,
                "psnr": "",
                "ssim": "",
                "predicted_task": "",
                "routing_confidence": "",
                "p_derain": "",
                "p_dehaze": "",
                "p_denoise": "",
                "routing_entropy": "",
                "local_global_js": "",
            }
            if routing is not None and routing["task_probs"] is not None:
                probabilities = routing["task_probs"][0].detach().float().cpu()
                predicted_task_id = int(probabilities.argmax().item())
                predicted_task = next(
                    name for name, task_id in TASK_IDS.items()
                    if task_id == predicted_task_id
                )
                row.update(
                    {
                        "predicted_task": predicted_task,
                        "routing_confidence": float(probabilities.max().item()),
                        "p_derain": float(probabilities[TASK_IDS["derain"]].item()),
                        "p_dehaze": float(probabilities[TASK_IDS["dehaze"]].item()),
                        "p_denoise": float(probabilities[TASK_IDS["denoise"]].item()),
                        "routing_entropy": float(
                            -(probabilities * probabilities.clamp_min(1e-8).log()).sum().item()
                        ),
                        "local_global_js": (
                            float(routing["local_global_js"])
                            if routing.get("local_global_js") is not None else ""
                        ),
                    }
                )
                if args.task != "auto":
                    true_task_id = TASK_IDS[args.task]
                    routing_confusion[true_task_id][predicted_task_id] += 1
                    routing_accuracy.update(
                        float(predicted_task_id == true_task_id)
                    )
            if target is not None:
                metric_output, metric_target = center_crop_pair_to_common(
                    restored.float(),
                    target.float(),
                )
                psnr = calc_psnr(metric_output, metric_target)
                ssim = calc_ssim(metric_output, metric_target)
                psnr_meter.update(psnr)
                ssim_meter.update(ssim)
                row.update({"psnr": psnr, "ssim": ssim})
            rows.append(row)
        except Exception as error:
            failures.append({"image": filename, "error": str(error)})
            tqdm.write("FAILED {}: {}".format(filename, error))

    metrics_csv = os.path.join(args.save_dir, "per_image_metrics.csv")
    with open(metrics_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "image",
                "psnr",
                "ssim",
                "predicted_task",
                "routing_confidence",
                "p_derain",
                "p_dehaze",
                "p_denoise",
                "routing_entropy",
                "local_global_js",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "task": args.task,
        "routing": "oracle" if use_oracle_routing else model_config["condition_mode"],
        "weights": os.path.abspath(args.weights),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "model_config": model_config,
        "input_count": len(input_paths),
        "success_count": len(rows),
        "metric_count": psnr_meter.count,
        "missing_target_count": missing_targets,
        "failure_count": len(failures),
        "average_psnr": psnr_meter.avg if psnr_meter.count else None,
        "average_ssim": ssim_meter.avg if ssim_meter.count else None,
        "tile_size": args.tile_size,
        "overlap": args.overlap,
        "global_blind_routing": not args.local_blind_routing,
        "amp": amp_enabled,
        "precision_source": precision_source,
        "routing_accuracy": routing_accuracy.avg
        if routing_accuracy.count
        else None,
        "routing_confusion": routing_confusion if routing_accuracy.count else None,
        "routing_confusion_labels": (
            ["derain", "dehaze", "denoise"] if routing_accuracy.count else None
        ),
        "failures": failures,
    }
    with open(
        os.path.join(args.save_dir, "metrics.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if psnr_meter.count:
        print(
            "FINAL_METRICS task={} PSNR={:.4f} SSIM={:.6f} count={}".format(
                args.task,
                psnr_meter.avg,
                ssim_meter.avg,
                psnr_meter.count,
            )
        )
    if failures:
        print("{} images failed; see metrics.json".format(len(failures)))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
