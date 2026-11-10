"""Extract balanced Blind-v3 routing features for paper visualization.

This diagnostic performs router-only inference.  It never reads a clean target
as routing input.  Denoising inputs follow the same continuous RandomState(0)
stream and sigma order as the formal AIO3 evaluation.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.checkpoint import load_model_for_inference  # noqa: E402


VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
TASK_LABELS = {"derain": 0, "dehaze": 1, "denoise": 2}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data-root", default="data/test")
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-task", type=int, default=100)
    parser.add_argument("--crop-base", type=int, default=16)
    parser.add_argument("--noise-seed", type=int, default=0)
    return parser.parse_args()


def image_files(directory):
    return sorted(
        path
        for path in Path(directory).iterdir()
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    )


def crop_to_base(array, base):
    height, width = array.shape[:2]
    crop_h, crop_w = height % base, width % base
    return np.ascontiguousarray(
        array[
            crop_h // 2 : height - crop_h + crop_h // 2,
            crop_w // 2 : width - crop_w + crop_w // 2,
            ...,
        ]
    )


def load_rgb_uint8(path, base):
    with Image.open(path) as image:
        return crop_to_base(np.asarray(image.convert("RGB"), dtype=np.uint8), base)


def to_tensor(array, device):
    tensor = torch.from_numpy(array.transpose(2, 0, 1).copy()).float().div_(255.0)
    return tensor.unsqueeze(0).to(device)


def evenly_spaced_indices(length, count):
    count = min(int(count), int(length))
    if count <= 0:
        return set()
    return set(np.linspace(0, length - 1, count, dtype=np.int64).tolist())


def paired_samples(data_root, task, count, crop_base):
    paths = image_files(Path(data_root) / task / "input")
    selected = evenly_spaced_indices(len(paths), count)
    for index, path in enumerate(paths):
        if index in selected:
            yield task, path.name, load_rgb_uint8(path, crop_base)


def denoise_samples(data_root, count, crop_base, noise_seed):
    paths = image_files(Path(data_root) / "denoise" / "target")
    sigmas = (15, 25, 50)
    selected = evenly_spaced_indices(len(paths) * len(sigmas), count)
    rng = np.random.RandomState(noise_seed)
    flat_index = 0
    for sigma in sigmas:
        for path in paths:
            clean = load_rgb_uint8(path, crop_base)
            noise = rng.randn(*clean.shape)
            noisy = np.clip(clean + noise * sigma, 0, 255).astype(np.uint8)
            if flat_index in selected:
                yield "denoise", f"sigma{sigma}/{path.name}", noisy
            flat_index += 1


def scalar(tensor):
    return float(tensor.detach().float().cpu().reshape(-1)[0])


def vector(tensor):
    return tensor.detach().float().cpu().reshape(-1).tolist()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint, config = load_model_for_inference(args.weights, device)
    if getattr(model, "condition_mode", None) != "blind_v3":
        raise RuntimeError("Embedding extraction requires a blind_v3 checkpoint")

    samples = []
    samples.extend(
        paired_samples(args.data_root, "derain", args.per_task, args.crop_base)
    )
    samples.extend(
        paired_samples(args.data_root, "dehaze", args.per_task, args.crop_base)
    )
    samples.extend(
        denoise_samples(
            args.data_root, args.per_task, args.crop_base, args.noise_seed
        )
    )

    rows = []
    with torch.inference_mode():
        for index, (task, name, array) in enumerate(samples, start=1):
            context = model.predict_task_context(
                to_tensor(array, device), max_size=config.get("routing_context_size", 256)
            )
            row = {
                "task": task,
                "task_id": TASK_LABELS[task],
                "image": name,
                "predicted_task": int(context["task_probs"].argmax(dim=1).item()),
                "local_route_weight": scalar(context["local_route_weight"]),
                "local_global_js": scalar(context["local_global_js"]),
            }
            for prefix, value in (
                ("fused_prob", context["task_probs"]),
                ("global_prob", context["global_task_probs"]),
                ("local_prob", context["local_task_probs"]),
                ("global_feature", context["degradation_feature"]),
                ("local_feature", context["local_degradation_feature"]),
            ):
                for feature_index, number in enumerate(vector(value)):
                    row[f"{prefix}_{feature_index}"] = number
            rows.append(row)
            if index % 25 == 0 or index == len(samples):
                print(f"processed {index}/{len(samples)}", flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {len(rows)} rows to {output}")
    print(f"checkpoint_epoch={checkpoint.get('epoch')}")


if __name__ == "__main__":
    main()
