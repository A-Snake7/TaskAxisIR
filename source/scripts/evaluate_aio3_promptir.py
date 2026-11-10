"""PromptIR-compatible AIO-3 evaluation for ProMamba.

The primary protocol mirrors PromptIR's public test code:
  * center-crop each image to a multiple of 16;
  * full-image inference in FP32 by default;
  * skimage RGB PSNR/SSIM with data_range=1;
  * one NumPy RandomState(0) stream for CBSD68, evaluated in the order
    sigma=15, 25, 50;
  * Gaussian noise is added in uint8 [0, 255], clipped, converted back to
    uint8, then normalized to [0, 1].

Unlike the reference code, filenames are sorted and recorded so every model
uses the exact same per-image noise realization.  The output JSON records the
protocol, versions, checkpoint hash, file order, and per-condition metrics.
"""

import argparse
import csv
import hashlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import skimage
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.checkpoint import load_model_for_inference  # noqa: E402
from utils.inference import tiled_inference  # noqa: E402
from utils.modes import BLIND_MODES  # noqa: E402


PROTOCOL_ID = "AIO3-PROMPTIR-COMPAT-V1"
TASK_IDS = {"derain": 0, "dehaze": 1, "denoise": 2}
VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate ProMamba under a PromptIR-compatible AIO-3 protocol"
    )
    parser.add_argument("--weights", default="runs/promamba_aio3_stage1/epoch_0100.pth")
    parser.add_argument("--data-root", default="data/test")
    parser.add_argument("--save-dir", default="evaluations/promptir_compat")
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=("all", "derain", "dehaze", "denoise"),
        default=("all",),
    )
    parser.add_argument("--sigmas", nargs="+", type=int, default=(15, 25, 50))
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--crop-base", type=int, default=16)
    parser.add_argument("--inference", choices=("full", "tile"), default="full")
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--precision", choices=("fp32", "amp"), default="fp32")
    parser.add_argument(
        "--metric-domain",
        choices=("float", "png8"),
        default="float",
        help=(
            "Prediction domain used for PSNR/SSIM: 'float' evaluates the clipped "
            "model tensor directly (legacy/default behavior); 'png8' performs the "
            "same uint8 truncation as --save-images, then a lossless PNG encode/decode "
            "round-trip before evaluation"
        ),
    )
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def crop_to_base(array, base=16):
    """Match PromptIR ``crop_img`` exactly: symmetric center crop."""
    height, width = array.shape[:2]
    crop_h = height % base
    crop_w = width % base
    return array[
        crop_h // 2 : height - crop_h + crop_h // 2,
        crop_w // 2 : width - crop_w + crop_w // 2,
        ...,
    ]


def image_files(directory):
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError("Image directory not found: {}".format(root))
    files = sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    )
    if not files:
        raise RuntimeError("No supported images found in {}".format(root))
    return files


def load_rgb_uint8(path, crop_base):
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return np.ascontiguousarray(crop_to_base(array, crop_base))


def uint8_to_tensor(array, device):
    tensor = torch.from_numpy(array.transpose(2, 0, 1).copy()).float().div_(255.0)
    return tensor.unsqueeze(0).to(device)


def tensor_to_rgb_float(tensor):
    return (
        tensor.detach()
        .float()
        .clamp(0, 1)
        .squeeze(0)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )


def png8_roundtrip(array):
    """Apply the exact saved-PNG quantization and lossless decode in memory."""
    quantized = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(quantized, mode="RGB").save(buffer, format="PNG")
    buffer.seek(0)
    with Image.open(buffer) as image:
        decoded = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return decoded.astype(np.float32) / 255.0


def prediction_for_metrics(restored, metric_domain):
    """Select the reproducible prediction representation used by the metrics."""
    if metric_domain == "float":
        return restored
    if metric_domain == "png8":
        return png8_roundtrip(restored)
    raise ValueError("Unsupported metric domain: {}".format(metric_domain))


def compute_metrics(restored, clean):
    if restored.shape != clean.shape:
        raise ValueError("Metric shape mismatch: {} != {}".format(restored.shape, clean.shape))
    restored = np.clip(restored, 0.0, 1.0)
    clean = np.clip(clean, 0.0, 1.0)
    psnr = peak_signal_noise_ratio(clean, restored, data_range=1)
    ssim = structural_similarity(clean, restored, data_range=1, channel_axis=-1)
    return float(psnr), float(ssim)


def build_target_index(target_dir):
    result = {}
    for path in image_files(target_dir):
        result[path.stem] = path
    return result


def find_target(input_path, target_index, task):
    stem = input_path.stem
    candidates = [stem]
    if task == "dehaze" and "_" in stem:
        candidates.append(stem.split("_", 1)[0])
    if task == "derain":
        if stem.startswith("rain-"):
            candidates.extend((stem[len("rain-") :], "norain-" + stem[len("rain-") :]))
        elif stem.startswith("rain"):
            candidates.extend((stem[len("rain") :], "norain" + stem[len("rain") :]))
    for candidate in candidates:
        if candidate in target_index:
            return target_index[candidate]
    raise KeyError("No target for {} (candidates={})".format(input_path.name, candidates))


def checkpoint_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def infer(model, degraded, prompt, args):
    amp_enabled = args.precision == "amp" and degraded.device.type == "cuda"
    effective_prompt = (
        None
        if getattr(model, "condition_mode", "oracle") in BLIND_MODES
        else prompt
    )
    if args.inference == "tile":
        return tiled_inference(
            model,
            degraded,
            effective_prompt,
            tile_size=args.tile_size,
            overlap=args.overlap,
            amp_enabled=amp_enabled,
        ).clamp(0, 1)
    with torch.inference_mode():
        with torch.autocast(
            device_type=degraded.device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            return model(degraded, effective_prompt).clamp(0, 1)


def save_png(array, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    quantized = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(quantized, mode="RGB").save(path)


def write_condition(save_dir, condition, rows):
    condition_dir = Path(save_dir) / condition
    condition_dir.mkdir(parents=True, exist_ok=True)
    with open(condition_dir / "per_image_metrics.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("image", "target", "psnr", "ssim"))
        writer.writeheader()
        writer.writerows(rows)
    return {
        "count": len(rows),
        "average_psnr": float(np.mean([row["psnr"] for row in rows])),
        "average_ssim": float(np.mean([row["ssim"] for row in rows])),
        "per_image_csv": str(condition_dir / "per_image_metrics.csv"),
    }


def evaluate_paired(model, data_root, task, device, args):
    input_dir = Path(data_root) / task / "input"
    target_dir = Path(data_root) / task / "target"
    inputs = image_files(input_dir)
    target_index = build_target_index(target_dir)
    prompt = torch.tensor([TASK_IDS[task]], dtype=torch.long, device=device)
    rows = []
    images_dir = Path(args.save_dir) / task / "images"
    for input_path in tqdm(inputs, desc=task):
        target_path = find_target(input_path, target_index, task)
        degraded_u8 = load_rgb_uint8(input_path, args.crop_base)
        clean_u8 = load_rgb_uint8(target_path, args.crop_base)
        if degraded_u8.shape != clean_u8.shape:
            raise ValueError(
                "Strict geometry mismatch for {}: {} != {}".format(
                    input_path.name, degraded_u8.shape, clean_u8.shape
                )
            )
        restored = infer(model, uint8_to_tensor(degraded_u8, device), prompt, args)
        restored_np = tensor_to_rgb_float(restored)
        clean_np = clean_u8.astype(np.float32) / 255.0
        metric_prediction = prediction_for_metrics(restored_np, args.metric_domain)
        psnr, ssim = compute_metrics(metric_prediction, clean_np)
        rows.append(
            {
                "image": input_path.name,
                "target": target_path.name,
                "psnr": psnr,
                "ssim": ssim,
            }
        )
        if args.save_images:
            save_png(restored_np, images_dir / (input_path.stem + ".png"))
    return rows, [path.name for path in inputs]


def evaluate_denoise(model, data_root, device, args):
    clean_dir = Path(data_root) / "denoise" / "target"
    clean_paths = image_files(clean_dir)
    rng = np.random.RandomState(args.noise_seed)
    prompt = torch.tensor([TASK_IDS["denoise"]], dtype=torch.long, device=device)
    condition_rows = {}
    for sigma in args.sigmas:
        condition = "denoise_sigma{}".format(sigma)
        rows = []
        images_dir = Path(args.save_dir) / condition / "images"
        for clean_path in tqdm(clean_paths, desc=condition):
            clean_u8 = load_rgb_uint8(clean_path, args.crop_base)
            noise = rng.randn(*clean_u8.shape)
            noisy_u8 = np.clip(clean_u8 + noise * sigma, 0, 255).astype(np.uint8)
            restored = infer(model, uint8_to_tensor(noisy_u8, device), prompt, args)
            restored_np = tensor_to_rgb_float(restored)
            clean_np = clean_u8.astype(np.float32) / 255.0
            metric_prediction = prediction_for_metrics(restored_np, args.metric_domain)
            psnr, ssim = compute_metrics(metric_prediction, clean_np)
            rows.append(
                {
                    "image": clean_path.name,
                    "target": clean_path.name,
                    "psnr": psnr,
                    "ssim": ssim,
                }
            )
            if args.save_images:
                save_png(restored_np, images_dir / clean_path.with_suffix(".png").name)
        condition_rows[condition] = rows
    return condition_rows, [path.name for path in clean_paths]


def self_test():
    sample = np.zeros((19, 35, 3), dtype=np.uint8)
    cropped = crop_to_base(sample, 16)
    if cropped.shape != (16, 32, 3):
        raise AssertionError("crop_to_base mismatch: {}".format(cropped.shape))

    global_state = np.random.get_state()
    try:
        np.random.seed(0)
        expected = np.random.randn(4, 5, 3)
    finally:
        np.random.set_state(global_state)
    actual = np.random.RandomState(0).randn(4, 5, 3)
    if not np.array_equal(expected, actual):
        raise AssertionError("RandomState does not match np.random.seed legacy stream")

    clean = np.linspace(0, 1, 16 * 16 * 3, dtype=np.float32).reshape(16, 16, 3)
    restored = np.clip(clean + 0.01, 0, 1)
    psnr, ssim = compute_metrics(restored, clean)
    if not (35.0 < psnr < 45.0 and 0.9 < ssim <= 1.0):
        raise AssertionError("Metric self-test out of range: {}, {}".format(psnr, ssim))

    float_prediction = prediction_for_metrics(restored, "float")
    if not np.array_equal(float_prediction, restored):
        raise AssertionError("Float metric domain changed the legacy prediction values")

    png_sample = np.array([[[0.0, 0.5, 1.0], [0.999, 0.001, 0.501]]], dtype=np.float32)
    png_prediction = prediction_for_metrics(png_sample, "png8")
    expected_png = np.clip(png_sample * 255.0, 0, 255).astype(np.uint8).astype(np.float32) / 255.0
    if not np.array_equal(png_prediction, expected_png):
        raise AssertionError("PNG8 metric-domain round-trip mismatch")
    print(
        json.dumps(
            {
                "self_test": "ok",
                "protocol_id": PROTOCOL_ID,
                "metric_domains": ["float", "png8"],
                "numpy": np.__version__,
                "skimage": skimage.__version__,
                "torch": torch.__version__,
            },
            indent=2,
        )
    )


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return 0

    if "all" in args.tasks:
        tasks = ("derain", "dehaze", "denoise")
    else:
        tasks = tuple(dict.fromkeys(args.tasks))
    if tuple(args.sigmas) != (15, 25, 50) and "denoise" in tasks:
        print("WARNING: non-standard sigma order changes the continuous RNG stream")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint, model_config = load_model_for_inference(args.weights, device)
    model.eval()

    summary = {
        "protocol_id": PROTOCOL_ID,
        "weights": str(Path(args.weights).resolve()),
        "checkpoint_sha256": checkpoint_sha256(args.weights),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "model_config": model_config,
        "routing": (
            model_config["condition_mode"]
            if model_config.get("condition_mode", "oracle") in BLIND_MODES
            else "oracle"
        ),
        "code_commit": git_commit(),
        "device": str(device),
        "precision": args.precision,
        "metric_domain": args.metric_domain,
        "metric_domain_details": {
            "float": "clipped model prediction in [0,1] before uint8 serialization",
            "png8": (
                "clip(x*255,0,255).astype(uint8), lossless PNG encode/decode, "
                "then divide by 255"
            ),
            "active": args.metric_domain,
        },
        "inference": args.inference,
        "tile_size": args.tile_size if args.inference == "tile" else None,
        "overlap": args.overlap if args.inference == "tile" else None,
        "crop_base": args.crop_base,
        "noise_seed": args.noise_seed,
        "sigma_order": list(args.sigmas),
        "versions": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "skimage": skimage.__version__,
        },
        "conditions": {},
        "file_order": {},
    }

    if "derain" in tasks:
        rows, order = evaluate_paired(model, args.data_root, "derain", device, args)
        summary["conditions"]["derain"] = write_condition(args.save_dir, "derain", rows)
        summary["file_order"]["derain"] = order
    if "dehaze" in tasks:
        rows, order = evaluate_paired(model, args.data_root, "dehaze", device, args)
        summary["conditions"]["dehaze"] = write_condition(args.save_dir, "dehaze", rows)
        summary["file_order"]["dehaze"] = order
    if "denoise" in tasks:
        condition_rows, order = evaluate_denoise(model, args.data_root, device, args)
        for condition, rows in condition_rows.items():
            summary["conditions"][condition] = write_condition(args.save_dir, condition, rows)
        summary["file_order"]["denoise"] = order

    avg5_names = (
        "dehaze",
        "derain",
        "denoise_sigma15",
        "denoise_sigma25",
        "denoise_sigma50",
    )
    if all(name in summary["conditions"] for name in avg5_names):
        summary["avg5_psnr"] = float(
            np.mean([summary["conditions"][name]["average_psnr"] for name in avg5_names])
        )
        summary["avg5_ssim"] = float(
            np.mean([summary["conditions"][name]["average_ssim"] for name in avg5_names])
        )
    else:
        summary["avg5_psnr"] = None
        summary["avg5_ssim"] = None

    summary_path = save_dir / "metrics.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("FINAL_SUMMARY {}".format(summary_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
