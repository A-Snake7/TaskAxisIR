"""Evaluate ProMamba with every true-task / task-prompt combination.

This is the T01 task-prompt sensitivity experiment.  It preserves the
PromptIR-compatible preprocessing and metrics implemented in
``evaluate_aio3_promptir.py`` while deliberately overriding the task prompt.

The formal output is a 3 x 3 matrix whose rows are the true degradation
(derain, dehaze, denoise) and whose columns are the supplied prompt
(derain, dehaze, denoise).  A denoise cell is the macro-average over CBSD68
sigma 15/25/50; its per-sigma values and per-image CSV files are retained in
the JSON output.  The NumPy noise stream is reset for every prompt so all
three prompt columns see exactly the same noisy CBSD68 inputs.
"""

import argparse
import csv
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import skimage
import torch
from PIL import Image
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for import_root in (PROJECT_ROOT, SCRIPT_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import evaluate_aio3_promptir as base  # noqa: E402
from utils.checkpoint import load_model_for_inference  # noqa: E402
from utils.modes import BLIND_MODES  # noqa: E402


EXPERIMENT_ID = "T01"
PROTOCOL_ID = "AIO3-PROMPTIR-COMPAT-V1+TASK-PROMPT-3X3-V1"
TASKS = ("derain", "dehaze", "denoise")
CSV_FIELDS = (
    "experiment_id",
    "protocol_id",
    "checkpoint_sha256",
    "true_task",
    "true_task_id",
    "prompt_task",
    "prompt_task_id",
    "is_correct_prompt",
    "aggregation",
    "condition_count",
    "image_evaluations",
    "average_psnr",
    "average_ssim",
    "denoise_sigma15_psnr",
    "denoise_sigma15_ssim",
    "denoise_sigma25_psnr",
    "denoise_sigma25_ssim",
    "denoise_sigma50_psnr",
    "denoise_sigma50_ssim",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the ProMamba true-task x supplied-prompt 3x3 matrix"
    )
    parser.add_argument("--weights", default="runs/promamba_aio3_stage1/epoch_0100.pth")
    parser.add_argument("--data-root", default="data/test")
    parser.add_argument("--save-dir", default="evaluations/task_prompt_3x3")
    parser.add_argument("--sigmas", nargs="+", type=int, default=(15, 25, 50))
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--crop-base", type=int, default=16)
    parser.add_argument("--inference", choices=("full", "tile"), default="full")
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--precision", choices=("fp32", "amp"), default="fp32")
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def write_per_image_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("image", "target", "psnr", "ssim"))
        writer.writeheader()
        writer.writerows(rows)


def summarize_condition(save_dir, true_task, prompt_task, condition, rows):
    csv_path = (
        Path(save_dir)
        / "per_image"
        / ("true_{}".format(true_task))
        / ("prompt_{}".format(prompt_task))
        / condition
        / "per_image_metrics.csv"
    )
    write_per_image_csv(csv_path, rows)
    return {
        "count": len(rows),
        "average_psnr": float(np.mean([row["psnr"] for row in rows])),
        "average_ssim": float(np.mean([row["ssim"] for row in rows])),
        "per_image_csv": str(csv_path),
    }


def evaluate_paired(model, data_root, true_task, prompt_task, device, args):
    input_dir = Path(data_root) / true_task / "input"
    target_dir = Path(data_root) / true_task / "target"
    inputs = base.image_files(input_dir)
    target_index = base.build_target_index(target_dir)
    prompt = torch.tensor(
        [base.TASK_IDS[prompt_task]], dtype=torch.long, device=device
    )
    rows = []
    images_dir = (
        Path(args.save_dir)
        / "images"
        / ("true_{}".format(true_task))
        / ("prompt_{}".format(prompt_task))
    )
    description = "true={} prompt={}".format(true_task, prompt_task)
    for input_path in tqdm(inputs, desc=description):
        target_path = base.find_target(input_path, target_index, true_task)
        degraded_u8 = base.load_rgb_uint8(input_path, args.crop_base)
        clean_u8 = base.load_rgb_uint8(target_path, args.crop_base)
        if degraded_u8.shape != clean_u8.shape:
            raise ValueError(
                "Strict geometry mismatch for {}: {} != {}".format(
                    input_path.name, degraded_u8.shape, clean_u8.shape
                )
            )
        restored = base.infer(
            model, base.uint8_to_tensor(degraded_u8, device), prompt, args
        )
        restored_np = base.tensor_to_rgb_float(restored)
        clean_np = clean_u8.astype(np.float32) / 255.0
        psnr, ssim = base.compute_metrics(restored_np, clean_np)
        rows.append(
            {
                "image": input_path.name,
                "target": target_path.name,
                "psnr": psnr,
                "ssim": ssim,
            }
        )
        if args.save_images:
            base.save_png(restored_np, images_dir / (input_path.stem + ".png"))
    condition = summarize_condition(
        args.save_dir, true_task, prompt_task, true_task, rows
    )
    return {true_task: condition}, [path.name for path in inputs]


def evaluate_denoise(model, data_root, prompt_task, device, args):
    clean_dir = Path(data_root) / "denoise" / "target"
    clean_paths = base.image_files(clean_dir)

    # Resetting here is intentional: each prompt column must receive identical
    # noisy inputs, while sigma 15 -> 25 -> 50 remains one continuous stream.
    rng = np.random.RandomState(args.noise_seed)
    prompt = torch.tensor(
        [base.TASK_IDS[prompt_task]], dtype=torch.long, device=device
    )
    conditions = {}
    for sigma in args.sigmas:
        condition_name = "denoise_sigma{}".format(sigma)
        rows = []
        images_dir = (
            Path(args.save_dir)
            / "images"
            / "true_denoise"
            / ("prompt_{}".format(prompt_task))
            / condition_name
        )
        description = "true=denoise prompt={} sigma={}".format(prompt_task, sigma)
        for clean_path in tqdm(clean_paths, desc=description):
            clean_u8 = base.load_rgb_uint8(clean_path, args.crop_base)
            noise = rng.randn(*clean_u8.shape)
            noisy_u8 = np.clip(clean_u8 + noise * sigma, 0, 255).astype(np.uint8)
            restored = base.infer(
                model, base.uint8_to_tensor(noisy_u8, device), prompt, args
            )
            restored_np = base.tensor_to_rgb_float(restored)
            clean_np = clean_u8.astype(np.float32) / 255.0
            psnr, ssim = base.compute_metrics(restored_np, clean_np)
            rows.append(
                {
                    "image": clean_path.name,
                    "target": clean_path.name,
                    "psnr": psnr,
                    "ssim": ssim,
                }
            )
            if args.save_images:
                base.save_png(
                    restored_np,
                    images_dir / clean_path.with_suffix(".png").name,
                )
        conditions[condition_name] = summarize_condition(
            args.save_dir, "denoise", prompt_task, condition_name, rows
        )
    return conditions, [path.name for path in clean_paths]


def build_matrix_row(
    true_task,
    prompt_task,
    conditions,
    checkpoint_hash,
    sigmas=(15, 25, 50),
):
    if true_task == "denoise":
        condition_names = ["denoise_sigma{}".format(sigma) for sigma in sigmas]
        missing = [name for name in condition_names if name not in conditions]
        if missing:
            raise ValueError("Missing denoise conditions: {}".format(missing))
        average_psnr = float(
            np.mean([conditions[name]["average_psnr"] for name in condition_names])
        )
        average_ssim = float(
            np.mean([conditions[name]["average_ssim"] for name in condition_names])
        )
        aggregation = "macro_mean_sigma_{}".format("_".join(map(str, sigmas)))
    else:
        condition_names = [true_task]
        average_psnr = conditions[true_task]["average_psnr"]
        average_ssim = conditions[true_task]["average_ssim"]
        aggregation = "dataset_image_mean"

    row = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "checkpoint_sha256": checkpoint_hash,
        "true_task": true_task,
        "true_task_id": base.TASK_IDS[true_task],
        "prompt_task": prompt_task,
        "prompt_task_id": base.TASK_IDS[prompt_task],
        "is_correct_prompt": true_task == prompt_task,
        "aggregation": aggregation,
        "condition_count": len(condition_names),
        "image_evaluations": int(
            sum(conditions[name]["count"] for name in condition_names)
        ),
        "average_psnr": float(average_psnr),
        "average_ssim": float(average_ssim),
    }
    for sigma in (15, 25, 50):
        condition = conditions.get("denoise_sigma{}".format(sigma))
        row["denoise_sigma{}_psnr".format(sigma)] = (
            float(condition["average_psnr"]) if condition else None
        )
        row["denoise_sigma{}_ssim".format(sigma)] = (
            float(condition["average_ssim"]) if condition else None
        )
    return row


def write_matrix_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with open(temporary_path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    temporary_path.replace(path)


def self_test():
    sample = np.zeros((19, 35, 3), dtype=np.uint8)
    if base.crop_to_base(sample, 16).shape != (16, 32, 3):
        raise AssertionError("PromptIR-compatible crop self-test failed")

    noise_a = np.random.RandomState(0).randn(4, 5, 3)
    noise_b = np.random.RandomState(0).randn(4, 5, 3)
    if not np.array_equal(noise_a, noise_b):
        raise AssertionError("Per-prompt denoise stream reset is not reproducible")

    paired = {"derain": {"count": 2, "average_psnr": 31.0, "average_ssim": 0.9}}
    denoise = {
        "denoise_sigma15": {"count": 2, "average_psnr": 33.0, "average_ssim": 0.93},
        "denoise_sigma25": {"count": 2, "average_psnr": 31.0, "average_ssim": 0.89},
        "denoise_sigma50": {"count": 2, "average_psnr": 28.0, "average_ssim": 0.80},
    }
    paired_row = build_matrix_row("derain", "dehaze", paired, "test-hash")
    denoise_row = build_matrix_row("denoise", "denoise", denoise, "test-hash")
    if paired_row["is_correct_prompt"] or not denoise_row["is_correct_prompt"]:
        raise AssertionError("Correct/wrong prompt labels are incorrect")
    if denoise_row["average_psnr"] != (33.0 + 31.0 + 28.0) / 3.0:
        raise AssertionError("Denoise sigma macro-average is incorrect")

    rows = []
    for true_task in TASKS:
        for prompt_task in TASKS:
            source = denoise if true_task == "denoise" else {
                true_task: {
                    "count": 2,
                    "average_psnr": 30.0,
                    "average_ssim": 0.9,
                }
            }
            rows.append(
                build_matrix_row(true_task, prompt_task, source, "test-hash")
            )
    if len(rows) != 9 or sum(row["is_correct_prompt"] for row in rows) != 3:
        raise AssertionError("3x3 matrix shape/diagonal self-test failed")

    with tempfile.TemporaryDirectory() as temporary_directory:
        csv_path = Path(temporary_directory) / "matrix.csv"
        json_path = Path(temporary_directory) / "matrix.json"
        write_matrix_csv(csv_path, rows)
        write_json(json_path, {"matrix_rows": rows})
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as handle:
            if len(list(csv.DictReader(handle))) != 9:
                raise AssertionError("CSV round-trip did not preserve nine rows")
        with open(json_path, "r", encoding="utf-8") as handle:
            if len(json.load(handle)["matrix_rows"]) != 9:
                raise AssertionError("JSON round-trip did not preserve nine rows")

    print(
        json.dumps(
            {
                "self_test": "ok",
                "experiment_id": EXPERIMENT_ID,
                "protocol_id": PROTOCOL_ID,
                "matrix_shape": [3, 3],
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

    if tuple(args.sigmas) != (15, 25, 50):
        raise ValueError(
            "T01 formal protocol requires --sigmas 15 25 50 in this exact order"
        )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint, model_config = load_model_for_inference(args.weights, device)
    if model_config.get("condition_mode", "oracle") in BLIND_MODES:
        raise ValueError(
            "The task-prompt 3x3 matrix is an oracle-prompt experiment and "
            "cannot be run with a blind checkpoint."
        )
    model.eval()
    checkpoint_hash = base.checkpoint_sha256(args.weights)

    matrix_rows = []
    matrix = {task: {} for task in TASKS}
    file_order = {}
    for true_task in TASKS:
        for prompt_task in TASKS:
            if true_task == "denoise":
                conditions, order = evaluate_denoise(
                    model, args.data_root, prompt_task, device, args
                )
            else:
                conditions, order = evaluate_paired(
                    model, args.data_root, true_task, prompt_task, device, args
                )
            row = build_matrix_row(
                true_task,
                prompt_task,
                conditions,
                checkpoint_hash,
                sigmas=tuple(args.sigmas),
            )
            matrix_rows.append(row)
            matrix[true_task][prompt_task] = {
                "summary": row,
                "conditions": conditions,
            }
            file_order[true_task] = order

    summary = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "base_protocol_id": base.PROTOCOL_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "correct-versus-wrong explicit task-prompt sensitivity",
        "matrix_shape": [3, 3],
        "matrix_axes": {
            "rows": "true_task",
            "columns": "prompt_task supplied to model",
        },
        "interpretation_note": (
            "The supplied prompt controls both the task embedding and the "
            "task-specific output head in ProMamba-H."
        ),
        "denoise_cell_aggregation": "macro mean over sigma 15, 25, and 50",
        "noise_stream_reset_per_prompt": True,
        "weights": str(Path(args.weights).resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "model_config": model_config,
        "code_commit": base.git_commit(),
        "device": str(device),
        "precision": args.precision,
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
        "file_order": file_order,
        "matrix_rows": matrix_rows,
        "matrix": matrix,
    }

    csv_path = save_dir / "task_prompt_3x3.csv"
    json_path = save_dir / "task_prompt_3x3.json"
    write_matrix_csv(csv_path, matrix_rows)
    write_json(json_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("FINAL_MATRIX_CSV {}".format(csv_path))
    print("FINAL_MATRIX_JSON {}".format(json_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
