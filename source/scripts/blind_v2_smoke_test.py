"""Bounded synthetic-only integration checks for blind-v2.

No real dataset or historical checkpoint is read, and every toy training run
lives in a TemporaryDirectory. These losses/metrics are software checks, not
restoration results or evidence for a paper. Run with the project environment:

    python scripts/blind_v2_smoke_test.py --output synthetic_smoke.json
    python scripts/blind_v2_smoke_test.py --skip_gpu
"""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE_TIMEOUT_SECONDS = 180
GPU_RESULT_PREFIX = "SYNTHETIC_GPU_RESULT="
METRIC_KEYS = (
    "train_loss",
    "train_routing_consistency_loss",
    "train_feature_consistency_loss",
    "val_macro_psnr",
    "val_local_task_accuracy",
    "val_local_global_js",
    "val_routing_ece",
    "val_routing_entropy",
)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def text_output(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value or ""


def run_stage(name, command, expected_error=None):
    """Capture each bounded child, with short success logs and full failures."""
    environment = os.environ.copy()
    environment.update({
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    })
    started = time.monotonic()
    print("[{}] synthetic stage started".format(name), flush=True)
    try:
        result = subprocess.run(
            [str(value) for value in command],
            cwd=str(PROJECT_ROOT),
            env=environment,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=STAGE_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        print(text_output(error.stdout), file=sys.stderr, flush=True)
        print(text_output(error.stderr), file=sys.stderr, flush=True)
        raise RuntimeError(
            "{} exceeded {} seconds".format(name, STAGE_TIMEOUT_SECONDS)
        ) from error
    elapsed = time.monotonic() - started
    print("[{}] exit={} elapsed={:.2f}s".format(name, result.returncode, elapsed), flush=True)
    if result.returncode:
        print("[{}] full stdout:\n{}".format(name, result.stdout), file=sys.stderr)
        print("[{}] full stderr:\n{}".format(name, result.stderr), file=sys.stderr, flush=True)
    else:
        for label, stream in (("stdout", result.stdout), ("stderr", result.stderr)):
            lines = stream.strip().splitlines()
            if lines:
                print("[{}] {} tail:\n{}".format(name, label, "\n".join(lines[-5:])), flush=True)
    if expected_error is None:
        require(result.returncode == 0, "{} failed".format(name))
    else:
        require(result.returncode != 0, "{} should reject changed data".format(name))
        require(
            expected_error in result.stdout + result.stderr,
            "{} failed for an unexpected reason".format(name),
        )
    return result, {"returncode": result.returncode, "elapsed_seconds": round(elapsed, 3)}


def gpu_worker():
    import torch

    if not torch.cuda.is_available():
        return {"status": "skipped", "reason": "CUDA is unavailable", "synthetic_only": True}
    sys.path.insert(0, str(PROJECT_ROOT))
    from net.promamba import ProMamba
    from utils.losses import RestorationLoss, blind_v2_routing_loss

    torch.manual_seed(47)
    torch.cuda.manual_seed_all(47)
    torch.set_num_threads(1)
    torch.cuda.reset_peak_memory_stats()
    model = ProMamba(
        architecture="hierarchical", condition_mode="blind_v2", dim=48,
        stage_blocks=(2, 3, 4), bottleneck_type="mamba", d_state=16,
        norm_type="group", gate_mode="competitive", identity_init=True,
        batch_reverse_scans=True, routing_context_size=256,
        local_condition_scale=0.1,
    ).cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    scaler = torch.cuda.amp.GradScaler()
    criterion = RestorationLoss()
    tracked = {
        "degradation_classifier": model.hierarchical.degradation_encoder.classifier,
        "global_condition_projection": model.hierarchical.degradation_condition[-1],
        "local_condition_adapter": model.hierarchical.local_condition_adapter[-1],
        "stem": model.hierarchical.stem,
    }
    records = []
    # True identity initialization has TWO zero-output barriers: task heads,
    # then block outputs. Step 1 opens heads; step 2 opens block outputs; step 3
    # can send restoration gradients into the inner condition/gate adapters.
    # Do not randomize these layers merely to make a gradient test pass.
    for index in range(3):
        local = torch.rand(1, 3, 256, 256, device="cuda")
        context = torch.rand(1, 3, 256, 256, device="cuda")
        target = torch.rand_like(local)
        labels = torch.tensor([index % 3], dtype=torch.long, device="cuda")
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output, auxiliary = model(
                local, task_prompt=None, routing_context=context,
                teacher_forcing=0.0, return_aux=True,
            )
            restoration, _ = criterion(output, target)
            routing = blind_v2_routing_loss(auxiliary, labels, local_cls_weight=0.5)
            loss = (restoration.mean() + 0.1 * routing["combined_cls"]
                    + 0.05 * routing["probability_consistency"]
                    + 0.01 * routing["feature_consistency"])
        require(bool(torch.isfinite(output).all()), "GPU output is not finite")
        require(bool(torch.isfinite(loss)), "GPU loss is not finite")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradients = {}
        for name, module in tracked.items():
            gradient = module.weight.grad
            require(gradient is not None, "Missing gradient: {}".format(name))
            require(bool(torch.isfinite(gradient).all()), "Nonfinite gradient: {}".format(name))
            gradients[name] = float(gradient.float().abs().sum())
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        require(bool(torch.isfinite(norm)), "Nonfinite total GPU gradient norm")
        if index == 1:
            require(gradients["stem"] > 0, "Backbone gradient did not open on step 2")
        if index == 2:
            for name, magnitude in gradients.items():
                require(magnitude > 0, "Zero gradient after three default-init steps: {}".format(name))
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        require(scaler.get_scale() >= scale_before, "AMP skipped a GPU optimizer step")
        records.append({
            "step": index + 1, "loss": float(loss.detach()),
            "gradient_l1": gradients, "gradient_norm_before_clip": float(norm),
        })
    torch.cuda.synchronize()
    return {
        "status": "passed", "synthetic_only": True,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "model_config": model.model_config,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "input_shape": [1, 3, 256, 256], "context_shape": [1, 3, 256, 256],
        "precision": "AMP float16 with GradScaler", "teacher_forcing": 0.0,
        "steps": records,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / (1024 ** 2),
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / (1024 ** 2),
    }


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_fixture(directory):
    import numpy as np
    from PIL import Image

    root = directory / "synthetic_data"
    manifests = {}
    sizes = ((36, 44), (40, 36), (34, 46))
    for split_index, split in enumerate(("train", "val")):
        rows = []
        for task_index, task in enumerate(("derain", "dehaze", "denoise")):
            height, width = sizes[task_index]
            height += 2 * split_index
            width += 2 * split_index
            generator = np.random.RandomState(47 + 100 * split_index + task_index)
            clean = generator.randint(20, 220, size=(height, width, 3)).astype(np.uint8)
            task_root = root / task
            (task_root / "target").mkdir(parents=True, exist_ok=True)
            (task_root / "input").mkdir(exist_ok=True)
            name = "{}_{}.png".format(split, task)
            Image.fromarray(clean, mode="RGB").save(task_root / "target" / name)
            if task == "denoise":
                input_name, sigma = "", "25"
            else:
                degraded = clean.astype(np.float32)
                if task == "dehaze":
                    degraded = 0.65 * degraded + 75.0
                else:
                    degraded[:, ::5] += 25.0
                Image.fromarray(np.clip(degraded, 0, 255).astype(np.uint8), mode="RGB").save(
                    task_root / "input" / name
                )
                input_name, sigma = name, ""
            rows.append({"task": task, "input": input_name, "target": name, "sigma": sigma})
        manifest = directory / (split + ".csv")
        with manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("task", "input", "target", "sigma"))
            writer.writeheader()
            writer.writerows(rows)
        manifests[split] = manifest
    return root, manifests


def load_own_checkpoint(path):
    import torch

    # Only load checkpoints produced by this process's temporary toy runs.
    return torch.load(str(path), map_location="cpu", weights_only=False)


def check_checkpoint(path, root, manifests, expected_epoch):
    checkpoint = load_own_checkpoint(path)
    require(checkpoint["epoch"] == expected_epoch, "Unexpected checkpoint epoch")
    config = checkpoint["model_config"]
    for name, expected in (
        ("condition_mode", "blind_v2"), ("architecture", "hierarchical"),
        ("bottleneck_type", "conv"), ("dim", 8), ("routing_context_size", 32),
    ):
        require(config[name] == expected, "Checkpoint config mismatch: {}".format(name))
    protocol = checkpoint["data_protocol"]
    require(protocol["data_root"] == str(root.resolve()), "Wrong absolute data root")
    require(protocol["val_root"] == str(root.resolve()), "Wrong validation root")
    for split in ("train", "val"):
        record = protocol[split + "_manifest"]
        require(record["path"] == str(manifests[split].resolve()), "Manifest path was not restored")
        require(record["sha256"] == sha256(manifests[split]), "Manifest hash mismatch")
    require(checkpoint["val_protocol"]["inference"] == "tile", "Wrong validation inference")
    require(checkpoint["val_protocol"]["batch_size"] == 1, "Full-image validation must use batch 1")
    require(checkpoint["train_args"]["amp"] is False, "Resume did not restore no-AMP protocol")
    for state in ("python_rng_state", "numpy_rng_state", "torch_rng_state", "data_generator_state"):
        require(state in checkpoint, "Missing checkpoint RNG state: {}".format(state))
    with (path.parent / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    require(bool(rows), "No toy training metrics")
    row = rows[-1]
    require(int(row["epoch"]) == expected_epoch, "Metrics epoch mismatch")
    metrics = {}
    for name in METRIC_KEYS:
        require(row.get(name, "") != "", "Missing metric: {}".format(name))
        metrics[name] = float(row[name])
        require(math.isfinite(metrics[name]), "Nonfinite metric: {}".format(name))
    require(0 <= metrics["val_local_task_accuracy"] <= 1, "Invalid local route accuracy")
    require(float(row["teacher_forcing_ratio"]) == 0.0, "Toy run unexpectedly used Task ID routing")
    return {"epoch": expected_epoch, "metrics": metrics, "data_protocol": protocol}


def integration_check():
    from PIL import Image

    stages = {}
    with tempfile.TemporaryDirectory(prefix="taskaxisir_blind_v2_synthetic_") as temporary:
        directory = Path(temporary).resolve()
        root, manifests = build_fixture(directory)
        runs = directory / "runs"
        command = [
            sys.executable, "-B", str(PROJECT_ROOT / "train.py"),
            "--data_root", root, "--train_manifest", manifests["train"],
            "--val_manifest", manifests["val"], "--save_dir", runs,
            "--run_name", "initial", "--architecture", "hierarchical",
            "--condition_mode", "blind_v2", "--dim", "8",
            "--bottleneck_type", "conv", "--stage_blocks", "1", "1", "1",
            "--patch_size", "32", "--routing_context_size", "32",
            "--steps_per_epoch", "2", "--batch_size", "3", "--epochs", "1",
            "--router_warmup_epochs", "0", "--blind_teacher_forcing_start", "0",
            "--blind_teacher_forcing_end", "0", "--blind_teacher_forcing_epochs", "1",
            "--backbone_lr", "0.0002", "--condition_lr", "0.0002", "--router_lr", "0.0002",
            "--num_workers", "0", "--no-amp", "--val_inference", "tile",
            "--val_tile_size", "32", "--val_overlap", "8", "--save_every", "1",
        ]
        _, stages["initial_training"] = run_stage("synthetic initial train", command)
        initial_path = runs / "initial" / "latest.pth"
        initial = check_checkpoint(initial_path, root, manifests, expected_epoch=1)
        resume_command = [
            sys.executable, "-B", str(PROJECT_ROOT / "train.py"),
            "--resume", initial_path, "--epochs", "2", "--save_dir", runs,
            "--run_name", "resumed", "--num_workers", "0",
        ]
        _, stages["resume"] = run_stage("synthetic resume without data arguments", resume_command)
        resumed_path = runs / "resumed" / "latest.pth"
        resumed = check_checkpoint(resumed_path, root, manifests, expected_epoch=2)
        require(initial["data_protocol"] == resumed["data_protocol"], "Resume changed bound data")

        input_image = root / "dehaze" / "input" / "val_dehaze.png"
        output_image = directory / "blind_prediction.png"
        result, stages["blind_single_image"] = run_stage("synthetic blind PNG inference", [
            sys.executable, "-B", str(PROJECT_ROOT / "test_single.py"),
            "--weights", resumed_path, "--img_path", input_image,
            "--save_path", output_image, "--tile_size", "32", "--overlap", "8", "--no-amp",
        ])
        with Image.open(input_image) as source, Image.open(output_image) as prediction:
            prediction.load()
            require(prediction.format == "PNG", "Single-image output is not PNG")
            require(prediction.size == source.size, "Blind inference changed image dimensions")
            png_size = list(prediction.size)
        require("routing=blind_v2" in result.stdout, "Single-image test did not report blind-v2 routing")

        evaluation_dir = directory / "synthetic_batch_evaluation"
        _, stages["blind_directory"] = run_stage("synthetic automatic batch inference", [
            sys.executable, "-B", str(PROJECT_ROOT / "test.py"),
            "--weights", resumed_path, "--input_dir", root / "derain" / "input",
            "--target_dir", root / "derain" / "target", "--save_dir", evaluation_dir,
            "--tile_size", "32", "--overlap", "8", "--no-amp",
        ])
        summary = json.loads((evaluation_dir / "metrics.json").read_text(encoding="utf-8"))
        require(summary["task"] == "auto", "Batch evaluation used a manual task")
        require(summary["routing"] == "blind_v2", "Batch evaluation did not use blind-v2")
        require(summary["failure_count"] == 0, "A synthetic batch image failed")
        require(summary["success_count"] == 2, "Batch evaluation missed a fixture image")
        with (evaluation_dir / "per_image_metrics.csv").open(newline="", encoding="utf-8") as handle:
            batch_rows = list(csv.DictReader(handle))
        require(len(batch_rows) == 2, "Wrong number of batch routing rows")
        batch_js = []
        for row in batch_rows:
            require(row.get("local_global_js", "") != "", "Missing batch local/global JS")
            value = float(row["local_global_js"])
            require(math.isfinite(value), "Nonfinite batch local/global JS")
            batch_js.append(value)

        # Change only our disposable fixture. Valid CSV gains a blank line, but
        # its provenance hash changes and must be rejected before more training.
        with manifests["val"].open("ab") as handle:
            handle.write(b"\n")
        _, stages["changed_manifest_rejected"] = run_stage("synthetic manifest tamper rejection", [
            sys.executable, "-B", str(PROJECT_ROOT / "train.py"),
            "--resume", resumed_path, "--epochs", "3", "--save_dir", runs,
            "--run_name", "must_not_train", "--num_workers", "0",
        ], expected_error="Data protocol changed")
        require(not (runs / "must_not_train").exists(), "Changed data created a new training run")
        report = {
            "status": "passed", "synthetic_only": True, "stages": stages,
            "initial": initial, "resumed": resumed, "blind_png_size_wh": png_size,
            "batch_test": {"task": summary["task"], "routing": summary["routing"],
                           "success_count": summary["success_count"], "local_global_js": batch_js},
            "resume_omitted_data_arguments": True, "inference_omitted_task_id": True,
            "changed_manifest_rejected": True,
        }
    report["temporary_fixture_removed"] = True
    require(not directory.exists(), "Temporary synthetic fixture was not cleaned up")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional synthetic-only JSON report.")
    parser.add_argument("--skip_gpu", "--skip-gpu", action="store_true")
    parser.add_argument("--gpu_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.gpu_worker:
        print(GPU_RESULT_PREFIX + json.dumps(gpu_worker(), allow_nan=False), flush=True)
        return 0

    print("SYNTHETIC SOFTWARE SMOKE ONLY: not a formal training run or paper result.", flush=True)
    report = {
        "synthetic_only": True, "paper_results": False,
        "real_training_data_used": False, "child_timeout_seconds": STAGE_TIMEOUT_SECONDS,
    }
    exit_code = 0
    try:
        if args.skip_gpu:
            report["gpu"] = {"status": "skipped", "reason": "--skip_gpu"}
        else:
            result, stage = run_stage("synthetic real-Mamba GPU", [
                sys.executable, "-B", str(Path(__file__).resolve()), "--gpu_worker",
            ])
            lines = [line for line in result.stdout.splitlines() if line.startswith(GPU_RESULT_PREFIX)]
            require(len(lines) == 1, "GPU child did not return its report")
            report["gpu"] = json.loads(lines[0][len(GPU_RESULT_PREFIX):])
            report["gpu"]["process"] = stage
        report["integration"] = integration_check()
        report["status"] = "passed"
        report["gpu_validated"] = report["gpu"]["status"] == "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = "{}: {}".format(type(error).__name__, error)
        traceback.print_exc()
        exit_code = 1
    encoded = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded, flush=True)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
