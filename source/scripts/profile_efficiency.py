"""Profile ProMamba parameters, MACs, and inference efficiency.

This is intentionally a standalone, checkpoint-driven script.  The model
configuration is read from the checkpoint through ``load_model_for_inference``
so that the benchmark cannot accidentally use a different architecture than
the reported weights.  Run it from the repository root (or pass an absolute
checkpoint path):

    python scripts/profile_efficiency.py --weights runs/<run>/best.pth

The JSON report is written to stdout and, when ``--output`` is supplied, to a
file as well.  MACs are reported as complete only when the selected profiler
supports every operation.  In particular, Mamba selective scan operations are
never silently treated as zero-cost.
"""

import argparse
import contextlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import sys
from datetime import datetime, timezone

import torch


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _package_version(package_name):
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _nvidia_smi():
    """Return a small, JSON-safe GPU/driver snapshot when nvidia-smi exists."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    rows = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 3:
            rows.append(
                {
                    "name": fields[0],
                    "memory_total_mb": fields[1],
                    "driver_version": fields[2],
                }
            )
    return rows or None


def _hardware_software(device):
    properties = torch.cuda.get_device_properties(device)
    device_index = torch.cuda.current_device()
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torchvision": _package_version("torchvision"),
        "mamba_ssm": _package_version("mamba-ssm"),
        "causal_conv1d": _package_version("causal-conv1d"),
        "cuda_device_index": device_index,
        "cuda_device": properties.name,
        "cuda_capability": "{}.{}".format(
            properties.major, properties.minor
        ),
        "cuda_total_memory_gib": round(
            properties.total_memory / (1024**3), 3
        ),
        "nvidia_smi": _nvidia_smi(),
    }


def _parameter_summary(model):
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return {
        "total": int(total),
        "trainable": int(trainable),
        "total_m": total / 1_000_000.0,
        "trainable_m": trainable / 1_000_000.0,
    }


def _mamba_modules(model):
    modules = []
    for name, module in model.named_modules():
        module_name = module.__class__.__module__.lower()
        class_name = module.__class__.__name__.lower()
        if "mamba" in module_name or class_name == "mamba":
            modules.append(name or "<root>")
    return modules


def _profile_macs(model, device, resolution):
    """Profile MACs without making unsupported Mamba operations disappear."""
    image = torch.zeros(
        1, 3, resolution, resolution, device=device, dtype=torch.float32
    )
    prompt = torch.zeros(1, device=device, dtype=torch.long)
    mamba_modules = _mamba_modules(model)
    mamba_warning = ""
    if mamba_modules:
        mamba_warning = (
            " Model contains Mamba selective-scan modules: {}."
        ).format(", ".join(mamba_modules))
    result = {
        "resolution": [resolution, resolution],
        "status": "unavailable",
        "macs": None,
        "flops": None,
        "tool": None,
        "mamba_modules": mamba_modules,
        "reason": None,
    }

    # thop's value is explicitly MACs, but it can silently omit custom ops;
    # presence of Mamba therefore downgrades the result to partial.
    try:
        from thop import profile  # type: ignore

        with torch.inference_mode():
            macs, _ = profile(
                model, inputs=(image, prompt), verbose=False
            )
        result["tool"] = "thop"
        result["macs"] = float(macs)
        if mamba_modules:
            result["status"] = "partial"
            result["reason"] = (
                "thop does not provide a validated rule for the Mamba "
                "selective-scan modules: {}. The reported value excludes "
                "or may incompletely count those operations and is not a "
                "complete model MACs total."
            ).format(", ".join(mamba_modules))
        else:
            result["status"] = "complete"
            result["reason"] = "All model modules were handled by thop."
        return result
    except ImportError:
        pass
    except Exception as error:  # profiler failures must be visible in report
        result["tool"] = "thop"
        result["reason"] = "thop failed: {}.{}".format(error, mamba_warning)
        return result

    # fvcore reports FLOPs, not MACs.  Keep this as a partial diagnostic and
    # never divide by two: that convention is not universal and would be a
    # fabricated MACs claim for this experiment.
    try:
        from fvcore.nn import FlopCountAnalysis  # type: ignore

        with torch.inference_mode():
            analysis = FlopCountAnalysis(model, (image, prompt))
            flops = float(analysis.total())
            unsupported = analysis.unsupported_ops()
        result["tool"] = "fvcore"
        result["flops"] = flops
        result["status"] = "partial"
        unsupported_text = ""
        if unsupported:
            unsupported_text = " Unsupported operators: {}.".format(
                dict(unsupported)
            )
        result["reason"] = (
            "fvcore reports FLOPs rather than MACs; no FLOPs-to-MACs "
            "conversion is applied.{}"
        ).format(unsupported_text)
        if mamba_modules:
            result["reason"] += (
                " Mamba selective-scan modules are present and must be "
                "accounted for separately: {}."
            ).format(", ".join(mamba_modules))
        return result
    except ImportError:
        result["reason"] = (
            "Neither thop nor fvcore is installed; MACs are unavailable.{}"
        ).format(mamba_warning)
    except Exception as error:
        result["tool"] = "fvcore"
        result["reason"] = "fvcore failed: {}".format(error)
    return result


def _autocast(enabled):
    if not enabled:
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.float16)


def _benchmark(model, device, resolution, amp, warmup, repeats, task_id):
    image = torch.rand(
        1, 3, resolution, resolution, device=device, dtype=torch.float32
    )
    prompt = torch.tensor([task_id], device=device, dtype=torch.long)
    mode = "amp_fp16" if amp else "fp32"
    result = {
        "resolution": [resolution, resolution],
        "batch_size": 1,
        "mode": mode,
        "amp": bool(amp),
        "warmup": warmup,
        "repeats": repeats,
        "cuda_synchronize_before": True,
        "cuda_synchronize_after": True,
        "status": "ok",
    }
    try:
        with torch.inference_mode():
            for _ in range(warmup):
                with _autocast(amp):
                    model(image, prompt)
            # Exclude compilation/cache warmup from both timing and peak.
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            timings = []
            start_events = []
            end_events = []
            for _ in range(repeats):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                with _autocast(amp):
                    model(image, prompt)
                end.record()
                start_events.append(start)
                end_events.append(end)
            torch.cuda.synchronize(device)
            timings = [
                start.elapsed_time(end)
                for start, end in zip(start_events, end_events)
            ]
            peak_bytes = torch.cuda.max_memory_allocated(device)
        if not timings or not all(math.isfinite(value) for value in timings):
            raise RuntimeError("CUDA event timing returned no finite samples")
        mean_ms = statistics.mean(timings)
        result.update(
            {
                "latency_ms_mean": mean_ms,
                "latency_ms_p50": statistics.median(timings),
                "latency_ms_p95": _percentile(timings, 95.0),
                "throughput_images_per_second": 1000.0 / mean_ms,
                "peak_cuda_memory_mb": peak_bytes / (1024**2),
                "peak_cuda_memory_gib": peak_bytes / (1024**3),
                "timing_method": "torch.cuda.Event elapsed_time; one event pair per repeat",
            }
        )
    except RuntimeError as error:
        result["status"] = "error"
        result["error"] = str(error)
        if "out of memory" in str(error).lower():
            torch.cuda.empty_cache()
    return result


def _percentile(values, percentile):
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * percentile / 100.0
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return values[lower]
    fraction = rank - lower
    return values[lower] + fraction * (values[upper] - values[lower])


def _format_report(report):
    params = report["params"]
    checkpoint = report["checkpoint"]
    lines = [
        "F01/F02 ProMamba efficiency profile",
        "weights: {}".format(report["weights"]),
        "Params: {:,} ({:.6f} M)".format(params["total"], params["total_m"]),
        "Checkpoint: {:.3f} MiB ({:.3f} MB decimal)".format(
            checkpoint["size_mib"], checkpoint["size_mb_decimal"]
        ),
        "GPU: {} | torch={} | CUDA={} | mamba_ssm={}".format(
            report["hardware"]["cuda_device"],
            report["hardware"]["torch"],
            report["hardware"]["torch_cuda"],
            report["hardware"]["mamba_ssm"],
        ),
        "Protocol: batch=1 warmup={} repeats={} CUDA synchronize before/after="
        "true/true".format(
            report["protocol"]["warmup"],
            report["protocol"]["repeats"],
        ),
    ]
    for row in report["benchmarks"]:
        if row["status"] != "ok":
            lines.append(
                "{} {}: ERROR {}".format(
                    row["resolution"], row["mode"], row.get("error", "unknown")
                )
            )
            continue
        lines.append(
            "{} {}: {:.3f} ms mean (p50 {:.3f}, p95 {:.3f}), "
            "{:.3f} images/s, peak {:.1f} MiB".format(
                row["resolution"],
                row["mode"],
                row["latency_ms_mean"],
                row["latency_ms_p50"],
                row["latency_ms_p95"],
                row["throughput_images_per_second"],
                row["peak_cuda_memory_mb"],
            )
        )
    macs = report.get("macs")
    if macs and macs.get("status") != "skipped":
        lines.append(
            "MACs @ {}: status={} tool={} value={} reason={}".format(
                macs["resolution"][0],
                macs["status"],
                macs["tool"],
                macs["macs"],
                macs["reason"],
            )
        )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Profile checkpoint parameters, MACs, latency, throughput, and CUDA memory."
    )
    parser.add_argument("--weights", required=True, help="model checkpoint (.pth)")
    parser.add_argument(
        "--output", default="", help="optional JSON output path"
    )
    parser.add_argument(
        "--warmup", type=int, default=20, help="warmup forwards per case"
    )
    parser.add_argument(
        "--repeats", type=int, default=100, help="timed forwards per case"
    )
    parser.add_argument(
        "--resolutions",
        type=int,
        nargs="+",
        default=[256, 1024],
        help="square input resolutions (default: 256 1024)",
    )
    parser.add_argument(
        "--macs-resolution",
        type=int,
        default=256,
        help="square resolution used by the optional MACs profiler (default: 256)",
    )
    parser.add_argument(
        "--task-id", type=int, default=0, choices=(0, 1, 2)
    )
    parser.add_argument(
        "--skip-macs", action="store_true", help="skip optional MACs profiler"
    )
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.repeats <= 0:
        parser.error("--warmup must be >= 0 and --repeats must be > 0")
    if any(resolution <= 0 for resolution in args.resolutions):
        parser.error("all resolutions must be positive")
    if args.macs_resolution <= 0:
        parser.error("--macs-resolution must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the F01/F02 efficiency profile")

    device = torch.device("cuda")
    # Import after CUDA validation so --help and syntax checks do not require
    # the server-only mamba_ssm dependency.
    from utils.checkpoint import load_model_for_inference

    weights = os.path.abspath(args.weights)
    model, checkpoint, model_config = load_model_for_inference(weights, device)
    model.eval()
    params = _parameter_summary(model)
    size_bytes = os.path.getsize(weights)
    report = {
        "experiment": "F01/F02",
        "weights": weights,
        "checkpoint": {
            "size_bytes": size_bytes,
            "size_mib": size_bytes / (1024**2),
            "size_mb_decimal": size_bytes / 1_000_000.0,
            "epoch": checkpoint.get("epoch"),
            "format_version": checkpoint.get("format_version"),
        },
        "model_config": model_config,
        "params": params,
        "hardware": _hardware_software(device),
        "protocol": {
            "batch_size": 1,
            "resolutions": args.resolutions,
            "modes": ["fp32", "amp_fp16"],
            "warmup": args.warmup,
            "repeats": args.repeats,
            "cuda_synchronize_before": True,
            "cuda_synchronize_after": True,
            "task_id": args.task_id,
        },
        "benchmarks": [],
    }
    for resolution in args.resolutions:
        for amp in (False, True):
            report["benchmarks"].append(
                _benchmark(
                    model,
                    device,
                    resolution,
                    amp,
                    args.warmup,
                    args.repeats,
                    args.task_id,
                )
            )
    if not args.skip_macs:
        report["macs"] = _profile_macs(model, device, args.macs_resolution)
    else:
        report["macs"] = {
            "status": "skipped",
            "resolution": [args.macs_resolution, args.macs_resolution],
            "macs": None,
            "reason": "--skip-macs was supplied",
        }
    if args.output:
        output_path = os.path.abspath(args.output)
        output_directory = os.path.dirname(output_path)
        if output_directory:
            os.makedirs(output_directory, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
    print(_format_report(report))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
