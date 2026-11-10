"""Bounded Blind-v3 CUDA smoke test using a verified Blind-v2 checkpoint.

This is a software integration check, not a paper experiment. It migrates the
EMA weights from a v2 checkpoint, runs a few real-Mamba optimization steps on
synthetic tensors, updates EMA only after successful AMP steps, and exercises
the global/detail tiled-routing path.
"""

import argparse
import json
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from net.promamba import ProMamba  # noqa: E402
from utils.ema import ModelEMA  # noqa: E402
from utils.inference import tiled_inference  # noqa: E402
from utils.losses import RestorationLoss, blind_v3_routing_loss  # noqa: E402


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def load_v3_from_v2(path, device, fuse_prompt_condition=False):
    checkpoint = torch.load(path, map_location="cpu")
    config = dict(checkpoint["model_config"])
    require(config.get("condition_mode") == "blind_v2", "Expected Blind-v2 weights")
    config.update(
        condition_mode="blind_v3",
        v3_local_route_prior=0.5,
        v3_fuse_prompt_condition=bool(fuse_prompt_condition),
    )
    model = ProMamba(**config).to(device)
    state_dict = checkpoint.get("ema_state_dict", checkpoint["model_state_dict"])
    migration = model.load_state_dict(state_dict, strict=False)
    require(not migration.unexpected_keys, "Unexpected v2 migration tensors")
    require(bool(migration.missing_keys), "V3 migration should add routing tensors")
    require(
        all("routing_fusion" in name for name in migration.missing_keys),
        "Only routing_fusion tensors may be new",
    )
    return model, config, list(migration.missing_keys)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--fuse-prompt-condition", action="store_true")
    parser.add_argument("--amp-init-scale", type=float, default=1024.0)
    args = parser.parse_args()
    require(args.steps > 0, "--steps must be positive")
    require(args.patch_size >= 16 and args.patch_size % 4 == 0, "Invalid patch size")
    require(args.amp_init_scale > 0, "--amp-init-scale must be positive")
    require(torch.cuda.is_available(), "CUDA is unavailable")

    torch.manual_seed(73)
    torch.cuda.manual_seed_all(73)
    device = torch.device("cuda")
    model, config, missing_keys = load_v3_from_v2(
        args.weights,
        device,
        fuse_prompt_condition=args.fuse_prompt_condition,
    )
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(init_scale=args.amp_init_scale)
    criterion = RestorationLoss()
    ema = ModelEMA(model, decay=0.99)
    records = []
    successful_steps = 0
    skipped_steps = 0

    for step in range(1, args.steps + 1):
        degraded = torch.rand(1, 3, args.patch_size, args.patch_size, device=device)
        target = torch.rand_like(degraded)
        context = torch.rand(1, 3, args.patch_size + 16, args.patch_size + 8, device=device)
        labels = torch.tensor([(step - 1) % 3], dtype=torch.long, device=device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            restored, auxiliary = model(
                degraded,
                task_prompt=labels,
                routing_context=context,
                teacher_forcing=0.0,
                return_aux=True,
            )
            restoration, _ = criterion(restored, target)
            routing = blind_v3_routing_loss(auxiliary, labels)
            loss = (
                restoration.mean()
                + 0.1 * routing["combined_cls"]
                + 0.01 * routing["probability_consistency"]
                + 0.01 * routing["feature_consistency"]
            )
        require(bool(torch.isfinite(restored).all()), "Non-finite restoration")
        require(bool(torch.isfinite(loss)), "Non-finite total loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        fusion_gradient = model.hierarchical.routing_fusion.gate[-1].weight.grad
        require(fusion_gradient is not None, "Missing fusion-gate gradient")
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        optimizer_stepped = scaler.get_scale() >= scale_before
        if optimizer_stepped:
            require(bool(torch.isfinite(grad_norm)), "Finite AMP step has invalid norm")
            require(
                bool(torch.isfinite(fusion_gradient).all()),
                "Finite AMP step has invalid fusion gradient",
            )
            ema.update(model)
            successful_steps += 1
        else:
            skipped_steps += 1
        records.append(
            {
                "step": step,
                "loss": float(loss.detach()),
                "grad_norm": float(grad_norm),
                "fusion_grad_l1": float(fusion_gradient.float().abs().sum()),
                "amp_scale": float(scaler.get_scale()),
                "optimizer_stepped": optimizer_stepped,
            }
        )

    require(successful_steps > 0, "AMP skipped every synthetic optimizer step")

    ema.module.eval()
    inference_input = torch.rand(1, 3, 96, 112, device=device)
    restored, routing = tiled_inference(
        ema.module,
        inference_input,
        tile_size=64,
        overlap=16,
        amp_enabled=True,
        return_routing=True,
    )
    require(restored.shape == inference_input.shape, "Tiled inference changed shape")
    require(bool(torch.isfinite(restored).all()), "Tiled output is non-finite")
    require(routing["routing_aggregation"] == "coverage_weighted_per_tile_fusion", "Wrong routing aggregation")
    require(bool(torch.isfinite(routing["task_probs"]).all()), "Routing is non-finite")

    report = {
        "status": "passed",
        "synthetic_only": True,
        "paper_result": False,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "source_weights": str(Path(args.weights).resolve()),
        "source_mode": "blind_v2",
        "target_mode": config["condition_mode"],
        "fuse_prompt_condition": bool(args.fuse_prompt_condition),
        "missing_keys": missing_keys,
        "steps": records,
        "successful_steps": successful_steps,
        "skipped_steps": skipped_steps,
        "tiled_route": routing["task_probs"].detach().float().cpu().tolist(),
        "tile_count": routing["tile_count"],
        "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 2**20, 2),
    }
    print("BLIND_V3_GPU_SMOKE=" + json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
