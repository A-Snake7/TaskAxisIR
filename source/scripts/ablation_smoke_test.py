import os
import sys

import torch


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from net.promamba import ProMamba  # noqa: E402
from utils.checkpoint import load_model_for_inference  # noqa: E402


VARIANTS = {
    "full": {},
    "scan1": {"scan_directions": 1},
    "scan2": {"scan_directions": 2},
    "fixed4": {"direction_fusion": "mean"},
    "no_frequency": {"use_frequency_branch": False},
    "no_task_condition": {"use_task_condition": False},
    "shared_head": {"shared_head": True},
    "conv_bottleneck": {"bottleneck_type": "conv"},
    "mean_local_global": {"local_global_fusion": "mean"},
    "global_only": {"local_global_fusion": "global_only"},
    "local_only": {"local_global_fusion": "local_only"},
}


def run_variant(name, overrides, device):
    config = {
        "dim": 16,
        "architecture": "hierarchical",
        "stage_blocks": (1, 1, 1),
        "identity_init": True,
    }
    config.update(overrides)
    model = ProMamba(**config).to(device)
    image = torch.rand(3, 3, 32, 36, device=device)
    task = torch.tensor([0, 1, 2], dtype=torch.long, device=device)
    restored = model(image, task)
    if restored.shape != image.shape:
        raise RuntimeError(
            "{} returned {} instead of {}".format(
                name,
                tuple(restored.shape),
                tuple(image.shape),
            )
        )
    loss = (restored - image).square().mean()
    loss.backward()
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(
        "ABLATION_SMOKE_OK name={} params={} output={}".format(
            name,
            parameters,
            tuple(restored.shape),
        )
    )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Mamba smoke test")
    device = torch.device("cuda")
    for name, overrides in VARIANTS.items():
        run_variant(name, overrides, device)

    checkpoint_path = os.path.join(
        PROJECT_ROOT,
        "runs",
        "promamba_aio3_stage1",
        "epoch_0100.pth",
    )
    if os.path.isfile(checkpoint_path):
        model, checkpoint, config = load_model_for_inference(
            checkpoint_path,
            device,
        )
        with torch.inference_mode():
            output = model(
                torch.rand(1, 3, 32, 36, device=device),
                torch.tensor([1], device=device),
            )
        print(
            "LEGACY_FULL_CHECKPOINT_OK epoch={} output={} config={}".format(
                checkpoint.get("epoch"),
                tuple(output.shape),
                config,
            )
        )


if __name__ == "__main__":
    main()
