import argparse
import os
import sys

import torch


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from net.promamba import ProMamba  # noqa: E402
from utils.checkpoint import load_model_for_inference  # noqa: E402
from utils.losses import RestorationLoss  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--legacy-checkpoint", default="")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Mamba smoke test")
    device = torch.device("cuda")
    model = ProMamba(
        architecture="hierarchical",
        dim=48,
        stage_blocks=(2, 3, 4),
        d_state=16,
        norm_type="group",
        gate_mode="competitive",
        identity_init=True,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    image = torch.rand(
        args.batch_size,
        3,
        args.patch_size,
        args.patch_size,
        device=device,
    )
    target = torch.rand_like(image)
    prompt = torch.arange(args.batch_size, device=device) % 3
    criterion = RestorationLoss()
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        restored = model(image, prompt)
        losses, _ = criterion(restored, target)
        loss = losses.mean()
    loss.backward()
    if restored.shape != image.shape or not torch.isfinite(loss):
        raise RuntimeError("Hierarchical model smoke test failed")

    legacy_config = None
    if args.legacy_checkpoint:
        legacy_model, _, legacy_config = load_model_for_inference(
            args.legacy_checkpoint,
            device,
        )
        with torch.inference_mode():
            legacy_output = legacy_model(image[:1], prompt[:1])
        if legacy_output.shape != image[:1].shape:
            raise RuntimeError("Legacy checkpoint output shape mismatch")

    peak_memory = torch.cuda.max_memory_allocated() / (1024**3)
    print(
        "MODEL_V2_SMOKE_OK params={} loss={:.6f} peak_memory_gib={:.3f} "
        "legacy_config={}".format(
            parameter_count,
            loss.item(),
            peak_memory,
            legacy_config,
        )
    )


if __name__ == "__main__":
    main()
