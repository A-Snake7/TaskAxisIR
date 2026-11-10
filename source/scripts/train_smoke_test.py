import argparse
import os
import tempfile
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import AllInOneDataset, BalancedTaskBatchSampler
from net.promamba import ProMamba
from utils.checkpoint import (
    atomic_torch_save,
    build_training_checkpoint,
    load_model_for_inference,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", default=".")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = AllInOneDataset(
        "{}/data/train".format(args.project_root),
        patch_size=256,
        train=True,
    )
    sampler = BalancedTaskBatchSampler(dataset, batch_size=4, seed=42)
    scheduled_counts = Counter()
    for batch in sampler:
        for index in batch:
            scheduled_counts[dataset.samples[index][0]] += 1
    assert set(scheduled_counts.values()) == {13990}, scheduled_counts

    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)
    image_input, image_target, prompt = next(iter(loader))
    image_input = image_input.to(device)
    image_target = image_target.to(device)
    prompt = prompt.to(device)
    model = ProMamba(
        norm_type="group",
        gate_mode="competitive",
        identity_init=True,
    ).to(device)
    criterion = nn.L1Loss()
    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=device.type == "cuda",
    ):
        restored = model(image_input, prompt)
        loss = criterion(restored, image_target)
    loss.backward()
    assert restored.shape == image_target.shape
    assert torch.isfinite(loss)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    checkpoint = build_training_checkpoint(
        epoch=1,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        args=args,
        metrics={"loss": loss.item()},
    )
    with tempfile.TemporaryDirectory() as temporary_directory:
        checkpoint_path = os.path.join(temporary_directory, "smoke.pth")
        atomic_torch_save(checkpoint, checkpoint_path)
        loaded_model, loaded_checkpoint, loaded_config = load_model_for_inference(
            checkpoint_path,
            device,
        )
        assert loaded_checkpoint["epoch"] == 1
        assert loaded_config["norm_type"] == "group"
        assert loaded_config["gate_mode"] == "competitive"
        del loaded_model
    print(
        "TRAIN_SMOKE_TEST_OK counts={} batch={} loss={:.6f}".format(
            dict(scheduled_counts),
            tuple(image_input.shape),
            loss.item(),
        )
    )


if __name__ == "__main__":
    main()
