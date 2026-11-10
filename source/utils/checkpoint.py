import os
import random
import tempfile
from datetime import datetime

import numpy as np
import torch

from net.promamba import ProMamba


FORMAT_VERSION = 6
LEGACY_JUNE_2026_CONFIG = {
    "in_channels": 3,
    "out_channels": 3,
    "dim": 64,
    "num_blocks": 6,
    "num_tasks": 3,
    "d_state": 16,
    "norm_type": "batch",
    "gate_mode": "legacy",
    "identity_init": False,
}


class IncompatibleCheckpointError(RuntimeError):
    pass


def load_checkpoint(path, device="cpu"):
    if not os.path.isfile(path):
        raise FileNotFoundError("Checkpoint not found: {}".format(path))
    checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise IncompatibleCheckpointError(
            "Unsupported checkpoint object: {}".format(type(checkpoint).__name__)
        )
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise IncompatibleCheckpointError("Checkpoint has no model state dictionary")
    return checkpoint, state_dict


def infer_model_config(checkpoint, state_dict):
    if "model_config" in checkpoint:
        config = dict(checkpoint["model_config"])
        config["identity_init"] = False
        return config

    if "backbone.0.local_conv.0.weight" in state_dict:
        return dict(LEGACY_JUNE_2026_CONFIG)

    if "prompt_gen.extractor.0.weight" in state_dict:
        family = "automatic-prompt April model"
    elif "task_embed.weight" in state_dict:
        family = "pre-local/global explicit-prompt model"
    else:
        family = "unknown legacy model"
    raise IncompatibleCheckpointError(
        "{} is incompatible with the current ProMamba architecture. "
        "Use its matching archived source instead.".format(family)
    )


def load_model_for_inference(path, device):
    checkpoint, state_dict = load_checkpoint(path, device=device)
    if "ema_state_dict" in checkpoint:
        state_dict = checkpoint["ema_state_dict"]
    model_config = infer_model_config(checkpoint, state_dict)
    model = ProMamba(**model_config).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, checkpoint, model_config


def atomic_torch_save(payload, path):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    file_descriptor, temporary_path = tempfile.mkstemp(
        prefix=".checkpoint-",
        suffix=".pth",
        dir=directory,
    )
    os.close(file_descriptor)
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def build_training_checkpoint(
    epoch,
    model,
    optimizer,
    scheduler,
    scaler,
    args,
    metrics,
    ema_state_dict=None,
    best_validation=None,
    data_generator_state=None,
):
    checkpoint = {
        "format_version": FORMAT_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "epoch": epoch,
        "model_config": dict(model.model_config),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "train_args": vars(args),
        "metrics": metrics,
        "best_validation": best_validation,
        "torch_rng_state": torch.get_rng_state(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else None,
    }
    if ema_state_dict is not None:
        checkpoint["ema_state_dict"] = ema_state_dict
    if data_generator_state is not None:
        checkpoint["data_generator_state"] = data_generator_state
    return checkpoint
