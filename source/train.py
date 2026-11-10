import argparse
import contextlib
import csv
import hashlib
import json
import logging
import math
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from dataset import AllInOneDataset, BalancedTaskBatchSampler, TASK_NAMES
from net.promamba import ProMamba
from utils.checkpoint import (
    atomic_torch_save,
    build_training_checkpoint,
    infer_model_config,
    load_checkpoint,
)
from utils.common import AverageMeter, psnr_per_image
from utils.ema import ModelEMA
from utils.inference import tiled_inference
from utils.losses import (
    RestorationLoss,
    blind_v2_routing_loss,
    blind_v3_routing_loss,
)
from utils.modes import BLIND_MODES, MULTIVIEW_BLIND_MODES


def parse_args():
    parser = argparse.ArgumentParser(description="Train ProMamba")
    parser.add_argument("--data_root", default="./data/train")
    parser.add_argument(
        "--train_manifest",
        default="",
        help="Optional scene-disjoint training CSV created by create_split_manifests.py.",
    )
    parser.add_argument(
        "--val_root",
        default="",
        help="Optional validation root. Do not use the final test set for selection.",
    )
    parser.add_argument(
        "--val_manifest",
        default="",
        help="Optional validation CSV. Uses --data_root when --val_root is empty.",
    )
    parser.add_argument(
        "--val_inference",
        choices=("auto", "crop", "full", "tile"),
        default="auto",
        help=(
            "auto uses full-image tiled validation for multiview blind modes, "
            "and crop validation for older models."
        ),
    )
    parser.add_argument("--val_tile_size", type=int, default=256)
    parser.add_argument("--val_overlap", type=int, default=32)
    parser.add_argument(
        "--best_selection_metric",
        choices=("auto", "macro_psnr", "avg5_proxy_psnr"),
        default="auto",
        help=(
            "Metric used to save best.pth. auto preserves macro PSNR for existing "
            "modes and uses the AIO3-style Avg-5 proxy for blind_v3."
        ),
    )
    parser.add_argument("--save_dir", default="./runs")
    parser.add_argument("--run_name", default="")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=2000,
        help="Fixed optimizer steps per epoch; avoids oversampling tiny tasks.",
    )
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--backbone_lr", type=float, default=2e-5)
    parser.add_argument("--condition_lr", type=float, default=5e-5)
    parser.add_argument("--router_lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument(
        "--task_weights",
        type=float,
        nargs=3,
        default=(1.0, 1.0, 1.0),
        metavar=("DERAIN", "DEHAZE", "DENOISE"),
    )
    parser.add_argument("--save_every", type=int, default=10)
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument(
        "--resume",
        default="",
        help="Resume an identical architecture, optimizer, and scheduler.",
    )
    checkpoint_group.add_argument(
        "--init_weights",
        default="",
        help="Warm-start matching weights while allowing a new architecture.",
    )
    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument("--amp", dest="amp", action="store_true")
    amp_group.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=True)
    parser.add_argument(
        "--amp_init_scale",
        type=float,
        default=1024.0,
        help=(
            "Initial CUDA GradScaler value for new runs. Resume restores the "
            "checkpoint scaler state instead."
        ),
    )
    parser.add_argument(
        "--norm_type",
        choices=("batch", "group"),
        default="group",
        help="Used for new runs. Resume always uses checkpoint architecture.",
    )
    parser.add_argument(
        "--gate_mode",
        choices=("legacy", "competitive"),
        default="competitive",
        help="Used for new runs. Resume always uses checkpoint architecture.",
    )
    parser.add_argument(
        "--architecture",
        choices=("lightweight", "hierarchical"),
        default="hierarchical",
        help="New runs default to the three-scale four-direction model.",
    )
    parser.add_argument("--dim", type=int, default=48)
    parser.add_argument(
        "--stage_blocks",
        type=int,
        nargs=3,
        default=(2, 3, 4),
        metavar=("HIGH", "MID", "LOW"),
    )
    parser.add_argument("--d_state", type=int, default=16)
    parser.add_argument(
        "--scan_directions",
        type=int,
        choices=(1, 2, 4),
        default=4,
    )
    reverse_scan_group = parser.add_mutually_exclusive_group()
    reverse_scan_group.add_argument(
        "--batch_reverse_scans",
        dest="batch_reverse_scans",
        action="store_true",
    )
    reverse_scan_group.add_argument(
        "--no_batch_reverse_scans",
        dest="batch_reverse_scans",
        action="store_false",
    )
    parser.set_defaults(batch_reverse_scans=True)
    parser.add_argument(
        "--direction_fusion",
        choices=("gated", "mean"),
        default="gated",
    )
    parser.add_argument(
        "--local_global_fusion",
        choices=("gated", "mean", "global_only", "local_only"),
        default="gated",
    )
    frequency_group = parser.add_mutually_exclusive_group()
    frequency_group.add_argument(
        "--frequency_branch",
        dest="use_frequency_branch",
        action="store_true",
    )
    frequency_group.add_argument(
        "--no_frequency_branch",
        dest="use_frequency_branch",
        action="store_false",
    )
    condition_group = parser.add_mutually_exclusive_group()
    condition_group.add_argument(
        "--task_condition",
        dest="use_task_condition",
        action="store_true",
    )
    condition_group.add_argument(
        "--no_task_condition",
        dest="use_task_condition",
        action="store_false",
    )
    parser.set_defaults(
        use_frequency_branch=True,
        use_task_condition=True,
    )
    parser.add_argument("--shared_head", action="store_true")
    parser.add_argument(
        "--bottleneck_type",
        choices=("mamba", "conv"),
        default="mamba",
    )
    parser.add_argument(
        "--condition_mode",
        choices=("oracle",) + BLIND_MODES,
        default="oracle",
        help=(
            "oracle requires Task ID at inference; blind_soft predicts a soft "
            "task token; blind_v2 aligns full-image and local-crop conditions; "
            "blind_v3 reliability-fuses both routes for task-head selection."
        ),
    )
    parser.add_argument("--routing_context_size", type=int, default=256)
    parser.add_argument("--local_condition_scale", type=float, default=0.1)
    parser.add_argument(
        "--v3_local_route_prior",
        type=float,
        default=0.5,
        help="Initial blind_v3 probability assigned to the local/detail route.",
    )
    v3_condition_group = parser.add_mutually_exclusive_group()
    v3_condition_group.add_argument(
        "--v3_fuse_prompt_condition",
        dest="v3_fuse_prompt_condition",
        action="store_true",
        help="Use the stable fused route for the discrete PromptAffine task token.",
    )
    v3_condition_group.add_argument(
        "--no_v3_fuse_prompt_condition",
        dest="v3_fuse_prompt_condition",
        action="store_false",
        help="Ablation: keep PromptAffine on the global-only task route.",
    )
    parser.set_defaults(v3_fuse_prompt_condition=False)
    parser.add_argument(
        "--task_cls_weight",
        type=float,
        default=0.1,
        help="Cross-entropy weight for the blind degradation encoder.",
    )
    parser.add_argument(
        "--local_cls_weight", type=float, default=0.5,
        help="Local-view CE multiplier inside --task_cls_weight.",
    )
    parser.add_argument(
        "--global_cls_weight", type=float, default=0.5,
        help="blind_v3 global-view CE multiplier inside --task_cls_weight.",
    )
    parser.add_argument(
        "--fusion_cls_weight", type=float, default=1.0,
        help="blind_v3 fused-route CE multiplier inside --task_cls_weight.",
    )
    parser.add_argument(
        "--routing_consistency_weight", type=float, default=0.05,
        help=(
            "V2 fixed-global KL weight; V3 per-sample stronger-view "
            "distillation weight."
        ),
    )
    parser.add_argument(
        "--v3_consistency_weight",
        type=float,
        default=0.01,
        help="blind_v3 per-sample stronger-view distillation weight.",
    )
    parser.add_argument(
        "--feature_consistency_weight", type=float, default=0.01,
        help="View-feature consistency weight (teacher selection depends on mode).",
    )
    parser.add_argument(
        "--blind_teacher_forcing_start",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--blind_teacher_forcing_end",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--blind_teacher_forcing_epochs",
        type=int,
        default=20,
        help="Linearly replace oracle routing with blind routing over these epochs.",
    )
    parser.add_argument(
        "--router_warmup_epochs",
        type=int,
        default=3,
        help="Train only blind routing/continuous-condition modules initially.",
    )
    parser.add_argument("--ssim_weight", type=float, default=0.1)
    parser.add_argument("--edge_weight", type=float, default=0.05)
    parser.add_argument("--frequency_weight", type=float, default=0.05)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_logger(run_dir):
    logger = logging.getLogger("promamba.train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(
        os.path.join(run_dir, "training.log"),
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def prepare_run_directory(args):
    run_name = args.run_name or datetime.now().strftime("promamba_%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.save_dir, run_name)
    is_resume = bool(args.resume)
    os.makedirs(run_dir, exist_ok=is_resume)
    config_name = (
        "resume_config_{}.json".format(
            datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        if is_resume
        else "config.json"
    )
    with open(
        os.path.join(run_dir, config_name),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(vars(args), handle, ensure_ascii=False, indent=2)
    return run_dir


def create_model(args, device):
    resume_checkpoint = None
    init_report = None
    if args.resume:
        resume_checkpoint, state_dict = load_checkpoint(args.resume, device=device)
        model_config = infer_model_config(resume_checkpoint, state_dict)
        model = ProMamba(**model_config).to(device)
        model.load_state_dict(state_dict, strict=True)
    else:
        model = ProMamba(
            in_channels=3,
            out_channels=3,
            dim=args.dim,
            num_blocks=6,
            d_state=args.d_state,
            norm_type=args.norm_type,
            gate_mode=args.gate_mode,
            identity_init=True,
            architecture=args.architecture,
            stage_blocks=tuple(args.stage_blocks),
            scan_directions=args.scan_directions,
            direction_fusion=args.direction_fusion,
            local_global_fusion=args.local_global_fusion,
            use_frequency_branch=args.use_frequency_branch,
            use_task_condition=args.use_task_condition,
            shared_head=args.shared_head,
            bottleneck_type=args.bottleneck_type,
            condition_mode=args.condition_mode,
            batch_reverse_scans=args.batch_reverse_scans,
            routing_context_size=args.routing_context_size,
            local_condition_scale=args.local_condition_scale,
            v3_local_route_prior=args.v3_local_route_prior,
            v3_fuse_prompt_condition=args.v3_fuse_prompt_condition,
        ).to(device)
        if args.init_weights:
            init_checkpoint, init_state_dict = load_checkpoint(
                args.init_weights,
                device=device,
            )
            init_source_config = init_checkpoint.get("model_config", {})
            if "ema_state_dict" in init_checkpoint:
                init_state_dict = init_checkpoint["ema_state_dict"]
            current_state = model.state_dict()
            compatible_state = {
                key: value
                for key, value in init_state_dict.items()
                if key in current_state and current_state[key].shape == value.shape
            }
            incompatible_shapes = {
                key: (tuple(value.shape), tuple(current_state[key].shape))
                for key, value in init_state_dict.items()
                if key in current_state and current_state[key].shape != value.shape
            }
            load_result = model.load_state_dict(compatible_state, strict=False)
            migration = None
            if (
                init_source_config.get("condition_mode") == "blind_v2"
                and args.condition_mode == "blind_v3"
            ):
                unexpected_missing = [
                    key
                    for key in load_result.missing_keys
                    if "routing_fusion" not in key
                ]
                if unexpected_missing or incompatible_shapes:
                    raise RuntimeError(
                        "Unsafe blind_v2 to blind_v3 warm start: only routing_fusion "
                        "parameters may be new, but found missing={} shape_mismatches={}".format(
                            unexpected_missing,
                            sorted(incompatible_shapes),
                        )
                    )
                migration = "blind_v2_to_blind_v3"
            init_report = {
                "source": args.init_weights,
                "loaded_keys": len(compatible_state),
                "missing_keys": list(load_result.missing_keys),
                "unexpected_source_keys": [
                    key for key in init_state_dict if key not in current_state
                ],
                "incompatible_shapes": incompatible_shapes,
                "migration": migration,
            }
    return model, resume_checkpoint, init_report


def configure_training_protocol(args, model, resume_checkpoint=None):
    """Resolve data/loss/validation settings from the actual resumed model.

    In particular, a resumed blind-v2 checkpoint must not silently construct
    three-field crop-only datasets because the CLI defaults to oracle mode.
    """
    restored_settings = {}
    model_config = model.model_config
    # Keep run metadata truthful on resume as well as the model itself: CLI
    # defaults may describe a different width, block count or ablation setting.
    for name, value in model_config.items():
        if hasattr(args, name):
            setattr(args, name, value)
    args.condition_mode = model_config.get("condition_mode", "oracle")
    args.routing_context_size = model_config.get(
        "routing_context_size", args.routing_context_size
    )
    args.local_condition_scale = model_config.get(
        "local_condition_scale", args.local_condition_scale
    )
    args.v3_local_route_prior = model_config.get(
        "v3_local_route_prior", args.v3_local_route_prior
    )
    if args.condition_mode in MULTIVIEW_BLIND_MODES and resume_checkpoint is not None:
        saved_args = resume_checkpoint.get("train_args", {})
        for name in (
            "data_root", "train_manifest", "val_root", "val_manifest",
            "batch_size", "steps_per_epoch", "amp", "amp_init_scale",
            "ema_decay", "grad_clip",
            "lr", "backbone_lr", "condition_lr", "router_lr", "weight_decay",
            "task_cls_weight", "global_cls_weight", "local_cls_weight",
            "fusion_cls_weight", "routing_consistency_weight",
            "v3_consistency_weight",
            "feature_consistency_weight", "blind_teacher_forcing_start",
            "blind_teacher_forcing_end", "blind_teacher_forcing_epochs",
            "router_warmup_epochs", "val_inference", "val_tile_size",
            "val_overlap", "best_selection_metric", "patch_size", "seed",
            "task_weights",
            "ssim_weight", "edge_weight", "frequency_weight",
        ):
            if name in saved_args:
                if getattr(args, name) != saved_args[name]:
                    restored_settings[name] = saved_args[name]
                setattr(args, name, saved_args[name])
        saved_data = resume_checkpoint.get("data_protocol")
        if saved_data:
            # The provenance record is authoritative even if a caller omitted
            # paths from train_args. Never fall back to the legacy data/train.
            for name in ("data_root", "val_root"):
                value = saved_data[name]
                if getattr(args, name) != value:
                    restored_settings[name] = value
                setattr(args, name, value)
            for name in ("train_manifest", "val_manifest"):
                value = saved_data[name]["path"]
                if getattr(args, name) != value:
                    restored_settings[name] = value
                setattr(args, name, value)
    if args.val_inference == "auto":
        args.val_inference = (
            "tile" if args.condition_mode in MULTIVIEW_BLIND_MODES else "crop"
        )
    if args.val_inference not in ("crop", "full", "tile"):
        raise ValueError("Unsupported validation inference protocol")
    if args.val_tile_size <= 0 or not 0 <= args.val_overlap < args.val_tile_size:
        raise ValueError("Validation tiling requires 0 <= val_overlap < val_tile_size")
    for name in (
        "task_cls_weight", "global_cls_weight", "local_cls_weight",
        "fusion_cls_weight", "routing_consistency_weight", "v3_consistency_weight",
        "feature_consistency_weight",
    ):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            raise ValueError("{} must be finite and non-negative".format(name))
    if not math.isfinite(args.amp_init_scale) or args.amp_init_scale <= 0:
        raise ValueError("amp_init_scale must be finite and positive")
    if args.condition_mode in MULTIVIEW_BLIND_MODES:
        if model_config.get("architecture") != "hierarchical":
            raise ValueError("{} requires the hierarchical architecture".format(args.condition_mode))
        if not args.resume and not args.init_weights and args.router_warmup_epochs > 0:
            raise ValueError(
                "Training {} from scratch requires --router_warmup_epochs 0; "
                "router-only warmup would freeze a randomly initialized backbone. "
                "Use --init_weights only with an independently verified clean checkpoint."
                .format(args.condition_mode)
            )
    if args.condition_mode == "blind_v3" and not 0.0 < args.v3_local_route_prior < 1.0:
        raise ValueError("v3_local_route_prior must be strictly between 0 and 1")
    if args.best_selection_metric == "auto":
        args.best_selection_metric = (
            "avg5_proxy_psnr"
            if args.condition_mode == "blind_v3"
            else "macro_psnr"
        )
    selection_metric_key = "val_{}".format(args.best_selection_metric)
    args.val_protocol = {
        "inference": args.val_inference,
        "full_image": args.val_inference != "crop",
        "batch_size": args.batch_size if args.val_inference == "crop" else 1,
        "patch_size": args.patch_size if args.val_inference == "crop" else None,
        "tile_size": args.val_tile_size if args.val_inference == "tile" else None,
        "overlap": args.val_overlap if args.val_inference == "tile" else None,
        "routing_context_size": (
            args.routing_context_size
            if args.condition_mode in MULTIVIEW_BLIND_MODES
            else None
        ),
        "condition_mode": args.condition_mode,
        "v3_fuse_prompt_condition": (
            args.v3_fuse_prompt_condition
            if args.condition_mode == "blind_v3"
            else None
        ),
        "selection_metric": selection_metric_key,
        "selection_metric_description": (
            "EMA validation (derain + dehaze + 3 * denoise) / 5"
            if args.best_selection_metric == "avg5_proxy_psnr"
            else "EMA validation macro-task PSNR"
        ),
    }
    return restored_settings


def snapshot_data_protocol(args, resume_checkpoint=None):
    """Bind dataset roots and manifest bytes to this training run.

    This is a provenance/accidental-change guard, not a substitute for the
    scene and image-content leakage audits in scripts/audit_clean_aio3_*.py.
    Resuming v2 cannot silently change its selection data or split definition.
    """
    is_multiview = args.condition_mode in MULTIVIEW_BLIND_MODES
    if is_multiview and (not args.train_manifest or not args.val_manifest):
        raise ValueError(
            "Multiview blind training requires explicit --train_manifest and --val_manifest "
            "from an audited scene-disjoint split; the legacy data/train "
            "directory must not be used implicitly."
        )
    args.data_root = os.path.realpath(os.path.expanduser(args.data_root))
    args.val_root = args.val_root or (args.data_root if args.val_manifest else "")
    if args.val_root:
        args.val_root = os.path.realpath(os.path.expanduser(args.val_root))
    for name in ("data_root", "val_root"):
        path = getattr(args, name)
        if path and not os.path.isdir(path):
            raise FileNotFoundError("{} does not exist: {}".format(name, path))

    def file_record(path):
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return {"path": path, "sha256": digest.hexdigest()}

    protocol = {
        "version": 1,
        "data_root": args.data_root,
        "val_root": args.val_root,
    }
    for name in ("train_manifest", "val_manifest"):
        path = getattr(args, name)
        if path:
            path = os.path.realpath(os.path.expanduser(path))
            setattr(args, name, path)
        protocol[name] = file_record(path) if path else None
    if is_multiview and args.train_manifest == args.val_manifest:
        raise ValueError("Training and validation manifests must be different files")

    audit_paths = set()
    for root in (args.data_root, args.val_root):
        if root:
            audit_paths.add(os.path.join(root, "audit_report.json"))
    for manifest in (args.train_manifest, args.val_manifest):
        if manifest:
            audit_paths.add(os.path.join(os.path.dirname(manifest), "split_audit.json"))
    protocol["audit_records"] = [
        file_record(path) for path in sorted(audit_paths) if os.path.isfile(path)
    ]
    if is_multiview and resume_checkpoint is not None:
        expected = resume_checkpoint.get("data_protocol")
        if expected is None:
            raise ValueError(
                "Multiview blind resume requires checkpoint data_protocol provenance; "
                "use a verified clean initialization in a new run instead."
            )
        if expected != protocol:
            changed = [
                name for name in sorted(set(expected) | set(protocol))
                if expected.get(name) != protocol.get(name)
            ]
            raise ValueError(
                "Data protocol changed on resume ({}). Dataset paths, manifest "
                "SHA256 values and audit records must match the checkpoint; "
                "start a separately identified run for an intentional change."
                .format(", ".join(changed))
            )
    args.data_protocol = protocol
    return protocol


def blind_teacher_forcing_ratio(args, epoch, condition_mode=None):
    condition_mode = condition_mode or args.condition_mode
    if condition_mode not in BLIND_MODES:
        return 0.0
    start = float(args.blind_teacher_forcing_start)
    end = float(args.blind_teacher_forcing_end)
    if not 0.0 <= start <= 1.0 or not 0.0 <= end <= 1.0:
        raise ValueError("Blind teacher-forcing ratios must be between 0 and 1")
    schedule_epochs = max(1, int(args.blind_teacher_forcing_epochs))
    if schedule_epochs == 1:
        return end
    progress = min(max((epoch - 1) / float(schedule_epochs - 1), 0.0), 1.0)
    return start + progress * (end - start)


def _blind_parameter_group(name):
    if (
        "degradation_encoder" in name
        or "degradation_condition" in name
        or "local_condition_adapter" in name
        or "routing_fusion" in name
    ):
        return "router"
    if "task_embed" in name or "task_heads" in name:
        return "condition"
    return "backbone"


def build_optimizer(model, args):
    is_blind = model.model_config.get("condition_mode", "oracle") in BLIND_MODES
    if not is_blind:
        return torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    grouped_parameters = {"backbone": [], "condition": [], "router": []}
    for name, parameter in model.named_parameters():
        grouped_parameters[_blind_parameter_group(name)].append(parameter)
    learning_rates = {
        "backbone": args.backbone_lr,
        "condition": args.condition_lr,
        "router": args.router_lr,
    }
    parameter_groups = [
        {
            "params": grouped_parameters[name],
            "lr": learning_rates[name],
            "group_name": name,
        }
        for name in ("backbone", "condition", "router")
        if grouped_parameters[name]
    ]
    return torch.optim.AdamW(
        parameter_groups,
        weight_decay=args.weight_decay,
    )


def set_blind_trainable(model, epoch, warmup_epochs):
    is_blind = model.model_config.get("condition_mode", "oracle") in BLIND_MODES
    router_only = is_blind and epoch <= max(0, int(warmup_epochs))
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(
            not router_only or _blind_parameter_group(name) == "router"
        )
    return router_only


def realign_cosine_scheduler(scheduler, optimizer, completed_epochs, total_epochs):
    """Align resumed LR to the current requested cosine horizon."""
    total_epochs = max(1, int(total_epochs))
    completed_epochs = min(max(0, int(completed_epochs)), total_epochs)
    scheduler.T_max = total_epochs
    scheduler.last_epoch = completed_epochs
    cosine = 0.5 * (
        1.0 + math.cos(math.pi * completed_epochs / float(total_epochs))
    )
    learning_rates = []
    for base_lr, group in zip(scheduler.base_lrs, optimizer.param_groups):
        learning_rate = scheduler.eta_min + (
            base_lr - scheduler.eta_min
        ) * cosine
        group["lr"] = learning_rate
        learning_rates.append(learning_rate)
    scheduler._last_lr = learning_rates


def unpack_dataset_batch(batch):
    if len(batch) == 3:
        return batch[0], batch[1], batch[2], None
    if len(batch) == 4:
        return batch[0], batch[1], batch[2], batch[3]
    raise ValueError("Expected (input, target, task[, full-image routing context])")


def validate(
    model, loader, device, amp_enabled,
    inference_mode="crop", tile_size=256, overlap=32,
):
    if inference_mode not in ("crop", "full", "tile"):
        raise ValueError("Validation inference must be crop, full, or tile")
    task_psnr = {task_id: AverageMeter() for task_id in TASK_NAMES}
    task_psnr_gains = {task_id: [] for task_id in TASK_NAMES}
    task_accuracy = AverageMeter()
    global_task_accuracy = AverageMeter()
    local_task_accuracy = AverageMeter()
    local_route_weight = AverageMeter()
    routing_entropy = AverageMeter()
    local_global_js = AverageMeter()
    task_correct = {task_id: 0 for task_id in TASK_NAMES}
    task_total = {task_id: 0 for task_id in TASK_NAMES}
    calibration_bins = 15
    bin_count = torch.zeros(calibration_bins, dtype=torch.float64)
    bin_correct = torch.zeros_like(bin_count)
    bin_confidence = torch.zeros_like(bin_count)
    confusion = torch.zeros(
        len(TASK_NAMES),
        len(TASK_NAMES),
        dtype=torch.long,
    )
    global_confusion = torch.zeros_like(confusion)
    local_confusion = torch.zeros_like(confusion)
    global_task_correct = {task_id: 0 for task_id in TASK_NAMES}
    local_task_correct = {task_id: 0 for task_id in TASK_NAMES}
    condition_mode = model.model_config.get("condition_mode", "oracle")
    is_blind = condition_mode in BLIND_MODES
    is_multiview_blind = condition_mode in MULTIVIEW_BLIND_MODES
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            image_input, image_target, task_prompt, routing_context = unpack_dataset_batch(batch)
            image_input = image_input.to(device, non_blocking=True)
            image_target = image_target.to(device, non_blocking=True)
            task_prompt = task_prompt.to(device, non_blocking=True)
            auxiliary = None
            batch_js = None
            task_probs = None
            global_probs = None
            local_probs = None
            if inference_mode == "tile":
                restored, routing = tiled_inference(
                    model, image_input,
                    task_prompt=None if is_blind else task_prompt,
                    tile_size=tile_size, overlap=overlap,
                    amp_enabled=amp_enabled, return_routing=True,
                )
                if is_blind:
                    task_probs = routing["task_probs"]
                    batch_js = routing.get("local_global_js")
                    local_probs = routing.get("local_task_probs")
                    global_probs = routing.get("global_task_probs")
                    if routing.get("local_route_weight") is not None:
                        local_route_weight.update(
                            float(routing["local_route_weight"]), image_input.size(0)
                        )
            else:
                if routing_context is not None:
                    routing_context = routing_context.to(device, non_blocking=True)
                with (
                    torch.autocast(device_type="cuda", dtype=torch.float16)
                    if amp_enabled and device.type == "cuda" else contextlib.nullcontext()
                ):
                    if is_blind:
                        extra_kwargs = (
                            {"routing_context": routing_context}
                            if is_multiview_blind
                            else {}
                        )
                        restored, auxiliary = model(
                            image_input, task_prompt=None, return_aux=True,
                            **extra_kwargs,
                        )
                        task_probs = auxiliary["task_probs"]
                        global_probs = auxiliary.get("global_task_probs")
                        if auxiliary.get("local_route_weight") is not None:
                            local_route_weight.update(
                                auxiliary["local_route_weight"].float().mean().item(),
                                image_input.size(0),
                            )
                    else:
                        restored = model(image_input, task_prompt)
            if is_blind:
                if task_probs is None:
                    raise RuntimeError("Blind validation did not return routing probabilities")
                task_probs = task_probs.float()
                confidences, predictions = task_probs.max(dim=1)
                correct = predictions == task_prompt
                task_accuracy.update(correct.float().mean().item(), image_input.size(0))
                routing_entropy.update(
                    -(task_probs * task_probs.clamp_min(1e-8).log()).sum(1).mean().item(),
                    image_input.size(0),
                )
                batch_confusion = torch.bincount(
                    (task_prompt * len(TASK_NAMES) + predictions).detach().cpu(),
                    minlength=len(TASK_NAMES) ** 2,
                ).reshape(len(TASK_NAMES), len(TASK_NAMES))
                confusion += batch_confusion
                bin_indices = (
                    (confidences * calibration_bins).long().clamp(0, calibration_bins - 1)
                ).cpu()
                bin_count += torch.bincount(bin_indices, minlength=calibration_bins)
                bin_correct.scatter_add_(0, bin_indices, correct.double().cpu())
                bin_confidence.scatter_add_(0, bin_indices, confidences.double().cpu())
                for task_id in TASK_NAMES:
                    selected = task_prompt == task_id
                    task_total[task_id] += int(selected.sum().item())
                    task_correct[task_id] += int(correct[selected].sum().item())
                if auxiliary is not None and auxiliary.get("local_task_logits") is not None:
                    local_probs = auxiliary.get("local_task_probs")
                    if local_probs is None:
                        local_probs = F.softmax(auxiliary["local_task_logits"].float(), dim=1)
                    else:
                        local_probs = local_probs.float()
                    reference_probs = (
                        global_probs.float()
                        if global_probs is not None
                        else task_probs
                    )
                    midpoint = 0.5 * (reference_probs + local_probs)
                    midpoint_log = midpoint.clamp_min(1e-8).log()
                    batch_js = 0.5 * (
                        (
                            reference_probs
                            * (reference_probs.clamp_min(1e-8).log() - midpoint_log)
                        ).sum(1)
                        + (local_probs * (local_probs.clamp_min(1e-8).log() - midpoint_log)).sum(1)
                    )
                if local_probs is not None:
                    # For tiled validation this is image-level accuracy after
                    # averaging tile probabilities, not per-tile accuracy.
                    local_task_accuracy.update(
                        (local_probs.argmax(1) == task_prompt).float().mean().item(),
                        image_input.size(0),
                    )
                    local_predictions = local_probs.argmax(1)
                    local_confusion += torch.bincount(
                        (task_prompt * len(TASK_NAMES) + local_predictions).detach().cpu(),
                        minlength=len(TASK_NAMES) ** 2,
                    ).reshape(len(TASK_NAMES), len(TASK_NAMES))
                    for task_id in TASK_NAMES:
                        selected = task_prompt == task_id
                        local_task_correct[task_id] += int(
                            (local_predictions[selected] == task_prompt[selected]).sum().item()
                        )
                if global_probs is not None:
                    global_probs = global_probs.float()
                    global_predictions = global_probs.argmax(1)
                    global_task_accuracy.update(
                        (global_predictions == task_prompt).float().mean().item(),
                        image_input.size(0),
                    )
                    global_confusion += torch.bincount(
                        (task_prompt * len(TASK_NAMES) + global_predictions).detach().cpu(),
                        minlength=len(TASK_NAMES) ** 2,
                    ).reshape(len(TASK_NAMES), len(TASK_NAMES))
                    for task_id in TASK_NAMES:
                        selected = task_prompt == task_id
                        global_task_correct[task_id] += int(
                            (global_predictions[selected] == task_prompt[selected]).sum().item()
                        )
                if batch_js is not None:
                    local_global_js.update(
                        torch.as_tensor(batch_js).float().mean().item(), image_input.size(0)
                    )
            restored = restored.clamp(0, 1)
            values = psnr_per_image(restored.float(), image_target.float())
            input_values = psnr_per_image(
                image_input.float().clamp(0, 1), image_target.float()
            )
            gains = values - input_values
            for task_id in TASK_NAMES:
                selected = task_prompt == task_id
                if selected.any():
                    for value in values[selected]:
                        task_psnr[task_id].update(value.item())
                    task_psnr_gains[task_id].extend(
                        float(value) for value in gains[selected].detach().cpu()
                    )
    results = {
        "val_{}_psnr".format(TASK_NAMES[task_id]): meter.avg
        for task_id, meter in task_psnr.items()
        if meter.count
    }
    results["val_macro_psnr"] = (
        sum(results.values()) / len(results) if results else float("nan")
    )
    avg5_keys = ("val_derain_psnr", "val_dehaze_psnr", "val_denoise_psnr")
    if all(key in results for key in avg5_keys):
        results["val_avg5_proxy_psnr"] = (
            results["val_derain_psnr"]
            + results["val_dehaze_psnr"]
            + 3.0 * results["val_denoise_psnr"]
        ) / 5.0
    for task_id, task_name in TASK_NAMES.items():
        gains = task_psnr_gains[task_id]
        if gains:
            results["val_{}_gain_mean".format(task_name)] = float(np.mean(gains))
            results["val_{}_gain_p10".format(task_name)] = float(np.percentile(gains, 10))
            results["val_{}_gain_min".format(task_name)] = float(np.min(gains))
            results["val_{}_negative_gain_count".format(task_name)] = int(
                np.sum(np.asarray(gains) < 0.0)
            )
    if is_blind:
        results["val_task_accuracy"] = task_accuracy.avg
        results["val_routing_entropy"] = routing_entropy.avg
        populated = bin_count > 0
        results["val_routing_ece"] = (
            (
                (bin_correct[populated] - bin_confidence[populated]).abs().sum()
                / bin_count.sum().clamp_min(1)
            ).item()
        )
        if local_task_accuracy.count:
            results["val_local_task_accuracy"] = local_task_accuracy.avg
            results["val_local_task_confusion"] = json.dumps(local_confusion.tolist())
            for task_id, task_name in TASK_NAMES.items():
                results["val_local_{}_task_accuracy".format(task_name)] = (
                    local_task_correct[task_id] / task_total[task_id]
                    if task_total[task_id]
                    else float("nan")
                )
        if global_task_accuracy.count:
            results["val_global_task_accuracy"] = global_task_accuracy.avg
            results["val_global_task_confusion"] = json.dumps(global_confusion.tolist())
            for task_id, task_name in TASK_NAMES.items():
                results["val_global_{}_task_accuracy".format(task_name)] = (
                    global_task_correct[task_id] / task_total[task_id]
                    if task_total[task_id]
                    else float("nan")
                )
        if local_route_weight.count:
            results["val_local_route_weight"] = local_route_weight.avg
        if local_global_js.count:
            results["val_local_global_js"] = local_global_js.avg
        for task_id, task_name in TASK_NAMES.items():
            results["val_{}_task_accuracy".format(task_name)] = (
                task_correct[task_id] / task_total[task_id]
                if task_total[task_id]
                else float("nan")
            )
        results["val_task_confusion"] = json.dumps(confusion.tolist())
        per_task_route_accuracy = [
            results["val_{}_task_accuracy".format(task_name)]
            for task_name in TASK_NAMES.values()
            if math.isfinite(results["val_{}_task_accuracy".format(task_name)])
        ]
        results["val_macro_task_accuracy"] = (
            sum(per_task_route_accuracy) / len(per_task_route_accuracy)
            if per_task_route_accuracy
            else float("nan")
        )
        if local_task_accuracy.count:
            local_per_task = [
                results["val_local_{}_task_accuracy".format(task_name)]
                for task_name in TASK_NAMES.values()
                if math.isfinite(
                    results["val_local_{}_task_accuracy".format(task_name)]
                )
            ]
            results["val_local_macro_task_accuracy"] = (
                sum(local_per_task) / len(local_per_task)
                if local_per_task
                else float("nan")
            )
        if global_task_accuracy.count:
            global_per_task = [
                results["val_global_{}_task_accuracy".format(task_name)]
                for task_name in TASK_NAMES.values()
                if math.isfinite(
                    results["val_global_{}_task_accuracy".format(task_name)]
                )
            ]
            results["val_global_macro_task_accuracy"] = (
                sum(global_per_task) / len(global_per_task)
                if global_per_task
                else float("nan")
            )
    return results


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(args.amp and device.type == "cuda")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # Load the architecture before choosing dataset tuples and validation
    # geometry. A resumed v2 run must work without repeating its CLI mode.
    model, resume_checkpoint, init_report = create_model(args, device)
    initial_seed = args.seed
    restored_settings = configure_training_protocol(args, model, resume_checkpoint)
    snapshot_data_protocol(args, resume_checkpoint)
    amp_enabled = bool(args.amp and device.type == "cuda")
    if args.seed != initial_seed:
        seed_everything(args.seed)
    run_dir = prepare_run_directory(args)
    logger = build_logger(run_dir)
    is_blind = args.condition_mode in BLIND_MODES
    is_blind_v2 = args.condition_mode == "blind_v2"
    is_blind_v3 = args.condition_mode == "blind_v3"
    is_multiview_blind = args.condition_mode in MULTIVIEW_BLIND_MODES
    logger.info("Run directory: %s", run_dir)
    logger.info("Device: %s | AMP: %s", device, amp_enabled)
    if restored_settings:
        logger.info("Restored checkpoint training/protocol settings: %s", restored_settings)
    logger.info("Validation protocol: %s", args.val_protocol)
    logger.info("Dataset protocol: %s", args.data_protocol)
    if is_multiview_blind and args.val_inference == "crop":
        logger.warning(
            "Crop-only validation selected explicitly; this is an ablation protocol, "
            "not the default full-image multiview-blind selection protocol."
        )

    train_dataset = AllInOneDataset(
        args.data_root,
        patch_size=args.patch_size,
        train=True,
        manifest_path=args.train_manifest,
        noise_seed=args.seed,
        return_context=is_multiview_blind,
        context_size=args.routing_context_size,
    )
    batch_sampler = BalancedTaskBatchSampler(
        train_dataset,
        batch_size=args.batch_size,
        seed=args.seed,
        samples_per_epoch=args.steps_per_epoch * args.batch_size,
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    val_loader = None
    validation_root = args.val_root or (
        args.data_root if args.val_manifest else ""
    )
    if validation_root:
        val_dataset = AllInOneDataset(
            validation_root,
            patch_size=args.patch_size,
            train=False,
            manifest_path=args.val_manifest,
            noise_seed=args.seed,
            return_context=is_multiview_blind and args.val_inference != "tile",
            context_size=args.routing_context_size,
            full_image=args.val_inference != "crop",
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.val_protocol["batch_size"],
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=False,
            worker_init_fn=seed_worker,
            generator=generator,
        )
    else:
        logger.warning(
            "No validation set configured. Best-checkpoint selection is disabled."
        )

    ema = ModelEMA(model, decay=args.ema_decay)
    logger.info(
        "Model parameters: %d | config: %s",
        sum(parameter.numel() for parameter in model.parameters()),
        model.model_config,
    )
    if init_report is not None:
        logger.info(
            "Warm-started %d matching tensors from %s | new/missing tensors: %d | "
            "shape mismatches: %d",
            init_report["loaded_keys"],
            init_report["source"],
            len(init_report["missing_keys"]),
            len(init_report["incompatible_shapes"]),
        )
        if init_report.get("migration"):
            logger.info(
                "Checkpoint migration: %s (optimizer/scheduler/scaler start fresh)",
                init_report["migration"],
            )
    criterion = RestorationLoss(
        ssim_weight=args.ssim_weight,
        edge_weight=args.edge_weight,
        frequency_weight=args.frequency_weight,
    )
    optimizer = build_optimizer(model, args)
    logger.info(
        "Optimizer groups: %s",
        {
            group.get("group_name", "main"): {
                "lr": group["lr"],
                "parameters": sum(parameter.numel() for parameter in group["params"]),
            }
            for group in optimizer.param_groups
        },
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6,
    )
    scaler = GradScaler(
        enabled=amp_enabled,
        init_scale=args.amp_init_scale,
    )

    start_epoch = 1
    if resume_checkpoint is not None:
        if "optimizer_state_dict" in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in resume_checkpoint:
            scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])
            realign_cosine_scheduler(
                scheduler,
                optimizer,
                resume_checkpoint.get("epoch", 0),
                args.epochs,
            )
        if "scaler_state_dict" in resume_checkpoint:
            scaler.load_state_dict(resume_checkpoint["scaler_state_dict"])
        if "ema_state_dict" in resume_checkpoint:
            ema.load_state_dict(resume_checkpoint["ema_state_dict"])
        start_epoch = int(resume_checkpoint.get("epoch", 0)) + 1
        if "torch_rng_state" in resume_checkpoint:
            torch.set_rng_state(resume_checkpoint["torch_rng_state"].cpu())
        if "python_rng_state" in resume_checkpoint:
            random.setstate(resume_checkpoint["python_rng_state"])
        if "numpy_rng_state" in resume_checkpoint:
            np.random.set_state(resume_checkpoint["numpy_rng_state"])
        if "data_generator_state" in resume_checkpoint:
            generator.set_state(resume_checkpoint["data_generator_state"].cpu())
        if (
            device.type == "cuda"
            and resume_checkpoint.get("cuda_rng_state") is not None
        ):
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in resume_checkpoint["cuda_rng_state"]]
            )
        logger.info("Resuming from epoch %d", start_epoch)

    if start_epoch > args.epochs:
        raise ValueError(
            "Checkpoint already reached epoch {}, but --epochs is {}".format(
                start_epoch - 1, args.epochs
            )
        )

    metric_path = os.path.join(run_dir, "metrics.csv")
    fieldnames = [
        "epoch",
        "lr",
        "lr_backbone",
        "lr_condition",
        "lr_router",
        "train_loss",
        "train_restoration_loss",
        "train_task_cls_loss",
        "train_global_task_cls_loss",
        "train_local_task_cls_loss",
        "train_fused_task_cls_loss",
        "train_routing_consistency_loss",
        "train_feature_consistency_loss",
        "train_task_accuracy",
        "train_global_task_accuracy",
        "train_local_task_accuracy",
        "train_local_route_weight",
        "train_global_teacher_fraction",
        "train_amp_skipped_steps",
        "teacher_forcing_ratio",
        "train_derain_loss",
        "train_dehaze_loss",
        "train_denoise_loss",
        "train_derain_psnr",
        "train_dehaze_psnr",
        "train_denoise_psnr",
        "grad_norm",
        "val_derain_psnr",
        "val_dehaze_psnr",
        "val_denoise_psnr",
        "val_macro_psnr",
        "val_avg5_proxy_psnr",
        "val_derain_gain_mean",
        "val_derain_gain_p10",
        "val_derain_gain_min",
        "val_derain_negative_gain_count",
        "val_dehaze_gain_mean",
        "val_dehaze_gain_p10",
        "val_dehaze_gain_min",
        "val_dehaze_negative_gain_count",
        "val_denoise_gain_mean",
        "val_denoise_gain_p10",
        "val_denoise_gain_min",
        "val_denoise_negative_gain_count",
        "val_task_accuracy",
        "val_macro_task_accuracy",
        "val_derain_task_accuracy",
        "val_dehaze_task_accuracy",
        "val_denoise_task_accuracy",
        "val_task_confusion",
        "val_local_task_accuracy",
        "val_local_macro_task_accuracy",
        "val_local_derain_task_accuracy",
        "val_local_dehaze_task_accuracy",
        "val_local_denoise_task_accuracy",
        "val_local_task_confusion",
        "val_global_task_accuracy",
        "val_global_macro_task_accuracy",
        "val_global_derain_task_accuracy",
        "val_global_dehaze_task_accuracy",
        "val_global_denoise_task_accuracy",
        "val_global_task_confusion",
        "val_local_route_weight",
        "val_routing_entropy",
        "val_routing_ece",
        "val_local_global_js",
        "val_protocol",
    ]
    selection_metric_key = args.val_protocol["selection_metric"]
    best_validation = float("-inf")
    if resume_checkpoint is not None:
        saved_best = resume_checkpoint.get("best_validation")
        saved_metric_key = resume_checkpoint.get("best_validation_metric")
        legacy_macro_selection = (
            saved_metric_key is None
            and selection_metric_key == "val_macro_psnr"
        )
        if saved_metric_key not in (None, selection_metric_key):
            saved_best = None
        elif saved_metric_key is None and not legacy_macro_selection:
            saved_best = None
        if saved_best is None:
            saved_best = resume_checkpoint.get("metrics", {}).get(
                selection_metric_key
            )
        if saved_best not in (None, ""):
            best_validation = float(saved_best)
    task_weight_tensor = torch.tensor(
        args.task_weights,
        dtype=torch.float32,
        device=device,
    )

    metric_mode = (
        "a"
        if resume_checkpoint is not None and os.path.isfile(metric_path)
        else "w"
    )
    if metric_mode == "a":
        with open(metric_path, "r", newline="", encoding="utf-8") as existing_file:
            existing_header = next(csv.reader(existing_file), [])
        if existing_header:
            fieldnames = existing_header
    with open(
        metric_path,
        metric_mode,
        newline="",
        encoding="utf-8",
    ) as metric_file:
        writer = csv.DictWriter(
            metric_file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        if metric_mode == "w":
            writer.writeheader()

        for epoch in range(start_epoch, args.epochs + 1):
            batch_sampler.set_epoch(epoch)
            model.train()
            router_only = set_blind_trainable(
                model,
                epoch,
                args.router_warmup_epochs,
            )
            if is_blind and (epoch == start_epoch or epoch == args.router_warmup_epochs + 1):
                logger.info(
                    "Epoch %d trainable mode: %s",
                    epoch,
                    "router_only" if router_only else "full_finetune",
                )
            loss_meter = AverageMeter()
            restoration_loss_meter = AverageMeter()
            classification_loss_meter = AverageMeter()
            global_classification_meter = AverageMeter()
            local_classification_meter = AverageMeter()
            fused_classification_meter = AverageMeter()
            routing_consistency_meter = AverageMeter()
            feature_consistency_meter = AverageMeter()
            task_accuracy_meter = AverageMeter()
            global_task_accuracy_meter = AverageMeter()
            local_task_accuracy_meter = AverageMeter()
            local_route_weight_meter = AverageMeter()
            global_teacher_fraction_meter = AverageMeter()
            grad_meter = AverageMeter()
            amp_skipped_steps = 0
            task_loss = {task_id: AverageMeter() for task_id in TASK_NAMES}
            task_psnr = {task_id: AverageMeter() for task_id in TASK_NAMES}
            teacher_forcing_ratio = blind_teacher_forcing_ratio(
                args,
                epoch,
                condition_mode=model.model_config.get("condition_mode", "oracle"),
            )

            for step, batch in enumerate(
                train_loader,
                start=1,
            ):
                image_input, image_target, task_prompt, routing_context = unpack_dataset_batch(batch)
                image_input = image_input.to(device, non_blocking=True)
                image_target = image_target.to(device, non_blocking=True)
                task_prompt = task_prompt.to(device, non_blocking=True)
                if is_multiview_blind and routing_context is None:
                    raise RuntimeError(
                        "{} training requires a full-image routing context".format(
                            args.condition_mode
                        )
                    )
                if routing_context is not None:
                    routing_context = routing_context.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                with (
                    torch.autocast(device_type="cuda", dtype=torch.float16)
                    if amp_enabled and device.type == "cuda" else contextlib.nullcontext()
                ):
                    auxiliary = None
                    if is_blind:
                        extra_kwargs = (
                            {"routing_context": routing_context}
                            if is_multiview_blind
                            else {}
                        )
                        restored, auxiliary = model(
                            image_input,
                            task_prompt,
                            return_aux=True,
                            teacher_forcing=teacher_forcing_ratio,
                            **extra_kwargs,
                        )
                    else:
                        restored = model(image_input, task_prompt)
                    per_sample_loss, loss_parts = criterion(
                        restored,
                        image_target,
                    )
                    sample_weights = task_weight_tensor[task_prompt]
                    restoration_loss = (per_sample_loss * sample_weights).sum() / (
                        sample_weights.sum().clamp_min(1e-8)
                    )
                    classification_loss = restoration_loss.new_zeros(())
                    routing_losses = None
                    if is_blind_v2:
                        routing_losses = blind_v2_routing_loss(
                            auxiliary, task_prompt,
                            local_cls_weight=args.local_cls_weight,
                        )
                        classification_loss = routing_losses["combined_cls"]
                    elif is_blind_v3:
                        routing_losses = blind_v3_routing_loss(
                            auxiliary,
                            task_prompt,
                            global_cls_weight=args.global_cls_weight,
                            local_cls_weight=args.local_cls_weight,
                            fusion_cls_weight=args.fusion_cls_weight,
                        )
                        classification_loss = routing_losses["combined_cls"]
                    elif is_blind:
                        classification_loss = F.cross_entropy(
                            auxiliary["task_logits"].float(),
                            task_prompt,
                        )
                    loss = (
                        restoration_loss
                        + args.task_cls_weight * classification_loss
                    )
                    if routing_losses is not None:
                        loss = (
                            loss
                            + (
                                args.v3_consistency_weight
                                if is_blind_v3
                                else args.routing_consistency_weight
                            )
                            * routing_losses["probability_consistency"]
                            + args.feature_consistency_weight
                            * routing_losses["feature_consistency"]
                        )

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=args.grad_clip,
                )
                scale_before_step = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer_stepped = (
                    not amp_enabled or scaler.get_scale() >= scale_before_step
                )
                if optimizer_stepped:
                    ema.update(model)
                else:
                    amp_skipped_steps += 1

                batch_size = image_input.size(0)
                loss_meter.update(loss.item(), batch_size)
                restoration_loss_meter.update(restoration_loss.item(), batch_size)
                if is_blind:
                    classification_loss_meter.update(
                        classification_loss.item(),
                        batch_size,
                    )
                    predicted_tasks = auxiliary["task_logits"].argmax(dim=1)
                    task_accuracy_meter.update(
                        (predicted_tasks == task_prompt).float().mean().item(),
                        batch_size,
                    )
                    if auxiliary.get("global_task_logits") is not None:
                        global_predictions = auxiliary["global_task_logits"].argmax(dim=1)
                        global_task_accuracy_meter.update(
                            (global_predictions == task_prompt).float().mean().item(),
                            batch_size,
                        )
                    global_classification_meter.update(
                        (
                            routing_losses["global_cls"]
                            if routing_losses is not None else classification_loss
                        ).item(),
                        batch_size,
                    )
                if routing_losses is not None:
                    local_classification_meter.update(
                        routing_losses["local_cls"].item(), batch_size,
                    )
                    routing_consistency_meter.update(
                        routing_losses["probability_consistency"].item(), batch_size,
                    )
                    feature_consistency_meter.update(
                        routing_losses["feature_consistency"].item(), batch_size,
                    )
                    local_probabilities = auxiliary.get("local_task_probs")
                    local_predictions = (
                        local_probabilities.argmax(dim=1)
                        if local_probabilities is not None
                        else auxiliary["local_task_logits"].argmax(dim=1)
                    )
                    local_task_accuracy_meter.update(
                        (local_predictions == task_prompt).float().mean().item(), batch_size,
                    )
                    if routing_losses.get("fused_cls") is not None:
                        fused_classification_meter.update(
                            routing_losses["fused_cls"].item(), batch_size
                        )
                    if auxiliary.get("local_route_weight") is not None:
                        local_route_weight_meter.update(
                            auxiliary["local_route_weight"].float().mean().item(),
                            batch_size,
                        )
                    if routing_losses.get("global_teacher_fraction") is not None:
                        global_teacher_fraction_meter.update(
                            routing_losses["global_teacher_fraction"].item(), batch_size
                        )
                grad_meter.update(float(grad_norm), batch_size)
                values = psnr_per_image(
                    restored.detach().float().clamp(0, 1),
                    image_target.float(),
                )
                for task_id in TASK_NAMES:
                    selected = task_prompt == task_id
                    if selected.any():
                        for value in per_sample_loss[selected]:
                            task_loss[task_id].update(value.item())
                        for value in values[selected]:
                            task_psnr[task_id].update(value.item())

                if step == 1 or step % 100 == 0:
                    if is_multiview_blind:
                        logger.info(
                            "Epoch %d/%d step %d/%d loss %.5f rest %.5f "
                            "cls %.4f route_acc %.3f local_acc %.3f "
                            "route_cons %.5f feature_cons %.5f tf %.3f grad %.4f "
                            "amp_skip %d",
                            epoch, args.epochs, step, len(train_loader),
                            loss_meter.avg, restoration_loss_meter.avg,
                            classification_loss_meter.avg, task_accuracy_meter.avg,
                            local_task_accuracy_meter.avg, routing_consistency_meter.avg,
                            feature_consistency_meter.avg, teacher_forcing_ratio, grad_meter.avg,
                            amp_skipped_steps,
                        )
                    elif is_blind:
                        logger.info(
                            "Epoch %d/%d step %d/%d loss %.5f rest %.5f "
                            "cls %.4f acc %.3f tf %.3f grad %.4f",
                            epoch,
                            args.epochs,
                            step,
                            len(train_loader),
                            loss_meter.avg,
                            restoration_loss_meter.avg,
                            classification_loss_meter.avg,
                            task_accuracy_meter.avg,
                            teacher_forcing_ratio,
                            grad_meter.avg,
                        )
                    else:
                        logger.info(
                            "Epoch %d/%d step %d/%d loss %.5f grad %.4f",
                            epoch,
                            args.epochs,
                            step,
                            len(train_loader),
                            loss_meter.avg,
                            grad_meter.avg,
                        )

            scheduler.step()
            row = {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss": loss_meter.avg,
                "train_restoration_loss": restoration_loss_meter.avg,
                "train_task_cls_loss": classification_loss_meter.avg,
                "train_global_task_cls_loss": global_classification_meter.avg,
                "train_fused_task_cls_loss": fused_classification_meter.avg,
                "train_task_accuracy": task_accuracy_meter.avg,
                "train_global_task_accuracy": global_task_accuracy_meter.avg,
                "train_local_route_weight": local_route_weight_meter.avg,
                "train_global_teacher_fraction": global_teacher_fraction_meter.avg,
                "train_amp_skipped_steps": amp_skipped_steps,
                "teacher_forcing_ratio": teacher_forcing_ratio,
                "grad_norm": grad_meter.avg,
                "val_protocol": args.val_inference,
            }
            if is_multiview_blind:
                row.update({
                    "train_local_task_cls_loss": local_classification_meter.avg,
                    "train_local_task_accuracy": local_task_accuracy_meter.avg,
                    "train_routing_consistency_loss": routing_consistency_meter.avg,
                    "train_feature_consistency_loss": feature_consistency_meter.avg,
                })
            for group in optimizer.param_groups:
                group_name = group.get("group_name")
                if group_name:
                    row["lr_{}".format(group_name)] = group["lr"]
            for task_id, task_name in TASK_NAMES.items():
                row["train_{}_loss".format(task_name)] = task_loss[task_id].avg
                row["train_{}_psnr".format(task_name)] = task_psnr[task_id].avg
            validation = (
                validate(
                    ema.module, val_loader, device, amp_enabled,
                    inference_mode=args.val_inference,
                    tile_size=args.val_tile_size, overlap=args.val_overlap,
                )
                if val_loader
                else {}
            )
            row.update(validation)
            writer.writerow(row)
            metric_file.flush()

            selected_validation = validation.get(selection_metric_key)
            if validation and selected_validation is None:
                raise RuntimeError(
                    "Validation did not produce required selection metric {}. "
                    "Ensure all three restoration tasks are present."
                    .format(selection_metric_key)
                )
            is_best = bool(
                selected_validation is not None
                and math.isfinite(selected_validation)
                and selected_validation > best_validation
            )
            if is_best:
                best_validation = selected_validation
            checkpoint = build_training_checkpoint(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                args=args,
                metrics=row,
                ema_state_dict=ema.state_dict(),
                best_validation=best_validation,
                data_generator_state=generator.get_state(),
            )
            checkpoint["val_protocol"] = dict(args.val_protocol)
            checkpoint["data_protocol"] = dict(args.data_protocol)
            checkpoint["best_validation_metric"] = selection_metric_key
            atomic_torch_save(checkpoint, os.path.join(run_dir, "latest.pth"))
            if epoch % args.save_every == 0 or epoch == args.epochs:
                atomic_torch_save(
                    checkpoint,
                    os.path.join(run_dir, "epoch_{:04d}.pth".format(epoch)),
                )
            if is_best:
                atomic_torch_save(checkpoint, os.path.join(run_dir, "best.pth"))

            logger.info(
                "Epoch %d complete | loss %.5f | blind_acc %s | task PSNR %s | "
                "validation %s",
                epoch,
                loss_meter.avg,
                round(task_accuracy_meter.avg, 4) if is_blind else "n/a",
                {
                    TASK_NAMES[task_id]: round(task_psnr[task_id].avg, 3)
                    for task_id in TASK_NAMES
                },
                validation or "disabled",
            )


if __name__ == "__main__":
    main()
