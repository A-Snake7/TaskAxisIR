import contextlib
import math

import torch
import torch.nn.functional as F

from utils.modes import BLIND_MODES, MULTIVIEW_BLIND_MODES


def _autocast_context(device, enabled):
    if not enabled or device.type != "cuda":
        return contextlib.nullcontext()
    return torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
    )


def _tile_positions(length, tile_size, stride):
    if length <= tile_size:
        return [0]
    positions = list(range(0, length - tile_size + 1, stride))
    if positions[-1] != length - tile_size:
        positions.append(length - tile_size)
    return positions


def _blend_window(height, width, device):
    window_y = torch.hann_window(
        height,
        periodic=False,
        dtype=torch.float32,
        device=device,
    ).clamp_min(1e-3)
    window_x = torch.hann_window(
        width,
        periodic=False,
        dtype=torch.float32,
        device=device,
    ).clamp_min(1e-3)
    return torch.outer(window_y, window_x).view(1, 1, height, width)


def tiled_inference(
    model,
    image,
    task_prompt=None,
    tile_size=256,
    overlap=32,
    amp_enabled=True,
    teacher_forcing=0.0,
    global_blind_routing=True,
    return_routing=False,
):
    if image.ndim != 4 or image.size(0) != 1:
        raise ValueError("tiled_inference expects a single BCHW image")
    if tile_size <= 0 or overlap < 0 or overlap >= tile_size:
        raise ValueError("overlap must satisfy 0 <= overlap < tile_size")

    condition_mode = getattr(model, "condition_mode", "oracle")
    is_blind = condition_mode in BLIND_MODES
    is_v2 = condition_mode == "blind_v2"
    is_v3 = condition_mode == "blind_v3"
    is_multiview = condition_mode in MULTIVIEW_BLIND_MODES
    oracle_blind_routing = (
        is_blind and task_prompt is not None and teacher_forcing > 0.0
    )
    routing_probs_override = None
    condition_override = None
    routing_summary = None
    used_routing_summary = None
    global_routing_summary = None
    local_route_weight = None
    routing_aggregation = None
    routing_source_image = image
    if (
        global_blind_routing
        and is_blind
        and not is_v3
        and (not oracle_blind_routing or is_v2)
    ):
        with torch.inference_mode(), _autocast_context(
            image.device,
            amp_enabled,
        ):
            context = (
                model.predict_task_context(
                    image, task_prompt=task_prompt, teacher_forcing=teacher_forcing
                )
                if is_v2 and oracle_blind_routing
                else model.predict_task_context(image)
            )
            routing_probs_override = context.get("routing_probs", context["task_probs"])
            condition_override = context["condition"]
            routing_summary = context["task_probs"]
            used_routing_summary = routing_probs_override
            global_routing_summary = context["task_probs"]
    elif oracle_blind_routing:
        used_routing_summary = F.one_hot(
            task_prompt,
            num_classes=getattr(model, "num_tasks", 3),
        ).float()
        if not is_v2:
            routing_summary = used_routing_summary

    _, _, original_height, original_width = image.shape
    pad_height = max(0, tile_size - original_height)
    pad_width = max(0, tile_size - original_width)
    if pad_height or pad_width:
        padding_mode = (
            "reflect"
            if original_height > pad_height and original_width > pad_width
            else "replicate"
        )
        image = F.pad(
            image,
            (0, pad_width, 0, pad_height),
            mode=padding_mode,
        )

    _, _, height, width = image.shape
    stride = tile_size - overlap
    y_positions = _tile_positions(height, tile_size, stride)
    x_positions = _tile_positions(width, tile_size, stride)
    output_sum = torch.zeros_like(image, dtype=torch.float32)
    weight_sum = torch.zeros(
        (1, 1, height, width),
        dtype=torch.float32,
        device=image.device,
    )
    window = _blend_window(tile_size, tile_size, image.device)
    local_routing_sum = None
    local_routing_count = 0
    used_local_routing_sum = None
    local_condition_probs_sum = None
    local_condition_count = 0
    local_global_js_sum = None

    # V3 performs a cheap routing-only pass over the same tiles. Each tile is
    # fused with the global view exactly as a training crop is, then the fused
    # probabilities are coverage-weighted into one image-level head route.
    # That route and the global PromptAffine condition are cached for every
    # restoration tile, avoiding both train/inference gate drift and seams.
    if global_blind_routing and is_v3:
        with torch.inference_mode(), _autocast_context(image.device, amp_enabled):
            global_context = model.predict_global_task_context(
                routing_source_image,
                task_prompt=task_prompt if oracle_blind_routing else None,
                teacher_forcing=teacher_forcing if oracle_blind_routing else 0.0,
            )
            fused_probs_sum = None
            detail_probs_sum = None
            route_weight_sum = image.new_zeros((), dtype=torch.float32)
            route_js_sum = image.new_zeros((), dtype=torch.float32)
            detail_weight_sum = image.new_zeros((), dtype=torch.float32)
            routing_window_sum = torch.zeros(
                (1, 1, original_height, original_width),
                dtype=torch.float32,
                device=image.device,
            )
            for top in y_positions:
                for left in x_positions:
                    valid_height = max(0, min(tile_size, original_height - top))
                    valid_width = max(0, min(tile_size, original_width - left))
                    if not valid_height or not valid_width:
                        continue
                    routing_window_sum[
                        :, :, top : top + valid_height, left : left + valid_width
                    ] += window[:, :, :valid_height, :valid_width]
            for top in y_positions:
                for left in x_positions:
                    valid_height = max(0, min(tile_size, original_height - top))
                    valid_width = max(0, min(tile_size, original_width - left))
                    if not valid_height or not valid_width:
                        continue
                    # Routing never sees the synthetic padding used only to
                    # satisfy the restoration tile size.
                    tile = routing_source_image[
                        :, :, top : top + valid_height, left : left + valid_width
                    ]
                    valid_window = window[:, :, :valid_height, :valid_width]
                    tile_weight = (
                        valid_window
                        / routing_window_sum[
                            :, :, top : top + valid_height, left : left + valid_width
                        ].clamp_min(1e-8)
                    ).sum()
                    detail_context = model.predict_local_task_context(tile)
                    tile_fusion = model.fuse_task_context(
                        global_context,
                        detail_context["task_logits"],
                        detail_context["degradation_feature"],
                    )
                    if global_routing_summary is None:
                        global_routing_summary = tile_fusion["global_task_probs"]
                    fused_probs_sum = (
                        tile_fusion["task_probs"] * tile_weight
                        if fused_probs_sum is None
                        else fused_probs_sum + tile_fusion["task_probs"] * tile_weight
                    )
                    detail_probs_sum = (
                        tile_fusion["local_task_probs"] * tile_weight
                        if detail_probs_sum is None
                        else detail_probs_sum
                        + tile_fusion["local_task_probs"] * tile_weight
                    )
                    route_weight_sum += tile_fusion["local_route_weight"].mean() * tile_weight
                    route_js_sum += tile_fusion["local_global_js"].mean() * tile_weight
                    detail_weight_sum += tile_weight
            routing_summary = fused_probs_sum / detail_weight_sum.clamp_min(1e-8)
            routing_summary = routing_summary / routing_summary.sum(
                dim=1, keepdim=True
            ).clamp_min(1e-8)
            routing_probs_override = routing_summary
            if oracle_blind_routing:
                oracle_probs = F.one_hot(
                    task_prompt,
                    num_classes=getattr(model, "num_tasks", routing_summary.size(1)),
                ).float()
                routing_probs_override = (
                    float(teacher_forcing) * oracle_probs
                    + (1.0 - float(teacher_forcing)) * routing_summary
                )
            condition_override = model.build_v3_condition(
                global_context, routing_probs_override
            )
            used_routing_summary = routing_probs_override
            local_condition_probs_sum = (
                detail_probs_sum / detail_weight_sum.clamp_min(1e-8)
            )
            local_condition_count = 1
            local_global_js_sum = route_js_sum / detail_weight_sum.clamp_min(1e-8)
            local_route_weight = route_weight_sum / detail_weight_sum.clamp_min(1e-8)
            routing_aggregation = "coverage_weighted_per_tile_fusion"

    with torch.inference_mode():
        for top in y_positions:
            for left in x_positions:
                tile = image[:, :, top : top + tile_size, left : left + tile_size]
                with _autocast_context(image.device, amp_enabled):
                    need_local_routing = (
                        return_routing
                        and is_blind
                        and (
                            is_v2
                            or (routing_probs_override is None and not oracle_blind_routing)
                        )
                    )
                    model_output = model(
                        tile,
                        task_prompt,
                        return_aux=need_local_routing,
                        teacher_forcing=teacher_forcing,
                        routing_probs_override=routing_probs_override,
                        condition_override=condition_override,
                    )
                    if need_local_routing:
                        restored, auxiliary = model_output
                        if auxiliary.get("task_probs") is not None:
                            tile_probs = auxiliary["task_probs"].float()
                            local_routing_sum = (
                                tile_probs if local_routing_sum is None
                                else local_routing_sum + tile_probs
                            )
                            used_probs = auxiliary.get("routing_probs", tile_probs).float()
                            used_local_routing_sum = (
                                used_probs if used_local_routing_sum is None
                                else used_local_routing_sum + used_probs
                            )
                            local_routing_count += 1
                        if is_multiview and auxiliary.get("local_task_logits") is not None:
                            local_probs = auxiliary.get("local_task_probs")
                            if local_probs is None:
                                local_probs = auxiliary["local_task_logits"].float().softmax(dim=1)
                            else:
                                local_probs = local_probs.float()
                            reference_probs = (
                                global_routing_summary
                                if global_routing_summary is not None
                                else auxiliary.get("global_task_probs")
                            )
                            reference_probs = (
                                reference_probs.float()
                                if reference_probs is not None
                                else auxiliary["task_probs"].float()
                            )
                            mixture = 0.5 * (local_probs + reference_probs)
                            js = 0.5 * (
                                (local_probs * (local_probs.clamp_min(1e-8).log() - mixture.clamp_min(1e-8).log())).sum(dim=1)
                                + (reference_probs * (reference_probs.clamp_min(1e-8).log() - mixture.clamp_min(1e-8).log())).sum(dim=1)
                            ).mean()
                            local_global_js_sum = (
                                js if local_global_js_sum is None else local_global_js_sum + js
                            )
                            local_condition_probs_sum = (
                                local_probs if local_condition_probs_sum is None
                                else local_condition_probs_sum + local_probs
                            )
                            local_condition_count += 1
                    else:
                        restored = model_output
                output_sum[
                    :, :, top : top + tile_size, left : left + tile_size
                ] += restored.float() * window
                weight_sum[
                    :, :, top : top + tile_size, left : left + tile_size
                ] += window

    restored = output_sum / weight_sum.clamp_min(1e-8)
    restored = restored[:, :, :original_height, :original_width]
    if not return_routing:
        return restored
    if routing_summary is None and local_routing_sum is not None:
        routing_summary = local_routing_sum / max(local_routing_count, 1)
    if used_routing_summary is None and used_local_routing_sum is not None:
        used_routing_summary = used_local_routing_sum / max(local_routing_count, 1)
    return restored, {
        "task_probs": routing_summary,
        "routing_probs": used_routing_summary,
        "local_task_probs": (
            local_condition_probs_sum / local_condition_count
            if local_condition_count else None
        ),
        "local_global_js": (
            local_global_js_sum / local_condition_count
            if local_condition_count else None
        ),
        "global_task_probs": global_routing_summary,
        "local_route_weight": local_route_weight,
        "routing_aggregation": routing_aggregation,
        "global_routing": routing_probs_override is not None,
        "tile_count": len(y_positions) * len(x_positions),
    }


def center_crop_pair_to_common(image, target):
    if image.ndim != 4 or target.ndim != 4:
        raise ValueError("Expected BCHW tensors")
    common_height = min(image.shape[-2], target.shape[-2])
    common_width = min(image.shape[-1], target.shape[-1])

    def crop(tensor):
        top = (tensor.shape[-2] - common_height) // 2
        left = (tensor.shape[-1] - common_width) // 2
        return tensor[
            :,
            :,
            top : top + common_height,
            left : left + common_width,
        ]

    return crop(image), crop(target)
