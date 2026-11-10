"""Shared preprocessing for blind-v2 training and image-level routing."""

import torch.nn.functional as F


def prepare_routing_context(image, context_size=256):
    """Downsample a degraded BCHW image and pad it to a fixed square.

    Aspect ratio is preserved and small images are never enlarged. Padding is
    on the right/bottom, using reflection where valid and replication otherwise.
    Both the dataset and inference call this function, so the router sees the
    same preprocessing. The input must already contain any synthetic noise.
    """
    if image.ndim != 4 or not image.is_floating_point():
        raise ValueError("Routing context requires a floating-point BCHW tensor")
    context_size = int(context_size)
    if context_size <= 0:
        raise ValueError("context_size must be positive")
    height, width = image.shape[-2:]
    if min(height, width) <= 0:
        raise ValueError("Routing context cannot contain an empty image")
    if max(height, width) > context_size:
        scale = context_size / float(max(height, width))
        image = F.interpolate(
            image.float(),
            size=(max(1, round(height * scale)), max(1, round(width * scale))),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    height, width = image.shape[-2:]
    pad_height = context_size - height
    pad_width = context_size - width
    if pad_height or pad_width:
        mode = (
            "reflect"
            if height > pad_height and width > pad_width
            else "replicate"
        )
        image = F.pad(image, (0, pad_width, 0, pad_height), mode=mode)
    return image
