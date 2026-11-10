import math

import torch
import torch.nn.functional as F


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, value, count=1):
        value = float(value)
        self.val = value
        self.sum += value * count
        self.count += count
        self.avg = self.sum / self.count if self.count else 0.0


def psnr_per_image(image, target, max_val=1.0):
    if image.shape != target.shape:
        raise ValueError(
            "PSNR inputs must have the same shape: {} != {}".format(
                tuple(image.shape), tuple(target.shape)
            )
        )
    reduce_dims = tuple(range(1, image.ndim))
    mse = torch.mean((image.detach() - target.detach()) ** 2, dim=reduce_dims)
    max_tensor = torch.as_tensor(max_val, dtype=mse.dtype, device=mse.device)
    return 20 * torch.log10(max_tensor) - 10 * torch.log10(
        mse.clamp_min(torch.finfo(mse.dtype).eps)
    )


def calc_psnr(image, target, max_val=1.0):
    return psnr_per_image(image, target, max_val=max_val).mean().item()


_WINDOW_CACHE = {}


def _gaussian_window(window_size, channels, device, dtype):
    key = (window_size, channels, str(device), dtype)
    if key in _WINDOW_CACHE:
        return _WINDOW_CACHE[key]
    coordinates = torch.arange(window_size, device=device, dtype=dtype)
    coordinates = coordinates - window_size // 2
    kernel_1d = torch.exp(-(coordinates**2) / (2 * 1.5**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    window = kernel_2d.expand(channels, 1, window_size, window_size).contiguous()
    _WINDOW_CACHE[key] = window
    return window


def ssim_per_image(image, target, window_size=11, max_val=1.0):
    if image.shape != target.shape:
        raise ValueError(
            "SSIM inputs must have the same shape: {} != {}".format(
                tuple(image.shape), tuple(target.shape)
            )
        )
    if image.ndim != 4:
        raise ValueError("SSIM expects BCHW tensors")
    _, channels, height, width = image.shape
    if min(height, width) < window_size:
        raise ValueError("Images are smaller than the SSIM window")

    image = image.detach()
    target = target.detach()
    window = _gaussian_window(
        window_size,
        channels,
        image.device,
        image.dtype,
    )
    padding = window_size // 2
    image_pad = F.pad(image, [padding] * 4, mode="reflect")
    target_pad = F.pad(target, [padding] * 4, mode="reflect")

    mean_image = F.conv2d(image_pad, window, groups=channels)
    mean_target = F.conv2d(target_pad, window, groups=channels)
    mean_image_sq = mean_image.square()
    mean_target_sq = mean_target.square()
    mean_product = mean_image * mean_target

    variance_image = (
        F.conv2d(image_pad * image_pad, window, groups=channels) - mean_image_sq
    ).clamp_min(0)
    variance_target = (
        F.conv2d(target_pad * target_pad, window, groups=channels) - mean_target_sq
    ).clamp_min(0)
    covariance = (
        F.conv2d(image_pad * target_pad, window, groups=channels) - mean_product
    )

    constant_1 = (0.01 * max_val) ** 2
    constant_2 = (0.03 * max_val) ** 2
    ssim_map = (
        (2 * mean_product + constant_1) * (2 * covariance + constant_2)
    ) / (
        (mean_image_sq + mean_target_sq + constant_1)
        * (variance_image + variance_target + constant_2)
    )
    return ssim_map.mean(dim=(1, 2, 3))


def calc_ssim(image, target, window_size=11, max_val=1.0):
    return (
        ssim_per_image(
            image,
            target,
            window_size=window_size,
            max_val=max_val,
        )
        .mean()
        .item()
    )
