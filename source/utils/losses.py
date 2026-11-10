import torch
import torch.nn as nn
import torch.nn.functional as F


def _mean_per_image(value):
    return value.flatten(1).mean(dim=1)


def blind_v2_routing_loss(auxiliary, task_labels, local_cls_weight=0.5):
    """Supervise both views and align local routing to a detached global view.

    Returns unweighted consistency terms. The global classifier is trained by
    its own cross-entropy, not by gradients from the consistency objectives.
    Continuous features are learned representations, not severity labels.
    """
    if local_cls_weight < 0:
        raise ValueError("local_cls_weight must be non-negative")
    global_logits = auxiliary["task_logits"]
    local_logits = auxiliary["local_task_logits"]
    global_feature = auxiliary["degradation_feature"]
    local_feature = auxiliary["local_degradation_feature"]
    if global_logits.shape != local_logits.shape or global_logits.ndim != 2:
        raise ValueError("Global/local routing logits must have identical BxT shapes")
    if global_feature.shape != local_feature.shape or global_feature.ndim != 2:
        raise ValueError("Global/local routing features must have identical BxD shapes")

    # Explicitly disable AMP: log probabilities and feature normalization need
    # fp32 even when this helper is called inside the restoration autocast.
    with torch.autocast(device_type=global_logits.device.type, enabled=False):
        global_logits = global_logits.float()
        local_logits = local_logits.float()
        global_cls = F.cross_entropy(global_logits, task_labels)
        local_cls = F.cross_entropy(local_logits, task_labels)
        global_log_probs = F.log_softmax(global_logits.detach(), dim=1)
        local_log_probs = F.log_softmax(local_logits, dim=1)
        probability_consistency = (
            local_log_probs.exp() * (local_log_probs - global_log_probs)
        ).sum(dim=1).mean()
        feature_consistency = F.mse_loss(
            F.normalize(local_feature.float(), dim=1, eps=1e-6),
            F.normalize(global_feature.detach().float(), dim=1, eps=1e-6),
        )
    return {
        "global_cls": global_cls,
        "local_cls": local_cls,
        "combined_cls": global_cls + float(local_cls_weight) * local_cls,
        "probability_consistency": probability_consistency,
        "feature_consistency": feature_consistency,
    }


def blind_v3_routing_loss(
    auxiliary,
    task_labels,
    global_cls_weight=0.5,
    local_cls_weight=0.5,
    fusion_cls_weight=1.0,
):
    """Supervise global, detail, and fused routes without a fixed weak teacher.

    V2 always detached the global view and forced the local view to imitate it.
    V3 instead selects the view assigning more probability to the known task as
    a per-sample teacher.  The weaker view receives the consistency gradient,
    so fine-detail evidence is not overwritten merely because it is local.
    """
    for name, value in (
        ("global_cls_weight", global_cls_weight),
        ("local_cls_weight", local_cls_weight),
        ("fusion_cls_weight", fusion_cls_weight),
    ):
        if value < 0:
            raise ValueError("{} must be non-negative".format(name))
    fused_logits = auxiliary["task_logits"]
    global_logits = auxiliary["global_task_logits"]
    local_logits = auxiliary["local_task_logits"]
    global_feature = auxiliary["degradation_feature"]
    local_feature = auxiliary["local_degradation_feature"]
    if not all(logits is not None for logits in (fused_logits, global_logits, local_logits)):
        raise ValueError("blind_v3 routing loss requires fused, global, and local logits")
    if not (fused_logits.shape == global_logits.shape == local_logits.shape):
        raise ValueError("Fused/global/local routing logits must have identical BxT shapes")
    if global_feature.shape != local_feature.shape or global_feature.ndim != 2:
        raise ValueError("Global/local routing features must have identical BxD shapes")

    with torch.autocast(device_type=fused_logits.device.type, enabled=False):
        fused_logits = fused_logits.float()
        global_logits = global_logits.float()
        local_logits = local_logits.float()
        global_cls = F.cross_entropy(global_logits, task_labels)
        local_cls = F.cross_entropy(local_logits, task_labels)
        fused_cls = F.cross_entropy(fused_logits, task_labels)
        global_probs = F.softmax(global_logits, dim=1)
        local_probs = F.softmax(local_logits, dim=1)
        label_index = task_labels[:, None]
        global_true = global_probs.gather(1, label_index)
        local_true = local_probs.gather(1, label_index)
        global_is_teacher = (global_true >= local_true).detach()
        teacher_probs = torch.where(
            global_is_teacher,
            global_probs.detach(),
            local_probs.detach(),
        )
        student_probs = torch.where(global_is_teacher, local_probs, global_probs)
        probability_consistency = (
            student_probs
            * (
                student_probs.clamp_min(1e-8).log()
                - teacher_probs.clamp_min(1e-8).log()
            )
        ).sum(dim=1).mean()

        normalized_global = F.normalize(global_feature.float(), dim=1, eps=1e-6)
        normalized_local = F.normalize(local_feature.float(), dim=1, eps=1e-6)
        teacher_feature = torch.where(
            global_is_teacher,
            normalized_global.detach(),
            normalized_local.detach(),
        )
        student_feature = torch.where(
            global_is_teacher,
            normalized_local,
            normalized_global,
        )
        feature_consistency = F.mse_loss(student_feature, teacher_feature)
    return {
        "global_cls": global_cls,
        "local_cls": local_cls,
        "fused_cls": fused_cls,
        "combined_cls": (
            float(global_cls_weight) * global_cls
            + float(local_cls_weight) * local_cls
            + float(fusion_cls_weight) * fused_cls
        ),
        "probability_consistency": probability_consistency,
        "feature_consistency": feature_consistency,
        "global_teacher_fraction": global_is_teacher.float().mean(),
    }


class RestorationLoss(nn.Module):
    """Charbonnier loss with structure, edge, and frequency constraints."""

    def __init__(
        self,
        ssim_weight=0.1,
        edge_weight=0.05,
        frequency_weight=0.05,
        epsilon=1e-3,
    ):
        super().__init__()
        self.ssim_weight = float(ssim_weight)
        self.edge_weight = float(edge_weight)
        self.frequency_weight = float(frequency_weight)
        self.epsilon = float(epsilon)

    def _charbonnier(self, restored, target):
        difference = restored - target
        return _mean_per_image(
            torch.sqrt(difference.square() + self.epsilon**2)
        )

    @staticmethod
    def _ssim(restored, target):
        # Local RGB SSIM. A 7x7 window is a good speed/quality compromise for
        # training; final benchmark SSIM remains computed by the test script.
        window = 7
        padding = window // 2
        mean_x = F.avg_pool2d(restored, window, 1, padding)
        mean_y = F.avg_pool2d(target, window, 1, padding)
        variance_x = F.avg_pool2d(restored.square(), window, 1, padding)
        variance_x = variance_x - mean_x.square()
        variance_y = F.avg_pool2d(target.square(), window, 1, padding)
        variance_y = variance_y - mean_y.square()
        covariance = F.avg_pool2d(restored * target, window, 1, padding)
        covariance = covariance - mean_x * mean_y
        c1 = 0.01**2
        c2 = 0.03**2
        numerator = (2 * mean_x * mean_y + c1) * (2 * covariance + c2)
        denominator = (
            (mean_x.square() + mean_y.square() + c1)
            * (variance_x + variance_y + c2)
        )
        return _mean_per_image(numerator / denominator.clamp_min(1e-8))

    def _edge(self, restored, target):
        restored_x = restored[:, :, :, 1:] - restored[:, :, :, :-1]
        target_x = target[:, :, :, 1:] - target[:, :, :, :-1]
        restored_y = restored[:, :, 1:, :] - restored[:, :, :-1, :]
        target_y = target[:, :, 1:, :] - target[:, :, :-1, :]
        loss_x = _mean_per_image(
            torch.sqrt((restored_x - target_x).square() + self.epsilon**2)
        )
        loss_y = _mean_per_image(
            torch.sqrt((restored_y - target_y).square() + self.epsilon**2)
        )
        return 0.5 * (loss_x + loss_y)

    @staticmethod
    def _frequency(restored, target):
        # FFT is explicitly evaluated in fp32 because cuFFT does not support
        # every half-precision image size.
        restored_fft = torch.fft.rfft2(
            restored.float(),
            norm="ortho",
        )
        target_fft = torch.fft.rfft2(
            target.float(),
            norm="ortho",
        )
        difference = torch.view_as_real(restored_fft - target_fft).abs()
        return _mean_per_image(difference)

    def forward(self, restored, target):
        charbonnier = self._charbonnier(restored, target)
        ssim_loss = 1.0 - self._ssim(restored.float(), target.float())
        edge_loss = self._edge(restored, target)
        if self.frequency_weight:
            frequency_loss = self._frequency(restored, target)
        else:
            frequency_loss = torch.zeros_like(charbonnier)
        total = (
            charbonnier
            + self.ssim_weight * ssim_loss
            + self.edge_weight * edge_loss
            + self.frequency_weight * frequency_loss
        )
        parts = {
            "charbonnier": charbonnier.detach(),
            "ssim": ssim_loss.detach(),
            "edge": edge_loss.detach(),
            "frequency": frequency_loss.detach(),
        }
        return total, parts
