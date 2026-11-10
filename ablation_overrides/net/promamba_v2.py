import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from utils.context import prepare_routing_context
from utils.modes import BLIND_MODES, MULTIVIEW_BLIND_MODES

try:
    from mamba_ssm import Mamba
except ImportError as error:
    raise ImportError(
        "ProMamba requires mamba-ssm and causal-conv1d. "
        "Activate the airnet environment before running this project."
    ) from error


def _group_norm(channels):
    groups = min(8, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class DegradationEncoder(nn.Module):
    """Infer a compact degradation representation from shallow features."""

    def __init__(self, in_channels, cond_dim, num_tasks):
        super().__init__()
        hidden_channels = in_channels * 2
        self.extractor = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=in_channels,
            ),
            _group_norm(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            _group_norm(hidden_channels),
            nn.SiLU(),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=hidden_channels,
            ),
            _group_norm(hidden_channels),
            nn.SiLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embedding = nn.Sequential(
            nn.Linear(hidden_channels, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.classifier = nn.Linear(cond_dim, num_tasks)

    def forward(self, feature):
        pooled = self.pool(self.extractor(feature)).flatten(1)
        degradation_feature = self.embedding(pooled)
        task_logits = self.classifier(degradation_feature)
        return degradation_feature, task_logits


class ReliabilityAwareRoutingFusion(nn.Module):
    """Fuse global and detail-view routing without trusting logit magnitude.

    The v2 global router can be confidently wrong after fine rain streaks have
    been reduced to a small context image.  Mixing calibrated probabilities,
    rather than raw logits, prevents one over-confident view from dominating
    solely because its classifier has a larger logit scale.  A small gate uses
    both probability reliability statistics and normalized degradation
    features to learn which view is more useful for each image.
    """

    def __init__(self, cond_dim, num_tasks, local_prior=0.5, route_mode="learned"):
        super().__init__()
        if route_mode not in ("learned", "equal", "global", "local"):
            raise ValueError("Unsupported V3 route mode: {}".format(route_mode))
        self.route_mode = route_mode
        local_prior = float(local_prior)
        if not 0.0 < local_prior < 1.0:
            raise ValueError("v3_local_route_prior must be strictly between 0 and 1")
        self.num_tasks = int(num_tasks)
        self.log_temperatures = nn.Parameter(torch.zeros(2))
        statistic_dim = 3 * self.num_tasks + 7 + 3 * int(cond_dim)
        self.gate = nn.Sequential(
            nn.Linear(statistic_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, 1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(
            self.gate[-1].bias,
            math.log(local_prior / (1.0 - local_prior)),
        )

    @staticmethod
    def _margin(probabilities):
        if probabilities.size(1) == 1:
            return probabilities[:, :1]
        top_two = probabilities.topk(2, dim=1).values
        return top_two[:, :1] - top_two[:, 1:2]

    def forward(self, global_logits, local_logits, global_feature, local_feature):
        if global_logits.shape != local_logits.shape or global_logits.ndim != 2:
            raise ValueError("Global/local logits must have identical BxT shapes")
        if global_logits.size(1) != self.num_tasks:
            raise ValueError("Routing logits do not match the configured task count")
        if global_feature.shape != local_feature.shape or global_feature.ndim != 2:
            raise ValueError("Global/local features must have identical BxD shapes")

        # Probability calibration and entropy calculations are intentionally
        # kept in FP32 even when restoration runs under autocast.
        with torch.autocast(device_type=global_logits.device.type, enabled=False):
            temperatures = self.log_temperatures.float().exp().clamp(0.25, 4.0)
            global_probs = F.softmax(global_logits.float() / temperatures[0], dim=1)
            local_probs = F.softmax(local_logits.float() / temperatures[1], dim=1)
            entropy_scale = max(math.log(max(self.num_tasks, 2)), 1e-8)
            global_entropy = -(
                global_probs * global_probs.clamp_min(1e-8).log()
            ).sum(dim=1, keepdim=True) / entropy_scale
            local_entropy = -(
                local_probs * local_probs.clamp_min(1e-8).log()
            ).sum(dim=1, keepdim=True) / entropy_scale
            midpoint = 0.5 * (global_probs + local_probs)
            midpoint_log = midpoint.clamp_min(1e-8).log()
            js_divergence = 0.5 * (
                (
                    global_probs
                    * (global_probs.clamp_min(1e-8).log() - midpoint_log)
                ).sum(dim=1, keepdim=True)
                + (
                    local_probs
                    * (local_probs.clamp_min(1e-8).log() - midpoint_log)
                ).sum(dim=1, keepdim=True)
            )
            normalized_global = F.normalize(global_feature.float(), dim=1, eps=1e-6)
            normalized_local = F.normalize(local_feature.float(), dim=1, eps=1e-6)
            statistics = torch.cat(
                (
                    global_probs,
                    local_probs,
                    (global_probs - local_probs).abs(),
                    global_entropy,
                    local_entropy,
                    global_probs.max(dim=1, keepdim=True).values,
                    local_probs.max(dim=1, keepdim=True).values,
                    self._margin(global_probs),
                    self._margin(local_probs),
                    js_divergence,
                    normalized_global,
                    normalized_local,
                    (normalized_global - normalized_local).abs(),
                ),
                dim=1,
            )
            # Keep the same parameter allocation/initialization in every mode,
            # so shared backbone initialization does not shift with RNG usage.
            # The gate is stored but unused (no gradient) in fixed-route modes.
            if self.route_mode == "learned":
                local_weight = torch.sigmoid(self.gate(statistics))
            else:
                alpha = {"equal": 0.5, "global": 0.0, "local": 1.0}[self.route_mode]
                local_weight = global_probs.new_full((global_probs.size(0), 1), alpha)
            fused_probs = (
                (1.0 - local_weight) * global_probs
                + local_weight * local_probs
            )
            fused_probs = fused_probs / fused_probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
            fused_logits = fused_probs.clamp_min(1e-8).log()
        return {
            "task_logits": fused_logits,
            "task_probs": fused_probs,
            "global_task_probs": global_probs,
            "local_task_probs": local_probs,
            "local_route_weight": local_weight,
            "local_global_js": js_divergence.squeeze(1),
        }


class PromptAffine(nn.Module):
    def __init__(self, cond_dim, channels):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(cond_dim, channels),
            nn.SiLU(),
            nn.Linear(channels, channels * 2),
        )
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, feature, condition):
        gamma, beta = self.proj(condition).chunk(2, dim=1)
        return (
            feature * (1.0 + gamma[:, :, None, None])
            + beta[:, :, None, None]
        )


class PromptLocalFrequencyBlock(nn.Module):
    """Efficient prompt-conditioned local and frequency-decomposition block."""

    def __init__(self, channels, cond_dim, use_frequency_branch=True):
        super().__init__()
        self.use_frequency_branch = bool(use_frequency_branch)
        self.affine = PromptAffine(cond_dim, channels)
        self.local = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
            ),
            _group_norm(channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=1),
        )
        if self.use_frequency_branch:
            self.frequency = nn.Sequential(
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=3,
                    padding=1,
                    groups=channels,
                ),
                nn.SiLU(),
                nn.Conv2d(channels, channels, kernel_size=1),
            )
            self.frequency_gate = nn.Linear(cond_dim, channels * 2)
            self.branch_gate = nn.Linear(cond_dim, channels * 2)
        self.output = nn.Conv2d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, feature, condition, gate_condition=None):
        gate_condition = condition if gate_condition is None else gate_condition
        modulated = self.affine(feature, condition)
        local_feature = self.local(modulated)

        if not self.use_frequency_branch:
            return feature + self.output(local_feature)

        # A differentiable low/high-frequency decomposition. It is much cheaper
        # than running FFTs at every high-resolution encoder stage.
        low_frequency = F.avg_pool2d(
            modulated,
            kernel_size=5,
            stride=1,
            padding=2,
        )
        high_frequency = modulated - low_frequency
        frequency_weights = self.frequency_gate(gate_condition)
        frequency_weights = rearrange(
            frequency_weights,
            "b (n c) -> b n c 1 1",
            n=2,
        ).softmax(dim=1)
        frequency_feature = self.frequency(
            frequency_weights[:, 0] * low_frequency
            + frequency_weights[:, 1] * high_frequency
        )

        branch_weights = self.branch_gate(gate_condition)
        branch_weights = rearrange(
            branch_weights,
            "b (n c) -> b n c 1 1",
            n=2,
        ).softmax(dim=1)
        fused = (
            branch_weights[:, 0] * local_feature
            + branch_weights[:, 1] * frequency_feature
        )
        return feature + self.output(fused)


class FourDirectionMambaBlock(nn.Module):
    """Prompt-conditioned row/column scans without flattened-image bias."""

    def __init__(
        self,
        channels,
        cond_dim,
        d_state=16,
        scan_directions=4,
        direction_fusion="gated",
        local_global_fusion="gated",
        use_frequency_branch=True,
        bottleneck_type="mamba",
        batch_reverse_scans=False,
    ):
        super().__init__()
        if scan_directions not in (1, 2, 4):
            raise ValueError("scan_directions must be 1, 2, or 4")
        if direction_fusion not in ("gated", "mean"):
            raise ValueError("Unsupported direction_fusion: {}".format(direction_fusion))
        if local_global_fusion not in (
            "gated",
            "mean",
            "global_only",
            "local_only",
        ):
            raise ValueError(
                "Unsupported local_global_fusion: {}".format(local_global_fusion)
            )
        if bottleneck_type not in ("mamba", "conv"):
            raise ValueError("Unsupported bottleneck_type: {}".format(bottleneck_type))
        self.scan_directions = int(scan_directions)
        self.direction_fusion = direction_fusion
        self.local_global_fusion = local_global_fusion
        self.bottleneck_type = bottleneck_type
        self.batch_reverse_scans = bool(batch_reverse_scans)
        self.affine = PromptAffine(cond_dim, channels)
        if self.bottleneck_type == "mamba":
            self.norm_h = nn.LayerNorm(channels)
            self.mamba_h = Mamba(d_model=channels, d_state=d_state, expand=2)
            if self.scan_directions == 4:
                self.norm_v = nn.LayerNorm(channels)
                self.mamba_v = Mamba(d_model=channels, d_state=d_state, expand=2)
            if self.direction_fusion == "gated" and self.scan_directions > 1:
                self.direction_gate = nn.Linear(
                    cond_dim,
                    channels * self.scan_directions,
                )
        else:
            self.conv_global = nn.Sequential(
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=5,
                    padding=2,
                    groups=channels,
                ),
                _group_norm(channels),
                nn.SiLU(),
                nn.Conv2d(channels, channels, kernel_size=1),
            )
        if self.local_global_fusion != "global_only":
            self.local_frequency = PromptLocalFrequencyBlock(
                channels,
                cond_dim,
                use_frequency_branch=use_frequency_branch,
            )
        if self.local_global_fusion == "gated":
            self.global_gate = nn.Linear(cond_dim, channels * 2)
        self.output = nn.Conv2d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, feature, condition, gate_condition=None):
        gate_condition = condition if gate_condition is None else gate_condition
        modulated = self.affine(feature, condition)
        if self.local_global_fusion != "local_only":
            if self.bottleneck_type == "mamba":
                batch, channels, height, width = modulated.shape
                horizontal = rearrange(modulated, "b c h w -> (b h) w c")
                if self.scan_directions >= 2:
                    if self.batch_reverse_scans and max(height, width) <= 64:
                        horizontal_pair = torch.cat(
                            (horizontal, torch.flip(horizontal, dims=(1,))),
                            dim=0,
                        )
                        horizontal_pair = self.mamba_h(
                            self.norm_h(horizontal_pair)
                        )
                        horizontal_forward, horizontal_reverse = horizontal_pair.chunk(
                            2,
                            dim=0,
                        )
                        horizontal_reverse = torch.flip(
                            horizontal_reverse,
                            dims=(1,),
                        )
                    else:
                        horizontal_forward = self.mamba_h(self.norm_h(horizontal))
                        horizontal_reverse = torch.flip(
                            self.mamba_h(
                                self.norm_h(torch.flip(horizontal, dims=(1,)))
                            ),
                            dims=(1,),
                        )
                else:
                    horizontal_forward = self.mamba_h(self.norm_h(horizontal))
                    horizontal_reverse = None
                directional_features = [
                    rearrange(
                        horizontal_forward,
                        "(b h) w c -> b c h w",
                        b=batch,
                        h=height,
                    )
                ]
                if horizontal_reverse is not None:
                    directional_features.append(
                        rearrange(
                            horizontal_reverse,
                            "(b h) w c -> b c h w",
                            b=batch,
                            h=height,
                        )
                    )
                if self.scan_directions == 4:
                    vertical = rearrange(modulated, "b c h w -> (b w) h c")
                    if self.batch_reverse_scans and max(height, width) <= 64:
                        vertical_pair = torch.cat(
                            (vertical, torch.flip(vertical, dims=(1,))),
                            dim=0,
                        )
                        vertical_pair = self.mamba_v(self.norm_v(vertical_pair))
                        vertical_forward, vertical_reverse = vertical_pair.chunk(
                            2,
                            dim=0,
                        )
                        vertical_reverse = torch.flip(vertical_reverse, dims=(1,))
                    else:
                        vertical_forward = self.mamba_v(self.norm_v(vertical))
                        vertical_reverse = torch.flip(
                            self.mamba_v(
                                self.norm_v(torch.flip(vertical, dims=(1,)))
                            ),
                            dims=(1,),
                        )
                    directional_features.extend(
                        [
                            rearrange(
                                vertical_forward,
                                "(b w) h c -> b c h w",
                                b=batch,
                                w=width,
                            ),
                            rearrange(
                                vertical_reverse,
                                "(b w) h c -> b c h w",
                                b=batch,
                                w=width,
                            ),
                        ]
                    )
                if self.direction_fusion == "gated" and len(directional_features) > 1:
                    direction_weights = self.direction_gate(gate_condition)
                    direction_weights = rearrange(
                        direction_weights,
                        "b (n c) -> b n c 1 1",
                        n=len(directional_features),
                    ).softmax(dim=1)
                    global_feature = sum(
                        direction_weights[:, index] * directional_feature
                        for index, directional_feature in enumerate(
                            directional_features
                        )
                    )
                else:
                    global_feature = torch.stack(
                        directional_features,
                        dim=1,
                    ).mean(dim=1)
            else:
                global_feature = self.conv_global(modulated)

        if self.local_global_fusion != "global_only":
            local_feature = self.local_frequency(
                feature, condition, gate_condition=gate_condition
            ) - feature

        if self.local_global_fusion == "global_only":
            fused = global_feature
        elif self.local_global_fusion == "local_only":
            fused = local_feature
        elif self.local_global_fusion == "mean":
            fused = 0.5 * (global_feature + local_feature)
        else:
            global_weights = self.global_gate(gate_condition)
            global_weights = rearrange(
                global_weights,
                "b (n c) -> b n c 1 1",
                n=2,
            ).softmax(dim=1)
            fused = (
                global_weights[:, 0] * global_feature
                + global_weights[:, 1] * local_feature
            )
        return feature + self.output(fused)


class HierarchicalProMamba(nn.Module):
    """Three-scale ProMamba with prompt injection at every resolution."""

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        dim=48,
        stage_blocks=(2, 3, 4),
        num_tasks=3,
        d_state=16,
        scan_directions=4,
        direction_fusion="gated",
        local_global_fusion="gated",
        use_frequency_branch=True,
        use_task_condition=True,
        shared_head=False,
        bottleneck_type="mamba",
        condition_mode="oracle",
        batch_reverse_scans=False,
        routing_context_size=256,
        local_condition_scale=0.1,
        v3_local_route_prior=0.5,
        v3_fuse_prompt_condition=False,
        v3_route_mode="learned",
    ):
        super().__init__()
        if len(stage_blocks) != 3:
            raise ValueError("stage_blocks must contain three integers")
        if condition_mode not in ("oracle",) + BLIND_MODES:
            raise ValueError("Unsupported condition_mode: {}".format(condition_mode))
        if int(routing_context_size) <= 0:
            raise ValueError("routing_context_size must be positive")
        if not 0.0 <= float(local_condition_scale) <= 1.0:
            raise ValueError("local_condition_scale must be between 0 and 1")
        dimensions = (dim, dim * 2, dim * 4)
        self.num_tasks = int(num_tasks)
        self.use_task_condition = bool(use_task_condition)
        self.shared_head = bool(shared_head)
        self.condition_mode = condition_mode
        self.routing_context_size = int(routing_context_size)
        self.local_condition_scale = float(local_condition_scale)
        self.v3_local_route_prior = float(v3_local_route_prior)
        self.v3_fuse_prompt_condition = bool(v3_fuse_prompt_condition)
        self.task_embed = nn.Embedding(num_tasks, dim)
        self.stem = nn.Conv2d(in_channels, dimensions[0], 3, padding=1)
        if self.condition_mode in BLIND_MODES:
            self.degradation_encoder = DegradationEncoder(
                dimensions[0],
                dim,
                num_tasks,
            )
            self.degradation_condition = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim),
            )
            nn.init.zeros_(self.degradation_condition[-1].weight)
            nn.init.zeros_(self.degradation_condition[-1].bias)
        if self.condition_mode in MULTIVIEW_BLIND_MODES:
            # Global conditions control affine modulation. V2 also uses the
            # global route for its heads; V3 corrects only the head route.
            # Branch/scan gates receive a bounded local correction in both.
            self.local_condition_adapter = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim),
            )
            nn.init.zeros_(self.local_condition_adapter[-1].weight)
            nn.init.zeros_(self.local_condition_adapter[-1].bias)
        if self.condition_mode == "blind_v3":
            self.routing_fusion = ReliabilityAwareRoutingFusion(
                dim,
                num_tasks,
                local_prior=self.v3_local_route_prior,
                route_mode=v3_route_mode,
            )

        self.encoder0 = nn.ModuleList(
            PromptLocalFrequencyBlock(
                dimensions[0], dim, use_frequency_branch=use_frequency_branch
            )
            for _ in range(stage_blocks[0])
        )
        self.down0 = nn.Conv2d(
            dimensions[0],
            dimensions[1],
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.encoder1 = nn.ModuleList(
            PromptLocalFrequencyBlock(
                dimensions[1], dim, use_frequency_branch=use_frequency_branch
            )
            for _ in range(stage_blocks[1])
        )
        self.down1 = nn.Conv2d(
            dimensions[1],
            dimensions[2],
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.bottleneck = nn.ModuleList(
            FourDirectionMambaBlock(
                dimensions[2],
                dim,
                d_state=d_state,
                scan_directions=scan_directions,
                direction_fusion=direction_fusion,
                local_global_fusion=local_global_fusion,
                use_frequency_branch=use_frequency_branch,
                bottleneck_type=bottleneck_type,
                batch_reverse_scans=batch_reverse_scans,
            )
            for _ in range(stage_blocks[2])
        )

        self.up1 = nn.Sequential(
            nn.Conv2d(dimensions[2], dimensions[1] * 4, kernel_size=1),
            nn.PixelShuffle(2),
        )
        self.fuse1 = nn.Conv2d(dimensions[1] * 2, dimensions[1], kernel_size=1)
        self.decoder1 = nn.ModuleList(
            PromptLocalFrequencyBlock(
                dimensions[1], dim, use_frequency_branch=use_frequency_branch
            )
            for _ in range(stage_blocks[1])
        )
        self.up0 = nn.Sequential(
            nn.Conv2d(dimensions[1], dimensions[0] * 4, kernel_size=1),
            nn.PixelShuffle(2),
        )
        self.fuse0 = nn.Conv2d(dimensions[0] * 2, dimensions[0], kernel_size=1)
        self.decoder0 = nn.ModuleList(
            PromptLocalFrequencyBlock(
                dimensions[0], dim, use_frequency_branch=use_frequency_branch
            )
            for _ in range(stage_blocks[0])
        )
        self.task_heads = nn.ModuleList(
            nn.Conv2d(dimensions[0], out_channels, kernel_size=3, padding=1)
            for _ in range(1 if self.shared_head else num_tasks)
        )
        for head in self.task_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    @staticmethod
    def _pad(feature):
        height, width = feature.shape[-2:]
        pad_height = (4 - height % 4) % 4
        pad_width = (4 - width % 4) % 4
        if pad_height or pad_width:
            padding_mode = (
                "reflect"
                if height > pad_height and width > pad_width
                else "replicate"
            )
            feature = F.pad(
                feature,
                (0, pad_width, 0, pad_height),
                mode=padding_mode,
            )
        return feature, height, width

    @staticmethod
    def _run(blocks, feature, condition, gate_condition=None):
        for block in blocks:
            feature = block(feature, condition, gate_condition=gate_condition)
        return feature

    def _build_condition(
        self,
        stem_feature,
        task_prompt,
        teacher_forcing,
        routing_probs_override,
        condition_override,
    ):
        if self.condition_mode == "oracle":
            if task_prompt is None:
                raise ValueError("Oracle condition mode requires task_prompt")
            routing_probs = F.one_hot(
                task_prompt,
                num_classes=self.num_tasks,
            ).to(dtype=stem_feature.dtype)
            condition = self.task_embed(task_prompt).to(dtype=stem_feature.dtype)
            return condition, routing_probs, None, None

        if routing_probs_override is not None:
            if routing_probs_override.shape != (
                stem_feature.size(0),
                self.num_tasks,
            ):
                raise ValueError(
                    "routing_probs_override must have shape (batch, num_tasks)"
                )
            routing_probs = routing_probs_override.float()
            routing_probs = routing_probs / routing_probs.sum(
                dim=1,
                keepdim=True,
            ).clamp_min(1e-8)
            if condition_override is not None:
                if condition_override.shape != (
                    stem_feature.size(0),
                    self.task_embed.embedding_dim,
                ):
                    raise ValueError(
                        "condition_override must have shape (batch, cond_dim)"
                    )
                condition = condition_override.to(dtype=stem_feature.dtype)
            else:
                condition = torch.matmul(
                    routing_probs.to(dtype=self.task_embed.weight.dtype),
                    self.task_embed.weight,
                ).to(dtype=stem_feature.dtype)
            return condition, routing_probs, None, None

        degradation_feature, task_logits = self.degradation_encoder(stem_feature)
        predicted_probs = task_logits.float().softmax(dim=1)
        teacher_forcing = float(teacher_forcing)
        if not 0.0 <= teacher_forcing <= 1.0:
            raise ValueError("teacher_forcing must be between 0 and 1")
        routing_probs = predicted_probs
        if task_prompt is not None and teacher_forcing > 0.0:
            oracle_probs = F.one_hot(
                task_prompt,
                num_classes=self.num_tasks,
            ).to(dtype=predicted_probs.dtype)
            routing_probs = (
                teacher_forcing * oracle_probs
                + (1.0 - teacher_forcing) * predicted_probs
            )
        condition = torch.matmul(
            routing_probs.to(dtype=self.task_embed.weight.dtype),
            self.task_embed.weight,
        )
        condition = condition + self.degradation_condition(degradation_feature)
        condition = condition.to(dtype=stem_feature.dtype)
        return condition, routing_probs, task_logits, degradation_feature

    def _v2_conditions(
        self,
        image,
        stem_feature,
        routing_context,
        task_prompt,
        teacher_forcing,
        routing_probs_override,
        condition_override,
    ):
        local_feature, local_logits = self.degradation_encoder(stem_feature)
        if routing_probs_override is None:
            if condition_override is not None:
                raise ValueError("blind_v2 condition_override needs routing_probs_override")
            context_image = image if routing_context is None else routing_context
            if context_image.ndim != 4 or context_image.shape[:2] != image.shape[:2]:
                raise ValueError("routing_context must have the same batch/channels as image")
            context = self.predict_task_context(
                context_image, task_prompt=task_prompt, teacher_forcing=teacher_forcing
            )
            condition = context["condition"].to(dtype=stem_feature.dtype)
            routing_probs = context["routing_probs"]
            task_logits = context["task_logits"]
            global_feature = context["degradation_feature"]
        else:
            if routing_context is not None:
                raise ValueError("Use routing_context or cached overrides, not both")
            expected_shape = (image.size(0), self.num_tasks)
            if routing_probs_override.shape != expected_shape:
                raise ValueError("routing_probs_override must have shape (batch, num_tasks)")
            routing_probs = routing_probs_override.to(device=image.device, dtype=torch.float32)
            if (
                not torch.isfinite(routing_probs).all()
                or (routing_probs < 0).any()
                or (routing_probs.sum(dim=1) <= 0).any()
            ):
                raise ValueError("Cached routing probabilities must be finite and nonnegative")
            routing_probs = routing_probs / routing_probs.sum(dim=1, keepdim=True)
            if condition_override is None or condition_override.shape != (
                image.size(0), self.task_embed.embedding_dim
            ):
                raise ValueError("blind_v2 needs the cached global condition (batch, cond_dim)")
            condition = condition_override.to(device=image.device, dtype=stem_feature.dtype)
            if not torch.isfinite(condition).all():
                raise ValueError("Cached condition must be finite")
            task_logits, global_feature = None, None
        gate_condition = condition + self.local_condition_scale * torch.tanh(
            self.local_condition_adapter(local_feature)
        ).to(dtype=condition.dtype)
        return (
            condition, gate_condition, routing_probs, task_logits,
            global_feature, local_logits, local_feature,
        )

    def _v3_conditions(
        self,
        image,
        stem_feature,
        routing_context,
        task_prompt,
        teacher_forcing,
        routing_probs_override,
        condition_override,
    ):
        """Build a stable global condition and a detail-corrected head route."""
        local_feature, local_logits = self.degradation_encoder(stem_feature)
        global_logits = None
        global_probs = None
        local_probs = None
        local_route_weight = None
        local_global_js = None
        condition_routing_probs = None
        base_global_condition = None
        if routing_probs_override is None:
            if condition_override is not None:
                raise ValueError("blind_v3 condition_override needs routing_probs_override")
            context_image = image if routing_context is None else routing_context
            if context_image.ndim != 4 or context_image.shape[:2] != image.shape[:2]:
                raise ValueError("routing_context must have the same batch/channels as image")
            global_context = self.predict_global_task_context(
                context_image,
                task_prompt=task_prompt,
                teacher_forcing=teacher_forcing,
            )
            fused_context = self.fuse_task_context(
                global_context,
                local_logits,
                local_feature,
                task_prompt=task_prompt,
                teacher_forcing=teacher_forcing,
            )
            base_global_condition = global_context["condition"].to(
                dtype=stem_feature.dtype
            )
            condition = fused_context["condition"].to(dtype=stem_feature.dtype)
            condition_routing_probs = fused_context["condition_routing_probs"]
            routing_probs = fused_context["routing_probs"]
            task_logits = fused_context["task_logits"]
            global_feature = global_context["degradation_feature"]
            global_logits = global_context["task_logits"]
            global_probs = fused_context["global_task_probs"]
            local_probs = fused_context["local_task_probs"]
            local_route_weight = fused_context["local_route_weight"]
            local_global_js = fused_context["local_global_js"]
        else:
            if routing_context is not None:
                raise ValueError("Use routing_context or cached overrides, not both")
            expected_shape = (image.size(0), self.num_tasks)
            if routing_probs_override.shape != expected_shape:
                raise ValueError("routing_probs_override must have shape (batch, num_tasks)")
            routing_probs = routing_probs_override.to(device=image.device, dtype=torch.float32)
            if (
                not torch.isfinite(routing_probs).all()
                or (routing_probs < 0).any()
                or (routing_probs.sum(dim=1) <= 0).any()
            ):
                raise ValueError("Cached routing probabilities must be finite and nonnegative")
            routing_probs = routing_probs / routing_probs.sum(dim=1, keepdim=True)
            if condition_override is None or condition_override.shape != (
                image.size(0), self.task_embed.embedding_dim
            ):
                raise ValueError("blind_v3 needs the cached global condition (batch, cond_dim)")
            condition = condition_override.to(device=image.device, dtype=stem_feature.dtype)
            if not torch.isfinite(condition).all():
                raise ValueError("Cached condition must be finite")
            task_logits, global_feature = None, None
            base_global_condition = condition
        gate_condition = condition + self.local_condition_scale * torch.tanh(
            self.local_condition_adapter(local_feature)
        ).to(dtype=condition.dtype)
        return (
            condition,
            gate_condition,
            routing_probs,
            task_logits,
            global_feature,
            local_logits,
            local_feature,
            global_logits,
            global_probs,
            local_probs,
            local_route_weight,
            local_global_js,
            condition_routing_probs,
            base_global_condition,
        )

    def forward(
        self,
        image,
        task_prompt=None,
        return_aux=False,
        teacher_forcing=0.0,
        routing_probs_override=None,
        condition_override=None,
        routing_context=None,
    ):
        if routing_context is not None and self.condition_mode not in MULTIVIEW_BLIND_MODES:
            raise ValueError("routing_context is only supported by multiview blind modes")
        padded, original_height, original_width = self._pad(image)
        stem_feature = self.stem(padded)
        local_logits, local_feature = None, None
        global_logits, global_probs, local_probs = None, None, None
        local_route_weight, local_global_js = None, None
        condition_routing_probs = None
        base_global_condition = None
        if self.condition_mode == "blind_v2":
            (
                condition, gate_condition, routing_probs, task_logits,
                degradation_feature, local_logits, local_feature,
            ) = self._v2_conditions(
                image, stem_feature, routing_context, task_prompt, teacher_forcing,
                routing_probs_override, condition_override,
            )
        elif self.condition_mode == "blind_v3":
            (
                condition,
                gate_condition,
                routing_probs,
                task_logits,
                degradation_feature,
                local_logits,
                local_feature,
                global_logits,
                global_probs,
                local_probs,
                local_route_weight,
                local_global_js,
                condition_routing_probs,
                base_global_condition,
            ) = self._v3_conditions(
                image,
                stem_feature,
                routing_context,
                task_prompt,
                teacher_forcing,
                routing_probs_override,
                condition_override,
            )
        else:
            condition, routing_probs, task_logits, degradation_feature = (
                self._build_condition(
                    stem_feature,
                    task_prompt,
                    teacher_forcing,
                    routing_probs_override,
                    condition_override,
                )
            )
            gate_condition = condition
        if not self.use_task_condition:
            condition = torch.zeros_like(condition)
            gate_condition = torch.zeros_like(gate_condition)

        feature0 = self._run(self.encoder0, stem_feature, condition, gate_condition)
        feature1 = self._run(self.encoder1, self.down0(feature0), condition, gate_condition)
        feature2 = self._run(self.bottleneck, self.down1(feature1), condition, gate_condition)

        decoded1 = self.up1(feature2)
        decoded1 = self.fuse1(torch.cat((decoded1, feature1), dim=1))
        decoded1 = self._run(self.decoder1, decoded1, condition, gate_condition)
        decoded0 = self.up0(decoded1)
        decoded0 = self.fuse0(torch.cat((decoded0, feature0), dim=1))
        decoded0 = self._run(self.decoder0, decoded0, condition, gate_condition)

        if self.shared_head:
            residual = self.task_heads[0](decoded0)
        else:
            task_outputs = torch.stack(
                [head(decoded0) for head in self.task_heads],
                dim=1,
            )
            if self.condition_mode in BLIND_MODES:
                residual = (
                    task_outputs
                    * routing_probs.to(dtype=task_outputs.dtype)[
                        :, :, None, None, None
                    ]
                ).sum(dim=1)
            else:
                gather_index = task_prompt[:, None, None, None, None].expand(
                    -1,
                    1,
                    task_outputs.size(2),
                    task_outputs.size(3),
                    task_outputs.size(4),
                )
                residual = torch.gather(task_outputs, 1, gather_index).squeeze(1)
        restored = padded + residual
        restored = restored[:, :, :original_height, :original_width]
        if not return_aux:
            return restored
        return restored, {
            "task_logits": task_logits,
            "task_probs": None
            if task_logits is None
            else task_logits.float().softmax(dim=1),
            "routing_probs": routing_probs,
            "degradation_feature": degradation_feature,
            "local_task_logits": local_logits,
            "local_task_probs": local_probs,
            "local_degradation_feature": local_feature,
            "global_task_logits": global_logits,
            "global_task_probs": global_probs,
            "local_route_weight": local_route_weight,
            "local_global_js": local_global_js,
            "condition_routing_probs": condition_routing_probs,
            "global_condition": (
                base_global_condition
                if base_global_condition is not None
                else condition
            ),
            "backbone_condition": condition,
            "gate_condition": gate_condition,
        }

    def predict_global_task_context(
        self, image, max_size=None, task_prompt=None, teacher_forcing=0.0
    ):
        """Predict one global routing distribution and continuous condition."""
        if self.condition_mode not in BLIND_MODES:
            raise RuntimeError("Task probabilities are only available in blind modes")
        if max_size is None:
            max_size = (
                self.routing_context_size
                if self.condition_mode in MULTIVIEW_BLIND_MODES
                else 256
            )
        if self.condition_mode in MULTIVIEW_BLIND_MODES:
            image = prepare_routing_context(image, max_size)
        height, width = image.shape[-2:]
        longest_side = max(height, width)
        if max_size and longest_side > max_size:
            scale = float(max_size) / float(longest_side)
            image = F.interpolate(
                image,
                size=(max(1, round(height * scale)), max(1, round(width * scale))),
                mode="bilinear",
                align_corners=False,
            )
        padded, _, _ = self._pad(image)
        stem_feature = self.stem(padded)
        degradation_feature, task_logits = self.degradation_encoder(stem_feature)
        task_probs = task_logits.float().softmax(dim=1)
        teacher_forcing = float(teacher_forcing)
        if not 0.0 <= teacher_forcing <= 1.0:
            raise ValueError("teacher_forcing must be between 0 and 1")
        routing_probs = task_probs
        if task_prompt is not None and teacher_forcing > 0.0:
            oracle_probs = F.one_hot(task_prompt, num_classes=self.num_tasks).float()
            routing_probs = teacher_forcing * oracle_probs + (1.0 - teacher_forcing) * task_probs
        condition = torch.matmul(
            routing_probs.to(dtype=self.task_embed.weight.dtype),
            self.task_embed.weight,
        )
        condition = condition + self.degradation_condition(degradation_feature)
        return {
            "task_logits": task_logits,
            "task_probs": task_probs,
            "routing_probs": routing_probs,
            "degradation_feature": degradation_feature,
            "condition": condition,
        }

    def predict_local_task_context(self, image):
        """Predict the detail-view route without running the restoration trunk."""
        if self.condition_mode != "blind_v3":
            raise RuntimeError("Local routing context is only exposed by blind_v3")
        padded, _, _ = self._pad(image)
        stem_feature = self.stem(padded)
        degradation_feature, task_logits = self.degradation_encoder(stem_feature)
        return {
            "task_logits": task_logits,
            "task_probs": task_logits.float().softmax(dim=1),
            "degradation_feature": degradation_feature,
        }

    def build_v3_condition(self, global_context, routing_probs):
        """Optionally let the stable fused route correct PromptAffine tokens."""
        if self.condition_mode != "blind_v3":
            raise RuntimeError("Fused PromptAffine conditions are only available in blind_v3")
        if not self.v3_fuse_prompt_condition:
            return global_context["condition"]
        if routing_probs.shape != (global_context["task_logits"].size(0), self.num_tasks):
            raise ValueError("routing_probs must have shape (batch, num_tasks)")
        condition = torch.matmul(
            routing_probs.to(dtype=self.task_embed.weight.dtype),
            self.task_embed.weight,
        )
        condition = condition + self.degradation_condition(
            global_context["degradation_feature"]
        )
        return condition

    def fuse_task_context(
        self,
        global_context,
        local_logits,
        local_feature,
        task_prompt=None,
        teacher_forcing=0.0,
    ):
        if self.condition_mode != "blind_v3":
            raise RuntimeError("Reliability-aware routing fusion is only available in blind_v3")
        fusion = self.routing_fusion(
            global_context["task_logits"],
            local_logits,
            global_context["degradation_feature"],
            local_feature,
        )
        teacher_forcing = float(teacher_forcing)
        if not 0.0 <= teacher_forcing <= 1.0:
            raise ValueError("teacher_forcing must be between 0 and 1")
        routing_probs = fusion["task_probs"]
        if task_prompt is not None and teacher_forcing > 0.0:
            oracle_probs = F.one_hot(task_prompt, num_classes=self.num_tasks).float()
            routing_probs = (
                teacher_forcing * oracle_probs
                + (1.0 - teacher_forcing) * routing_probs
            )
        condition = self.build_v3_condition(global_context, routing_probs)
        return {
            **fusion,
            "routing_probs": routing_probs,
            "condition": condition,
            "global_condition": global_context["condition"],
            "condition_routing_probs": (
                routing_probs
                if self.v3_fuse_prompt_condition
                else global_context["routing_probs"]
            ),
            "degradation_feature": global_context["degradation_feature"],
            "global_task_logits": global_context["task_logits"],
            "local_task_logits": local_logits,
            "local_degradation_feature": local_feature,
        }

    def predict_task_context(
        self, image, max_size=None, task_prompt=None, teacher_forcing=0.0
    ):
        if self.condition_mode != "blind_v3":
            return self.predict_global_task_context(
                image,
                max_size=max_size,
                task_prompt=task_prompt,
                teacher_forcing=teacher_forcing,
            )
        global_context = self.predict_global_task_context(
            image,
            max_size=max_size,
            task_prompt=task_prompt,
            teacher_forcing=teacher_forcing,
        )
        local_context = self.predict_local_task_context(image)
        return self.fuse_task_context(
            global_context,
            local_context["task_logits"],
            local_context["degradation_feature"],
            task_prompt=task_prompt,
            teacher_forcing=teacher_forcing,
        )

    def predict_task_probs(self, image, max_size=None):
        return self.predict_task_context(image, max_size=max_size)["task_probs"]
