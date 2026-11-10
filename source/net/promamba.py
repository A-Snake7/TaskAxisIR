import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from utils.modes import BLIND_MODES, MULTIVIEW_BLIND_MODES

from .promamba_v2 import DegradationEncoder, HierarchicalProMamba

try:
    from mamba_ssm import Mamba
except ImportError as error:
    raise ImportError(
        "ProMamba requires mamba-ssm and causal-conv1d. "
        "Activate the airnet environment before running this project."
    ) from error


def _make_spatial_norm(channels, norm_type):
    if norm_type == "batch":
        return nn.BatchNorm2d(channels)
    if norm_type == "group":
        groups = min(8, channels)
        while channels % groups:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    raise ValueError("Unsupported norm_type: {}".format(norm_type))


class ConditionAwareLGMambaBlock(nn.Module):
    """Prompt-conditioned local/global Mamba block."""

    def __init__(
        self,
        d_model,
        cond_dim,
        d_state=16,
        norm_type="batch",
        gate_mode="legacy",
        identity_init=False,
    ):
        super().__init__()
        if gate_mode not in ("legacy", "competitive"):
            raise ValueError("Unsupported gate_mode: {}".format(gate_mode))
        self.d_model = d_model
        self.gate_mode = gate_mode

        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model * 2),
        )
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model=d_model, d_state=d_state, expand=2)
        self.local_conv = nn.Sequential(
            nn.Conv2d(
                d_model,
                d_model,
                kernel_size=3,
                padding=1,
                groups=d_model,
            ),
            _make_spatial_norm(d_model, norm_type),
            nn.SiLU(),
        )
        self.gate = nn.Conv2d(d_model, d_model, kernel_size=1)
        self.silu = nn.SiLU()
        self.out_proj = nn.Conv2d(d_model, d_model, kernel_size=1)

        if identity_init:
            nn.init.zeros_(self.cond_mlp[-1].weight)
            nn.init.zeros_(self.cond_mlp[-1].bias)
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, x, cond):
        batch, channels, height, width = x.shape
        gamma, beta = self.cond_mlp(cond).chunk(2, dim=1)
        gamma = gamma.view(batch, channels, 1, 1)
        beta = beta.view(batch, channels, 1, 1)
        modulated = x * (1 + gamma) + beta

        local_feature = self.local_conv(modulated)
        sequence = rearrange(modulated, "b c h w -> b (h w) c")
        global_feature = self.mamba(self.norm(sequence))
        global_feature = rearrange(
            global_feature,
            "b (h w) c -> b c h w",
            h=height,
            w=width,
        )

        gate_logits = self.gate(modulated)
        if self.gate_mode == "competitive":
            gate_weight = torch.sigmoid(gate_logits)
            fused = gate_weight * global_feature + (1 - gate_weight) * local_feature
        else:
            # Kept for compatibility with the June 2026 epoch-130 checkpoint.
            fused = self.silu(gate_logits) * (global_feature + local_feature)

        return x + self.out_proj(fused)


class ProMamba(nn.Module):
    """Prompt-conditioned restoration model with legacy compatibility."""

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        dim=64,
        num_blocks=6,
        num_tasks=3,
        d_state=16,
        norm_type="batch",
        gate_mode="legacy",
        identity_init=False,
        architecture="lightweight",
        stage_blocks=(2, 3, 4),
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
    ):
        super().__init__()
        if architecture not in ("lightweight", "hierarchical"):
            raise ValueError("Unsupported architecture: {}".format(architecture))
        if condition_mode not in ("oracle",) + BLIND_MODES:
            raise ValueError("Unsupported condition_mode: {}".format(condition_mode))
        if condition_mode in MULTIVIEW_BLIND_MODES and architecture != "hierarchical":
            raise ValueError("{} requires the hierarchical architecture".format(condition_mode))
        self.architecture = architecture
        self.condition_mode = condition_mode
        self.num_tasks = int(num_tasks)
        self.model_config = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "dim": dim,
            "num_blocks": num_blocks,
            "num_tasks": num_tasks,
            "d_state": d_state,
            "norm_type": norm_type,
            "gate_mode": gate_mode,
            "identity_init": identity_init,
            "architecture": architecture,
            "stage_blocks": tuple(stage_blocks),
            "scan_directions": scan_directions,
            "direction_fusion": direction_fusion,
            "local_global_fusion": local_global_fusion,
            "use_frequency_branch": use_frequency_branch,
            "use_task_condition": use_task_condition,
            "shared_head": shared_head,
            "bottleneck_type": bottleneck_type,
            "condition_mode": condition_mode,
            "batch_reverse_scans": batch_reverse_scans,
            "routing_context_size": int(routing_context_size),
            "local_condition_scale": float(local_condition_scale),
            "v3_local_route_prior": float(v3_local_route_prior),
            "v3_fuse_prompt_condition": bool(v3_fuse_prompt_condition),
        }
        if architecture == "hierarchical":
            self.hierarchical = HierarchicalProMamba(
                in_channels=in_channels,
                out_channels=out_channels,
                dim=dim,
                stage_blocks=stage_blocks,
                num_tasks=num_tasks,
                d_state=d_state,
                scan_directions=scan_directions,
                direction_fusion=direction_fusion,
                local_global_fusion=local_global_fusion,
                use_frequency_branch=use_frequency_branch,
                use_task_condition=use_task_condition,
                shared_head=shared_head,
                bottleneck_type=bottleneck_type,
                condition_mode=condition_mode,
                batch_reverse_scans=batch_reverse_scans,
                routing_context_size=routing_context_size,
                local_condition_scale=local_condition_scale,
                v3_local_route_prior=v3_local_route_prior,
                v3_fuse_prompt_condition=v3_fuse_prompt_condition,
            )
            return

        self.task_embed = nn.Embedding(
            num_embeddings=num_tasks,
            embedding_dim=dim,
        )
        self.stem = nn.Conv2d(
            in_channels,
            dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        if self.condition_mode == "blind_soft":
            self.degradation_encoder = DegradationEncoder(
                dim,
                dim,
                num_tasks,
            )
            self.degradation_condition = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim),
            )
            nn.init.zeros_(self.degradation_condition[-1].weight)
            nn.init.zeros_(self.degradation_condition[-1].bias)
        self.backbone = nn.ModuleList(
            [
                ConditionAwareLGMambaBlock(
                    d_model=dim,
                    cond_dim=dim,
                    d_state=d_state,
                    norm_type=norm_type,
                    gate_mode=gate_mode,
                    identity_init=identity_init,
                )
                for _ in range(num_blocks)
            ]
        )
        self.head = nn.Conv2d(
            dim,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        if identity_init:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def forward(
        self,
        x,
        task_prompt=None,
        return_aux=False,
        teacher_forcing=0.0,
        routing_probs_override=None,
        condition_override=None,
        routing_context=None,
    ):
        if self.architecture == "hierarchical":
            return self.hierarchical(
                x,
                task_prompt,
                return_aux=return_aux,
                teacher_forcing=teacher_forcing,
                routing_probs_override=routing_probs_override,
                condition_override=condition_override,
                routing_context=routing_context,
            )
        if routing_context is not None:
            raise ValueError(
                "routing_context is only supported by hierarchical multiview blind modes"
            )
        feature = self.stem(x)
        task_logits = None
        degradation_feature = None
        if self.condition_mode == "oracle":
            if task_prompt is None:
                raise ValueError("Oracle condition mode requires task_prompt")
            routing_probs = F.one_hot(
                task_prompt,
                num_classes=self.num_tasks,
            ).to(dtype=feature.dtype)
            prompt_vector = self.task_embed(task_prompt).to(dtype=feature.dtype)
        else:
            if routing_probs_override is not None:
                if routing_probs_override.shape != (x.size(0), self.num_tasks):
                    raise ValueError(
                        "routing_probs_override must have shape (batch, num_tasks)"
                    )
                routing_probs = routing_probs_override.float()
                routing_probs = routing_probs / routing_probs.sum(
                    dim=1,
                    keepdim=True,
                ).clamp_min(1e-8)
            else:
                degradation_feature, task_logits = self.degradation_encoder(feature)
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
            if condition_override is not None:
                if condition_override.shape != (x.size(0), self.task_embed.embedding_dim):
                    raise ValueError(
                        "condition_override must have shape (batch, cond_dim)"
                    )
                prompt_vector = condition_override.to(dtype=feature.dtype)
            else:
                prompt_vector = torch.matmul(
                    routing_probs.to(dtype=self.task_embed.weight.dtype),
                    self.task_embed.weight,
                )
                if degradation_feature is not None:
                    prompt_vector = prompt_vector + self.degradation_condition(
                        degradation_feature
                    )
                prompt_vector = prompt_vector.to(dtype=feature.dtype)
        residual = feature
        for block in self.backbone:
            feature = block(feature, prompt_vector)
        output = self.head(feature + residual)
        restored = output + x
        if not return_aux:
            return restored
        return restored, {
            "task_logits": task_logits,
            "task_probs": None
            if task_logits is None
            else task_logits.float().softmax(dim=1),
            "routing_probs": routing_probs,
            "degradation_feature": degradation_feature,
        }

    def predict_task_context(self, x, max_size=None, task_prompt=None, teacher_forcing=0.0):
        """Predict global soft routing and its continuous degradation condition."""
        if self.condition_mode not in BLIND_MODES:
            raise RuntimeError("Task probabilities are only available in blind modes")
        if self.architecture == "hierarchical":
            return self.hierarchical.predict_task_context(
                x, max_size=max_size, task_prompt=task_prompt, teacher_forcing=teacher_forcing
            )
        if max_size is None:
            max_size = 256
        height, width = x.shape[-2:]
        longest_side = max(height, width)
        if max_size and longest_side > max_size:
            scale = float(max_size) / float(longest_side)
            x = F.interpolate(
                x,
                size=(max(1, round(height * scale)), max(1, round(width * scale))),
                mode="bilinear",
                align_corners=False,
            )
        feature = self.stem(x)
        degradation_feature, task_logits = self.degradation_encoder(feature)
        task_probs = task_logits.float().softmax(dim=1)
        condition = torch.matmul(
            task_probs.to(dtype=self.task_embed.weight.dtype),
            self.task_embed.weight,
        )
        condition = condition + self.degradation_condition(degradation_feature)
        return {
            "task_logits": task_logits,
            "task_probs": task_probs,
            "routing_probs": task_probs,
            "degradation_feature": degradation_feature,
            "condition": condition,
        }

    def predict_global_task_context(
        self, x, max_size=None, task_prompt=None, teacher_forcing=0.0
    ):
        if self.architecture != "hierarchical":
            return self.predict_task_context(
                x,
                max_size=max_size,
                task_prompt=task_prompt,
                teacher_forcing=teacher_forcing,
            )
        return self.hierarchical.predict_global_task_context(
            x,
            max_size=max_size,
            task_prompt=task_prompt,
            teacher_forcing=teacher_forcing,
        )

    def predict_local_task_context(self, x):
        if self.architecture != "hierarchical":
            raise RuntimeError(
                "Local routing context requires the hierarchical architecture"
            )
        return self.hierarchical.predict_local_task_context(x)

    def fuse_task_context(
        self,
        global_context,
        local_logits,
        local_feature,
        task_prompt=None,
        teacher_forcing=0.0,
    ):
        if self.architecture != "hierarchical":
            raise RuntimeError("Routing fusion requires the hierarchical architecture")
        return self.hierarchical.fuse_task_context(
            global_context,
            local_logits,
            local_feature,
            task_prompt=task_prompt,
            teacher_forcing=teacher_forcing,
        )

    def build_v3_condition(self, global_context, routing_probs):
        if self.architecture != "hierarchical":
            raise RuntimeError("Fused PromptAffine conditions require hierarchical mode")
        return self.hierarchical.build_v3_condition(global_context, routing_probs)

    def predict_task_probs(self, x, max_size=None):
        return self.predict_task_context(x, max_size=max_size)["task_probs"]
