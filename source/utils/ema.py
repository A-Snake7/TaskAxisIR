import copy

import torch


class ModelEMA:
    """Exponential moving average of model parameters and floating buffers."""

    def __init__(self, model, decay=0.999):
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must be in [0, 1)")
        self.decay = float(decay)
        self.module = copy.deepcopy(model).eval()
        self.module.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        ema_state = self.module.state_dict()
        model_state = model.state_dict()
        if ema_state.keys() != model_state.keys():
            raise RuntimeError("EMA/model state dictionaries do not match")
        for name, ema_value in ema_state.items():
            model_value = model_state[name].detach()
            if torch.is_floating_point(ema_value):
                ema_value.mul_(self.decay).add_(
                    model_value,
                    alpha=1.0 - self.decay,
                )
            else:
                ema_value.copy_(model_value)

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state_dict):
        self.module.load_state_dict(state_dict, strict=True)
