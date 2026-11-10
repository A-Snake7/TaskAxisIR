import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.inference import tiled_inference


class FakeBlindModel(nn.Module):
    condition_mode = "blind_soft"
    num_tasks = 3

    def __init__(self):
        super().__init__()
        self.context_calls = 0
        self.forward_calls = 0
        self.override_calls = 0

    def predict_task_context(self, image):
        self.context_calls += 1
        probabilities = torch.tensor(
            [[0.1, 0.7, 0.2]],
            dtype=image.dtype,
            device=image.device,
        )
        return {
            "task_probs": probabilities,
            "condition": torch.ones(
                image.size(0),
                4,
                dtype=image.dtype,
                device=image.device,
            ),
        }

    def forward(
        self,
        image,
        task_prompt=None,
        return_aux=False,
        teacher_forcing=0.0,
        routing_probs_override=None,
        condition_override=None,
    ):
        self.forward_calls += 1
        if routing_probs_override is not None:
            self.override_calls += 1
            probabilities = routing_probs_override
            if condition_override is None:
                raise AssertionError("Global continuous condition was not reused")
        elif task_prompt is not None and teacher_forcing > 0:
            probabilities = F.one_hot(task_prompt, num_classes=3).float()
        else:
            probabilities = torch.tensor(
                [[0.6, 0.3, 0.1]],
                dtype=image.dtype,
                device=image.device,
            )
        if return_aux:
            return image, {"task_probs": probabilities}
        return image


class BlindInferenceTests(unittest.TestCase):
    def test_metadata_prompt_does_not_disable_global_blind_routing(self):
        model = FakeBlindModel()
        image = torch.rand(1, 3, 40, 48)
        restored, routing = tiled_inference(
            model,
            image,
            torch.tensor([1]),
            tile_size=32,
            overlap=8,
            amp_enabled=False,
            teacher_forcing=0.0,
            return_routing=True,
        )
        self.assertTrue(torch.allclose(restored, image, atol=1e-6))
        self.assertEqual(model.context_calls, 1)
        self.assertEqual(model.override_calls, model.forward_calls)
        self.assertTrue(routing["global_routing"])
        self.assertTrue(
            torch.allclose(
                routing["task_probs"],
                torch.tensor([[0.1, 0.7, 0.2]]),
            )
        )

    def test_oracle_ablation_skips_blind_prediction(self):
        model = FakeBlindModel()
        image = torch.rand(1, 3, 20, 20)
        _, routing = tiled_inference(
            model,
            image,
            torch.tensor([2]),
            tile_size=32,
            overlap=8,
            amp_enabled=False,
            teacher_forcing=1.0,
            return_routing=True,
        )
        self.assertEqual(model.context_calls, 0)
        self.assertTrue(
            torch.equal(
                routing["task_probs"],
                torch.tensor([[0.0, 0.0, 1.0]]),
            )
        )


if __name__ == "__main__":
    unittest.main()
