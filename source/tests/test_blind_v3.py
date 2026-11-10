import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn

from net.promamba import ProMamba
from net.promamba_v2 import ReliabilityAwareRoutingFusion
from train import configure_training_protocol, parse_args, set_blind_trainable
from utils.checkpoint import load_model_for_inference
from utils.inference import tiled_inference
from utils.losses import RestorationLoss, blind_v3_routing_loss


def small_model(
    mode="blind_v3",
    v3_fuse_prompt_condition=False,
    v3_local_route_prior=0.5,
):
    model = ProMamba(
        architecture="hierarchical",
        dim=8,
        stage_blocks=(1, 1, 1),
        bottleneck_type="conv",
        condition_mode=mode,
        routing_context_size=32,
        local_condition_scale=0.2,
        v3_local_route_prior=v3_local_route_prior,
        v3_fuse_prompt_condition=v3_fuse_prompt_condition,
    )
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) and (
                name.endswith("output") or "task_heads" in name
            ):
                nn.init.normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)
    return model


class BlindV3Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(37)

    def test_zero_initialized_gate_starts_from_configured_probability_mix(self):
        fusion = ReliabilityAwareRoutingFusion(8, 3, local_prior=0.25)
        global_logits = torch.tensor([[3.0, 0.0, -1.0]])
        local_logits = torch.tensor([[-1.0, 2.0, 0.0]])
        feature = torch.randn(1, 8)
        result = fusion(global_logits, local_logits, feature, feature)
        expected = (
            0.75 * global_logits.softmax(1)
            + 0.25 * local_logits.softmax(1)
        )
        torch.testing.assert_close(result["task_probs"], expected)
        torch.testing.assert_close(
            result["local_route_weight"], torch.tensor([[0.25]])
        )

    def test_v2_warm_start_only_adds_fusion_parameters(self):
        v2 = small_model("blind_v2")
        v3 = small_model("blind_v3")
        result = v3.load_state_dict(v2.state_dict(), strict=False)
        self.assertFalse(result.unexpected_keys)
        self.assertTrue(result.missing_keys)
        self.assertTrue(all("routing_fusion" in key for key in result.missing_keys))

    def test_global_condition_is_stable_while_detail_changes_head_route(self):
        model = small_model().eval()
        context = torch.rand(1, 3, 32, 32)
        first_image = torch.zeros(1, 3, 24, 24)
        second_image = torch.rand(1, 3, 24, 24)
        with torch.no_grad():
            _, first = model(first_image, routing_context=context, return_aux=True)
            _, second = model(second_image, routing_context=context, return_aux=True)
        torch.testing.assert_close(
            first["global_condition"], second["global_condition"], atol=0, rtol=0
        )
        self.assertGreater(
            (first["local_task_logits"] - second["local_task_logits"]).abs().max().item(),
            1e-7,
        )
        self.assertGreater(
            (first["routing_probs"] - second["routing_probs"]).abs().max().item(),
            1e-7,
        )
        self.assertIsNotNone(first["local_route_weight"])
        torch.testing.assert_close(
            first["routing_probs"].sum(1), torch.ones(1), atol=1e-6, rtol=0
        )

    def test_cached_fused_route_matches_direct_full_image_inference(self):
        model = small_model().eval()
        image = torch.rand(1, 3, 24, 28)
        with torch.no_grad():
            direct = model(image)
            context = model.predict_task_context(image)
            cached = model(
                image,
                routing_probs_override=context["routing_probs"],
                condition_override=context["condition"],
            )
        torch.testing.assert_close(direct, cached, atol=1e-6, rtol=1e-6)

    def test_tiling_fuses_one_image_route_and_reuses_it_for_every_tile(self):
        model = small_model().eval()
        calls = []
        cached_routes = []
        cached_conditions = []
        hook = model.hierarchical.degradation_encoder.register_forward_hook(
            lambda *args: calls.append(1)
        )
        def capture_cached_context(_module, _args, kwargs):
            if kwargs.get("routing_probs_override") is not None:
                cached_routes.append(
                    kwargs["routing_probs_override"].detach().clone()
                )
                cached_conditions.append(
                    kwargs["condition_override"].detach().clone()
                )

        pre_hook = model.register_forward_pre_hook(
            capture_cached_context,
            with_kwargs=True,
        )
        image = torch.rand(1, 3, 40, 48)
        try:
            restored, routing = tiled_inference(
                model,
                image,
                tile_size=32,
                overlap=8,
                amp_enabled=False,
                return_routing=True,
            )
        finally:
            hook.remove()
            pre_hook.remove()
        self.assertEqual(restored.shape, image.shape)
        self.assertEqual(len(calls), 1 + 2 * routing["tile_count"])
        self.assertEqual(len(cached_routes), routing["tile_count"])
        for route, condition in zip(cached_routes[1:], cached_conditions[1:]):
            torch.testing.assert_close(route, cached_routes[0], atol=0, rtol=0)
            torch.testing.assert_close(condition, cached_conditions[0], atol=0, rtol=0)
        self.assertTrue(routing["global_routing"])
        self.assertIsNotNone(routing["global_task_probs"])
        self.assertIsNotNone(routing["local_task_probs"])
        self.assertGreaterEqual(float(routing["local_route_weight"]), 0.0)
        self.assertLessEqual(float(routing["local_route_weight"]), 1.0)
        self.assertEqual(
            routing["routing_aggregation"], "coverage_weighted_per_tile_fusion"
        )

    def test_fused_prompt_condition_is_built_once_and_cached_across_tiles(self):
        model = small_model(v3_fuse_prompt_condition=True).eval()
        cached_conditions = []

        def capture_cached_context(_module, _args, kwargs):
            if kwargs.get("condition_override") is not None:
                cached_conditions.append(
                    kwargs["condition_override"].detach().clone()
                )

        pre_hook = model.register_forward_pre_hook(
            capture_cached_context,
            with_kwargs=True,
        )
        image = torch.rand(1, 3, 40, 48)
        try:
            with torch.no_grad():
                _, routing = tiled_inference(
                    model,
                    image,
                    tile_size=32,
                    overlap=8,
                    amp_enabled=False,
                    return_routing=True,
                )
                global_context = model.predict_global_task_context(image)
                expected = model.build_v3_condition(
                    global_context,
                    routing["routing_probs"],
                )
        finally:
            pre_hook.remove()

        self.assertEqual(len(cached_conditions), routing["tile_count"])
        for condition in cached_conditions:
            torch.testing.assert_close(condition, expected, atol=1e-6, rtol=1e-6)
        self.assertGreater(
            (expected - global_context["condition"]).abs().max().item(),
            1e-7,
        )

    def test_single_tile_uses_temperature_once_and_matches_full_route(self):
        model = small_model().eval()
        with torch.no_grad():
            model.hierarchical.routing_fusion.log_temperatures.copy_(
                torch.tensor([0.7, -0.4])
            )
        image = torch.rand(1, 3, 23, 29)
        with torch.no_grad():
            expected = model.predict_task_context(image)
            _, routing = tiled_inference(
                model,
                image,
                tile_size=32,
                overlap=8,
                amp_enabled=False,
                return_routing=True,
            )
        torch.testing.assert_close(
            routing["task_probs"], expected["task_probs"], atol=1e-6, rtol=1e-6
        )
        torch.testing.assert_close(
            routing["global_task_probs"],
            expected["global_task_probs"],
            atol=1e-6,
            rtol=1e-6,
        )

    def test_label_aware_consistency_teaches_the_weaker_view(self):
        global_logits = torch.tensor([[-2.0, 2.0, 0.0]], requires_grad=True)
        local_logits = torch.tensor([[3.0, 0.0, -1.0]], requires_grad=True)
        global_feature = torch.randn(1, 8, requires_grad=True)
        local_feature = torch.randn(1, 8, requires_grad=True)
        auxiliary = {
            "task_logits": torch.log(
                0.5 * global_logits.softmax(1) + 0.5 * local_logits.softmax(1)
            ),
            "global_task_logits": global_logits,
            "local_task_logits": local_logits,
            "degradation_feature": global_feature,
            "local_degradation_feature": local_feature,
        }
        terms = blind_v3_routing_loss(auxiliary, torch.tensor([0]))
        self.assertAlmostEqual(float(terms["global_teacher_fraction"]), 0.0)
        (terms["probability_consistency"] + terms["feature_consistency"]).backward()
        self.assertGreater(global_logits.grad.abs().sum().item(), 0)
        self.assertLess(
            0.0 if local_logits.grad is None else local_logits.grad.abs().sum().item(),
            1e-8,
        )
        self.assertGreater(global_feature.grad.abs().sum().item(), 0)
        self.assertLess(
            0.0 if local_feature.grad is None else local_feature.grad.abs().sum().item(),
            1e-8,
        )

    def test_restoration_and_routing_gradients_reach_fusion_module(self):
        model = small_model().train()
        image = torch.rand(2, 3, 24, 24)
        output, auxiliary = model(
            image,
            routing_context=torch.rand(2, 3, 32, 32),
            return_aux=True,
        )
        restoration, _ = RestorationLoss()(output, torch.rand_like(output))
        routing = blind_v3_routing_loss(auxiliary, torch.tensor([0, 1]))
        loss = (
            restoration.mean()
            + 0.1 * routing["combined_cls"]
            + 0.01 * routing["probability_consistency"]
            + 0.01 * routing["feature_consistency"]
        )
        loss.backward()
        fusion = model.hierarchical.routing_fusion
        self.assertGreater(fusion.gate[-1].weight.grad.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(fusion.log_temperatures.grad).all())
        set_blind_trainable(model, 1, 3)
        self.assertTrue(fusion.gate[-1].weight.requires_grad)
        self.assertFalse(model.hierarchical.stem.weight.requires_grad)

    def test_checkpoint_and_cli_preserve_v3_configuration(self):
        model = small_model(v3_fuse_prompt_condition=True).eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v3.pth"
            torch.save(
                {
                    "model_config": model.model_config,
                    "model_state_dict": model.state_dict(),
                },
                path,
            )
            loaded, _, config = load_model_for_inference(str(path), "cpu")
            self.assertEqual(config["condition_mode"], "blind_v3")
            self.assertEqual(config["v3_local_route_prior"], 0.5)
            self.assertTrue(config["v3_fuse_prompt_condition"])
            image = torch.rand(1, 3, 24, 28)
            with torch.no_grad():
                torch.testing.assert_close(model(image), loaded(image), atol=0, rtol=0)
        with patch(
            "sys.argv",
            [
                "train.py",
                "--condition_mode",
                "blind_v3",
                "--v3_local_route_prior",
                "0.7",
                "--v3_fuse_prompt_condition",
                "--init_weights",
                "warmstart.pth",
            ],
        ):
            args = parse_args()
        self.assertEqual(args.condition_mode, "blind_v3")
        self.assertAlmostEqual(args.v3_local_route_prior, 0.7)
        self.assertTrue(args.v3_fuse_prompt_condition)
        self.assertEqual(args.amp_init_scale, 1024.0)
        protocol_model = small_model(
            v3_fuse_prompt_condition=True,
            v3_local_route_prior=0.7,
        )
        configure_training_protocol(args, protocol_model)
        self.assertEqual(args.best_selection_metric, "avg5_proxy_psnr")
        self.assertEqual(
            args.val_protocol["selection_metric"],
            "val_avg5_proxy_psnr",
        )


if __name__ == "__main__":
    unittest.main()
