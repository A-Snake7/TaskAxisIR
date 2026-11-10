import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from net.promamba import ProMamba
from train import (
    configure_training_protocol, parse_args, set_blind_trainable,
    snapshot_data_protocol, validate,
)
from utils.checkpoint import load_model_for_inference
from utils.context import prepare_routing_context
from utils.inference import tiled_inference
from utils.losses import RestorationLoss, blind_v2_routing_loss


def small_model(mode="blind_v2"):
    # Exercise actual conditioning/decoder code on CPU. The GPU smoke check
    # separately covers real Mamba kernels and AMP at the training resolution.
    model = ProMamba(
        architecture="hierarchical", dim=8, stage_blocks=(1, 1, 1),
        bottleneck_type="conv", condition_mode=mode,
        routing_context_size=32, local_condition_scale=0.2,
    )
    # Avoid vacuous comparisons caused by the model's identity initialization.
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) and (
                name.endswith("output") or "task_heads" in name
            ):
                nn.init.normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)
    return model


class BlindV2Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)

    def test_context_preprocessing_is_idempotent_and_handles_tiny_images(self):
        full = torch.rand(2, 3, 40, 72)
        context = prepare_routing_context(full, 32)
        self.assertEqual(tuple(context.shape), (2, 3, 32, 32))
        self.assertTrue(torch.equal(context, prepare_routing_context(context, 32)))
        tiny = torch.full((1, 3, 1, 2), 0.3)
        self.assertTrue(torch.allclose(prepare_routing_context(tiny, 32), torch.full((1, 3, 32, 32), 0.3)))

    def test_oracle_warm_start_preserves_output_with_teacher_forcing(self):
        oracle = small_model("oracle").eval()
        v2 = small_model().eval()
        missing = v2.load_state_dict(oracle.state_dict(), strict=False).missing_keys
        self.assertTrue(missing)
        self.assertTrue(all(
            any(module in key for module in (
                "degradation_encoder", "degradation_condition", "local_condition_adapter"
            )) for key in missing
        ))
        image = torch.rand(2, 3, 24, 28)
        task = torch.tensor([0, 2])
        with torch.no_grad():
            expected = oracle(image, task)
            actual = v2(image, task, teacher_forcing=1.0, routing_context=torch.rand(2, 3, 32, 32))
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    def test_blind_prediction_ignores_metadata_task_and_cached_context_matches(self):
        model = small_model().eval()
        image = torch.rand(1, 3, 24, 28)
        context = torch.rand(1, 3, 32, 32)
        with torch.no_grad():
            first, aux = model(image, torch.tensor([0]), routing_context=context, return_aux=True)
            second = model(image, torch.tensor([2]), routing_context=context)
            cache = model.predict_task_context(context)
            cached = model(
                image, routing_probs_override=cache["routing_probs"],
                condition_override=cache["condition"],
            )
        torch.testing.assert_close(first, second, atol=0, rtol=0)
        torch.testing.assert_close(first, cached, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(aux["task_probs"], cache["task_probs"])

    def test_local_evidence_changes_gates_but_not_global_heads(self):
        model = small_model().eval()
        adapter = model.hierarchical.local_condition_adapter[-1]
        with torch.no_grad():
            adapter.weight.copy_(torch.eye(8))
        context = torch.rand(1, 3, 32, 32)
        with torch.no_grad():
            _, first = model(torch.rand(1, 3, 24, 24), routing_context=context, return_aux=True)
            _, second = model(torch.rand(1, 3, 24, 24), routing_context=context, return_aux=True)
        torch.testing.assert_close(first["routing_probs"], second["routing_probs"], atol=0, rtol=0)
        torch.testing.assert_close(first["global_condition"], second["global_condition"], atol=0, rtol=0)
        self.assertGreater((first["gate_condition"] - second["gate_condition"]).abs().max().item(), 1e-6)
        self.assertLessEqual((first["gate_condition"] - first["global_condition"]).abs().max().item(), 0.20001)

    def test_tiling_reuses_one_global_context_and_computes_local_each_tile(self):
        model = small_model().eval()
        calls = []
        hook = model.hierarchical.degradation_encoder.register_forward_hook(
            lambda *args: calls.append(1)
        )
        image = torch.rand(1, 3, 40, 48)
        try:
            restored, routing = tiled_inference(
                model, image, tile_size=32, overlap=8, amp_enabled=False,
                return_routing=True,
            )
        finally:
            hook.remove()
        self.assertEqual(restored.shape, image.shape)
        self.assertTrue(torch.isfinite(restored).all())
        self.assertEqual(len(calls), routing["tile_count"] + 1)
        self.assertTrue(routing["global_routing"])
        self.assertIsNotNone(routing["local_global_js"])
        self.assertGreaterEqual(float(routing["local_global_js"]), -1e-7)

    def test_oracle_ablation_keeps_predicted_and_used_probabilities_separate(self):
        model = small_model().eval()
        _, routing = tiled_inference(
            model, torch.rand(1, 3, 32, 32), torch.tensor([2]),
            tile_size=32, overlap=8, amp_enabled=False,
            teacher_forcing=1.0, return_routing=True,
        )
        torch.testing.assert_close(routing["routing_probs"], torch.tensor([[0.0, 0.0, 1.0]]))
        self.assertFalse(torch.equal(routing["task_probs"], routing["routing_probs"]))

    def test_consistency_detaches_global_targets_and_is_zero_for_equal_views(self):
        global_logits = torch.randn(2, 3, requires_grad=True)
        local_logits = torch.randn(2, 3, requires_grad=True)
        global_feature = torch.randn(2, 8, requires_grad=True)
        local_feature = torch.randn(2, 8, requires_grad=True)
        aux = {
            "task_logits": global_logits, "local_task_logits": local_logits,
            "degradation_feature": global_feature, "local_degradation_feature": local_feature,
        }
        terms = blind_v2_routing_loss(aux, torch.tensor([0, 2]))
        (terms["probability_consistency"] + terms["feature_consistency"]).backward()
        self.assertIsNone(global_logits.grad)
        self.assertIsNone(global_feature.grad)
        self.assertGreater(local_logits.grad.abs().sum().item(), 0)
        self.assertGreater(local_feature.grad.abs().sum().item(), 0)
        aux["local_task_logits"] = global_logits.detach().clone()
        aux["local_degradation_feature"] = global_feature.detach().clone()
        terms = blind_v2_routing_loss(aux, torch.tensor([0, 2]))
        self.assertAlmostEqual(float(terms["probability_consistency"]), 0.0, places=6)
        self.assertAlmostEqual(float(terms["feature_consistency"]), 0.0, places=6)

    def test_restoration_and_routing_gradients_reach_new_modules(self):
        model = small_model().train()
        output, aux = model(
            torch.rand(2, 3, 24, 24), routing_context=torch.rand(2, 3, 32, 32),
            return_aux=True,
        )
        restoration, _ = RestorationLoss()(output, torch.rand_like(output))
        terms = blind_v2_routing_loss(aux, torch.tensor([0, 1]))
        loss = (restoration.mean() + 0.1 * terms["combined_cls"]
                + 0.05 * terms["probability_consistency"]
                + 0.01 * terms["feature_consistency"])
        loss.backward()
        for module in (
            model.hierarchical.degradation_encoder.classifier,
            model.hierarchical.degradation_condition[-1],
            model.hierarchical.local_condition_adapter[-1],
        ):
            self.assertIsNotNone(module.weight.grad)
            self.assertTrue(torch.isfinite(module.weight.grad).all())
            self.assertGreater(module.weight.grad.abs().sum().item(), 0)
        set_blind_trainable(model, 1, 3)
        self.assertTrue(model.hierarchical.local_condition_adapter[-1].weight.requires_grad)
        self.assertFalse(model.hierarchical.stem.weight.requires_grad)

    def test_checkpoint_round_trip_restores_v2_configuration(self):
        model = small_model().eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v2.pth"
            torch.save({"model_config": model.model_config, "model_state_dict": model.state_dict()}, path)
            loaded, _, config = load_model_for_inference(str(path), "cpu")
            self.assertEqual(config["condition_mode"], "blind_v2")
            self.assertEqual(config["routing_context_size"], 32)
            image = torch.rand(1, 3, 24, 28)
            with torch.no_grad():
                torch.testing.assert_close(model(image), loaded(image), atol=0, rtol=0)

    def test_resume_restores_v2_validation_and_loss_protocol(self):
        with patch("sys.argv", ["train.py", "--resume", "unused.pth"]):
            args = parse_args()
        model = SimpleNamespace(model_config={
            "condition_mode": "blind_v2", "architecture": "hierarchical",
            "routing_context_size": 64, "local_condition_scale": 0.15,
            "dim": 8, "stage_blocks": (1, 1, 1),
        })
        checkpoint = {"train_args": {
            "val_inference": "tile", "val_tile_size": 48, "val_overlap": 8,
            "routing_consistency_weight": 0.2, "patch_size": 32,
            "data_root": "/verified/clean", "train_manifest": "/verified/train.csv",
            "val_root": "/verified/clean", "val_manifest": "/verified/val.csv",
            "steps_per_epoch": 15, "batch_size": 3,
        }}
        configure_training_protocol(args, model, checkpoint)
        self.assertEqual(args.condition_mode, "blind_v2")
        self.assertEqual(args.routing_context_size, 64)
        self.assertEqual(args.dim, 8)
        self.assertEqual(tuple(args.stage_blocks), (1, 1, 1))
        self.assertEqual(args.val_protocol["batch_size"], 1)
        self.assertEqual(args.val_tile_size, 48)
        self.assertEqual(args.routing_consistency_weight, 0.2)
        self.assertEqual(args.data_root, "/verified/clean")
        self.assertEqual(args.val_manifest, "/verified/val.csv")
        self.assertEqual(args.steps_per_epoch, 15)
        self.assertEqual(args.batch_size, 3)
        args.resume = ""
        with self.assertRaisesRegex(ValueError, "router_warmup_epochs 0"):
            configure_training_protocol(args, model, None)

    def test_v2_data_provenance_rejects_missing_or_changed_manifests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_manifest = root / "train.csv"
            val_manifest = root / "val.csv"
            train_manifest.write_text("task,input,target\nderain,a.png,a.png\n", encoding="utf-8")
            val_manifest.write_text("task,input,target\nderain,b.png,b.png\n", encoding="utf-8")
            args = SimpleNamespace(
                condition_mode="blind_v2", data_root=directory, val_root="",
                train_manifest=str(train_manifest), val_manifest=str(val_manifest),
            )
            protocol = snapshot_data_protocol(args)
            self.assertEqual(args.val_root, str(root.resolve()))
            self.assertEqual(len(protocol["train_manifest"]["sha256"]), 64)
            checkpoint = {"data_protocol": protocol}
            self.assertEqual(snapshot_data_protocol(args, checkpoint), protocol)
            val_manifest.write_text("task,input,target\nderain,c.png,c.png\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Data protocol changed"):
                snapshot_data_protocol(args, checkpoint)
            with self.assertRaisesRegex(ValueError, "provenance"):
                snapshot_data_protocol(args, {})
            args.val_manifest = ""
            with self.assertRaisesRegex(ValueError, "explicit --train_manifest"):
                snapshot_data_protocol(args)

    def test_validation_keeps_old_blind_soft_and_reports_v2_local_accuracy(self):
        image = torch.rand(1, 3, 24, 28)
        loader = [(image, torch.rand_like(image), torch.tensor([1]))]
        baseline = validate(small_model("blind_soft"), loader, torch.device("cpu"), False)
        self.assertIn("val_task_accuracy", baseline)
        self.assertNotIn("val_local_task_accuracy", baseline)
        v2 = validate(
            small_model(), loader, torch.device("cpu"), False,
            inference_mode="tile", tile_size=32, overlap=8,
        )
        for name in ("val_local_task_accuracy", "val_local_global_js", "val_routing_ece"):
            self.assertIn(name, v2)
            self.assertGreaterEqual(v2[name], -1e-7)

    def test_incomplete_context_and_unsupported_architecture_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "hierarchical"):
            ProMamba(architecture="lightweight", condition_mode="blind_v2")
        model = small_model().eval()
        with self.assertRaisesRegex(ValueError, "cached global condition"):
            model(torch.rand(1, 3, 24, 24), routing_probs_override=torch.ones(1, 3))


if __name__ == "__main__":
    unittest.main()
