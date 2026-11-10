import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
import torchvision.transforms.functional as TF
from PIL import Image

from dataset import AllInOneDataset, BalancedTaskBatchSampler
from utils.common import calc_psnr, calc_ssim, psnr_per_image
from utils.context import prepare_routing_context


class FakeDataset:
    task_indices = {
        0: [0, 1],
        1: [2, 3, 4, 5],
        2: [6],
    }


class CoreTests(unittest.TestCase):
    @staticmethod
    def _write_image_pair(root, image_input, image_target):
        for role, image in (("input", image_input), ("target", image_target)):
            directory = root / "dehaze" / role
            directory.mkdir(parents=True, exist_ok=True)
            image.save(directory / "scene.png")

    def test_metrics_are_per_image(self):
        target = torch.zeros(2, 3, 16, 16)
        restored = target.clone()
        restored[1] += 0.1
        values = psnr_per_image(restored, target)
        self.assertEqual(tuple(values.shape), (2,))
        self.assertGreater(values[0].item(), values[1].item())
        self.assertAlmostEqual(calc_psnr(restored, target), values.mean().item())

    def test_identical_ssim(self):
        image = torch.rand(2, 3, 16, 16)
        self.assertAlmostEqual(calc_ssim(image, image), 1.0, places=5)

    def test_balanced_sampler_is_deterministic(self):
        first = list(BalancedTaskBatchSampler(FakeDataset(), 4, seed=7))
        second = list(BalancedTaskBatchSampler(FakeDataset(), 4, seed=7))
        self.assertEqual(first, second)
        flattened = [index for batch in first for index in batch]
        self.assertEqual(len(flattened), 12)
        index_to_task = {
            index: task_id
            for task_id, indices in FakeDataset.task_indices.items()
            for index in indices
        }
        for batch in first:
            counts = [
                sum(index_to_task[index] == task_id for index in batch)
                for task_id in FakeDataset.task_indices
            ]
            self.assertLessEqual(max(counts) - min(counts), 1)

    def test_balanced_sampler_accepts_fixed_epoch_size(self):
        sampler = BalancedTaskBatchSampler(
            FakeDataset(),
            batch_size=4,
            seed=7,
            samples_per_epoch=20,
        )
        batches = list(sampler)
        self.assertEqual(len(batches), 5)
        self.assertEqual(sum(len(batch) for batch in batches), 20)

    def test_nested_ots_pairing_and_clean_only_denoising(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            for task in ("derain", "dehaze", "denoise"):
                (root / task / "input").mkdir(parents=True)
                (root / task / "target").mkdir(parents=True)
            (root / "dehaze" / "input" / "ots").mkdir()
            (root / "dehaze" / "target" / "ots").mkdir()
            (root / "denoise" / "target" / "wed").mkdir()
            image = Image.new("RGB", (32, 32), color=(64, 96, 128))
            image.save(root / "dehaze" / "input" / "ots" / "0001_0.8_0.2.jpg")
            image.save(root / "dehaze" / "target" / "ots" / "0001.jpg")
            image.save(root / "denoise" / "target" / "wed" / "wed_0001.bmp")

            dataset = AllInOneDataset(
                str(root),
                patch_size=16,
                train=True,
            )
            self.assertEqual(len(dataset.haze_pairs), 1)
            self.assertEqual(len(dataset.denoise_pairs), 1)
            self.assertEqual(
                dataset.haze_pairs[0],
                ("ots/0001_0.8_0.2.jpg", "ots/0001.jpg"),
            )

    def test_manifest_denoising_is_deterministic(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            target_dir = root / "denoise" / "target"
            target_dir.mkdir(parents=True)
            Image.new("RGB", (20, 20), color=(64, 96, 128)).save(
                target_dir / "clean.png"
            )
            manifest = root / "val.csv"
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=("task", "input", "target", "sigma"),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "task": "denoise",
                        "input": "",
                        "target": "clean.png",
                        "sigma": 25,
                    }
                )
            first = AllInOneDataset(
                str(root),
                patch_size=16,
                train=False,
                manifest_path=str(manifest),
                noise_seed=9,
            )[0]
            second = AllInOneDataset(
                str(root),
                patch_size=16,
                train=False,
                manifest_path=str(manifest),
                noise_seed=9,
            )[0]
            self.assertTrue(torch.equal(first[0], second[0]))
            self.assertTrue(torch.equal(first[1], second[1]))
            self.assertFalse(torch.equal(first[0], first[1]))

    def test_tiny_images_use_safe_padding(self):
        image = Image.new("RGB", (2, 3), color=(10, 20, 30))
        padded = AllInOneDataset._pad_to_patch(image, 16)
        self.assertEqual(padded.size, (16, 16))

    def test_context_is_opt_in_and_full_validation_keeps_original_size(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_input = Image.new("RGB", (30, 20), color=(60, 80, 100))
            image_target = Image.new("RGB", (30, 20), color=(20, 40, 60))
            self._write_image_pair(root, image_input, image_target)
            for train in (True, False):
                legacy = AllInOneDataset(str(root), patch_size=16, train=train)[0]
                self.assertEqual(len(legacy), 3)
                self.assertEqual(tuple(legacy[0].shape), (3, 16, 16))

            for return_context in (False, True):
                full = AllInOneDataset(
                    str(root), patch_size=16, train=False,
                    full_image=True, return_context=return_context, context_size=24,
                )[0]
                self.assertEqual(len(full), 4 if return_context else 3)
                self.assertTrue(torch.equal(full[0], TF.to_tensor(image_input)))
                self.assertTrue(torch.equal(full[1], TF.to_tensor(image_target)))
                if return_context:
                    self.assertEqual(tuple(full[3].shape), (3, 24, 24))
            with self.assertRaisesRegex(ValueError, "train=False"):
                AllInOneDataset(str(root), train=True, full_image=True)

    def test_context_uses_whole_degraded_scene_instead_of_crop_or_target(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_input = Image.new("RGB", (48, 16), color=(0, 0, 0))
            image_input.paste((255, 0, 0), (0, 0, 16, 16))
            image_input.paste((0, 255, 0), (32, 0, 48, 16))
            image_target = Image.new("RGB", (48, 16), color=(0, 0, 0))
            self._write_image_pair(root, image_input, image_target)
            crop, _, task, context = AllInOneDataset(
                str(root), patch_size=16, train=False,
                return_context=True, context_size=48,
            )[0]
            expected = prepare_routing_context(
                TF.to_tensor(image_input).unsqueeze(0), 48
            ).squeeze(0)
            self.assertEqual(task.item(), 1)
            self.assertEqual(torch.count_nonzero(crop).item(), 0)
            self.assertTrue(torch.equal(context, expected))
            self.assertGreater(torch.count_nonzero(context).item(), 0)
            self.assertFalse(torch.equal(
                context, prepare_routing_context(crop.unsqueeze(0), 48).squeeze(0)
            ))

    def test_context_and_pair_share_augmentation_without_cropping_full_rotation(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            pattern = torch.arange(3 * 12 * 20).reshape(3, 12, 20).float() / 719
            image_input = TF.to_pil_image(pattern)
            image_target = TF.to_pil_image(1 - pattern)
            self._write_image_pair(root, image_input, image_target)
            dataset = AllInOneDataset(
                str(root), patch_size=8, train=True,
                return_context=True, context_size=20,
            )
            with patch("dataset.torch.randint", side_effect=[
                torch.tensor([1]), torch.tensor([3]),
            ]), patch("dataset.random.random", side_effect=[0.1, 0.1, 0.1]):
                crop, target, _, context = dataset[0]

            def orient(tensor):
                return torch.rot90(torch.flip(tensor, (-1, -2)), 1, (-2, -1))

            full_input = TF.to_tensor(image_input)
            full_target = TF.to_tensor(image_target)
            self.assertTrue(torch.equal(crop, orient(full_input[:, 1:9, 3:11])))
            self.assertTrue(torch.equal(target, orient(full_target[:, 1:9, 3:11])))
            rotated_full = orient(full_input)
            self.assertEqual(tuple(rotated_full.shape[-2:]), (20, 12))
            self.assertTrue(torch.equal(
                context, prepare_routing_context(rotated_full.unsqueeze(0), 20).squeeze(0)
            ))

    def test_training_denoising_crop_and_context_share_one_noise_realization(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "denoise" / "target"
            directory.mkdir(parents=True)
            Image.new("RGB", (32, 24), color=(128, 128, 128)).save(directory / "clean.png")
            dataset = AllInOneDataset(
                str(root), patch_size=16, train=True,
                return_context=True, context_size=32,
            )
            with patch("dataset.random.choice", return_value=25) as choose_sigma, patch(
                "dataset.torch.randint", side_effect=[torch.tensor([3]), torch.tensor([5])]
            ), patch("dataset.random.random", return_value=1.0):
                crop, target, task, context = dataset[0]
            choose_sigma.assert_called_once()
            self.assertEqual(task.item(), 2)
            # No resize is needed at this size; the original degraded image
            # occupies the upper-left 24x32 region of the padded global view.
            self.assertTrue(torch.equal(crop, context[:, 3:19, 5:21]))
            self.assertFalse(torch.equal(crop, target))

    def test_validation_denoising_full_crop_and_context_are_reproducible(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "denoise" / "target"
            directory.mkdir(parents=True)
            Image.new("RGB", (32, 24), color=(128, 128, 128)).save(directory / "clean.png")
            manifest = root / "val.csv"
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=("task", "input", "target", "sigma"))
                writer.writeheader()
                writer.writerow({"task": "denoise", "input": "", "target": "clean.png", "sigma": 25})
            options = dict(
                data_root=str(root), patch_size=16, train=False,
                manifest_path=str(manifest), noise_seed=9,
                return_context=True, context_size=32,
            )
            full = AllInOneDataset(**options, full_image=True)[0]
            dataset = AllInOneDataset(**options)
            first, second = dataset[0], dataset[0]
            self.assertTrue(torch.equal(first[0], TF.center_crop(full[0], 16)))
            self.assertTrue(torch.equal(first[1], TF.center_crop(full[1], 16)))
            self.assertTrue(torch.equal(first[3], full[3]))
            self.assertTrue(torch.equal(first[0], second[0]))
            self.assertTrue(torch.equal(first[3], second[3]))
            self.assertTrue(torch.equal(
                full[3], prepare_routing_context(full[0].unsqueeze(0), 32).squeeze(0)
            ))
            self.assertFalse(torch.equal(
                full[3], prepare_routing_context(full[1].unsqueeze(0), 32).squeeze(0)
            ))
            different_seed = AllInOneDataset(**dict(options, noise_seed=10))[0]
            self.assertFalse(torch.equal(first[0], different_seed[0]))

    def test_context_checks_pair_geometry_and_pads_tiny_images(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            tiny = Image.new("RGB", (1, 2), color=(20, 40, 60))
            self._write_image_pair(root, tiny, tiny)
            sample = AllInOneDataset(
                str(root), patch_size=16, train=True,
                return_context=True, context_size=8,
            )[0]
            self.assertEqual(tuple(sample[0].shape), (3, 16, 16))
            self.assertEqual(tuple(sample[3].shape), (3, 8, 8))
            mismatched = Image.new("RGB", (2, 2), color=(20, 40, 60))
            self._write_image_pair(root, tiny, mismatched)
            for train in (False, True):
                dataset = AllInOneDataset(
                    str(root), patch_size=16, train=train,
                    return_context=True, context_size=8,
                )
                with self.assertRaisesRegex(ValueError, "mismatched geometry"):
                    dataset[0]


if __name__ == "__main__":
    unittest.main()
