import csv
import math
import os
import random
import zlib
from collections import defaultdict

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset, Sampler

from utils.context import prepare_routing_context


TASK_NAMES = {0: "derain", 1: "dehaze", 2: "denoise"}
VALID_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
)


def _list_images(directory):
    """Return normalized paths relative to ``directory`` in stable order."""
    if not os.path.isdir(directory):
        return []
    paths = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.lower().endswith(VALID_EXTENSIONS):
                path = os.path.relpath(os.path.join(root, filename), directory)
                paths.append(path.replace(os.sep, "/"))
    return sorted(paths)


def _stem(path):
    return os.path.splitext(path.replace("\\", "/"))[0]


class AllInOneDataset(Dataset):
    """Indexed multi-task restoration dataset.

    Every index maps to a concrete sample. Task balancing is handled by
    ``BalancedTaskBatchSampler`` instead of random sampling inside
    ``__getitem__``.
    """

    def __init__(
        self,
        data_root,
        patch_size=256,
        train=True,
        noise_levels=(15, 25, 50),
        manifest_path="",
        noise_seed=0,
        return_context=False,
        context_size=256,
        full_image=False,
    ):
        super().__init__()
        self.data_root = data_root
        self.patch_size = patch_size
        self.train = train
        self.noise_levels = tuple(noise_levels)
        self.manifest_path = manifest_path
        self.noise_seed = int(noise_seed)
        self.return_context = bool(return_context)
        self.context_size = int(context_size)
        self.full_image = bool(full_image)
        if self.context_size <= 0:
            raise ValueError("context_size must be positive")
        if self.full_image and self.train:
            raise ValueError("full_image is only supported for validation (train=False)")

        self.task_dirs = {
            0: os.path.join(data_root, "derain"),
            1: os.path.join(data_root, "dehaze"),
            2: os.path.join(data_root, "denoise"),
        }
        if manifest_path:
            self.task_entries = self._load_manifest(manifest_path)
        else:
            self.task_entries = {
                0: [
                    (input_name, target_name, None)
                    for input_name, target_name in self._get_valid_pairs(
                        self.task_dirs[0], "derain"
                    )
                ],
                1: [
                    (input_name, target_name, None)
                    for input_name, target_name in self._get_valid_pairs(
                        self.task_dirs[1], "dehaze"
                    )
                ],
                2: [
                    (input_name, target_name, None)
                    for input_name, target_name in self._get_valid_pairs(
                        self.task_dirs[2], "denoise"
                    )
                ],
            }
        self.task_pairs = {
            task_id: [(entry[0], entry[1]) for entry in entries]
            for task_id, entries in self.task_entries.items()
        }

        self.samples = []
        self.task_indices = defaultdict(list)
        for task_id in sorted(self.task_entries):
            for input_name, target_name, fixed_sigma in self.task_entries[task_id]:
                index = len(self.samples)
                self.samples.append(
                    (task_id, input_name, target_name, fixed_sigma)
                )
                self.task_indices[task_id].append(index)

        counts = {TASK_NAMES[key]: len(value) for key, value in self.task_pairs.items()}
        print(
            "Dataset loaded:",
            ", ".join("{}={}".format(name, count) for name, count in counts.items()),
        )
        if not self.samples:
            raise RuntimeError("No valid image pairs found under {!r}".format(data_root))

    def _load_manifest(self, manifest_path):
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError("Manifest not found: {}".format(manifest_path))
        name_to_id = {name: task_id for task_id, name in TASK_NAMES.items()}
        entries = {task_id: [] for task_id in TASK_NAMES}
        with open(manifest_path, "r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"task", "target"}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise ValueError(
                    "Manifest must contain task and target columns: {}".format(
                        manifest_path
                    )
                )
            for row_number, row in enumerate(reader, start=2):
                task_value = row["task"].strip().lower()
                if task_value in name_to_id:
                    task_id = name_to_id[task_value]
                else:
                    try:
                        task_id = int(task_value)
                    except ValueError as error:
                        raise ValueError(
                            "Invalid task at {}:{}".format(
                                manifest_path,
                                row_number,
                            )
                        ) from error
                if task_id not in TASK_NAMES:
                    raise ValueError(
                        "Task ID out of range at {}:{}".format(
                            manifest_path,
                            row_number,
                        )
                    )
                input_name = (row.get("input") or "").strip().replace("\\", "/")
                target_name = row["target"].strip().replace("\\", "/")
                if not target_name:
                    raise ValueError(
                        "Empty target at {}:{}".format(manifest_path, row_number)
                    )
                sigma_text = (row.get("sigma") or "").strip()
                fixed_sigma = int(sigma_text) if sigma_text else None
                if fixed_sigma is not None and fixed_sigma <= 0:
                    raise ValueError(
                        "Sigma must be positive at {}:{}".format(
                            manifest_path,
                            row_number,
                        )
                    )
                target_path = os.path.join(
                    self.task_dirs[task_id],
                    "target",
                    target_name,
                )
                if not os.path.isfile(target_path):
                    raise FileNotFoundError(
                        "Manifest target not found: {}".format(target_path)
                    )
                if task_id != 2 and not input_name:
                    raise ValueError(
                        "Non-denoising row needs an input at {}:{}".format(
                            manifest_path,
                            row_number,
                        )
                    )
                if input_name:
                    input_path = os.path.join(
                        self.task_dirs[task_id],
                        "input",
                        input_name,
                    )
                    if task_id != 2 and not os.path.isfile(input_path):
                        raise FileNotFoundError(
                            "Manifest input not found: {}".format(input_path)
                        )
                entries[task_id].append(
                    (input_name, target_name, fixed_sigma)
                )
        return entries

    @property
    def rain_pairs(self):
        return self.task_pairs[0]

    @property
    def haze_pairs(self):
        return self.task_pairs[1]

    @property
    def denoise_pairs(self):
        return self.task_pairs[2]

    @property
    def rain_dir(self):
        return self.task_dirs[0]

    @property
    def haze_dir(self):
        return self.task_dirs[1]

    @property
    def denoise_dir(self):
        return self.task_dirs[2]

    def _get_valid_pairs(self, task_dir, task_type):
        input_dir = os.path.join(task_dir, "input")
        target_dir = os.path.join(task_dir, "target")
        if not os.path.isdir(target_dir):
            return []

        target_files = _list_images(target_dir)
        target_by_stem = {}
        for filename in target_files:
            relative_stem = _stem(filename)
            basename_stem = os.path.basename(relative_stem)
            # Prefer the full relative path (for example ``ots/0001``) so
            # identically named images from different datasets never collide.
            target_by_stem[relative_stem] = filename
            target_by_stem.setdefault(basename_stem, filename)

        # Clean-only denoising datasets are supported in both training and
        # validation. Validation noise is generated deterministically.
        if task_type == "denoise" and (
            self.train
            or not os.path.isdir(input_dir)
            or not _list_images(input_dir)
        ):
            return [(filename, filename) for filename in target_files]

        if not os.path.isdir(input_dir):
            return []

        input_files = _list_images(input_dir)
        valid_pairs = []
        for input_filename in input_files:
            input_stem = _stem(input_filename)
            parent = os.path.dirname(input_stem).replace("\\", "/")
            basename_stem = os.path.basename(input_stem)
            candidates = [input_stem, basename_stem]

            sidd_stem = basename_stem.replace("_NOISY_", "_GT_")
            if sidd_stem != basename_stem:
                if parent:
                    candidates.append(parent + "/" + sidd_stem)
                candidates.append(sidd_stem)

            if "_" in basename_stem:
                clean_stem = basename_stem.split("_", 1)[0]
                if parent:
                    candidates.append(parent + "/" + clean_stem)
                candidates.append(clean_stem)

            if basename_stem.startswith("rain-"):
                clean_stem = basename_stem[len("rain-") :]
                candidates.extend([clean_stem, "norain-" + clean_stem])
            elif basename_stem.startswith("rain"):
                clean_stem = basename_stem[len("rain") :]
                candidates.extend([clean_stem, "norain" + clean_stem])

            target_filename = next(
                (
                    target_by_stem[candidate]
                    for candidate in candidates
                    if candidate in target_by_stem
                ),
                None,
            )
            if target_filename is not None:
                valid_pairs.append((input_filename, target_filename))

        return valid_pairs

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _pad_to_patch(image, patch_size):
        width, height = image.size
        pad_width = max(0, patch_size - width)
        pad_height = max(0, patch_size - height)
        if pad_width == 0 and pad_height == 0:
            return image
        left = pad_width // 2
        top = pad_height // 2
        padding_mode = (
            "reflect"
            if width > max(left, pad_width - left)
            and height > max(top, pad_height - top)
            else "edge"
        )
        return TF.pad(
            image,
            [left, top, pad_width - left, pad_height - top],
            padding_mode=padding_mode,
        )

    def _paired_crop_and_augment(self, image_input, image_target):
        if image_input.size != image_target.size:
            raise ValueError(
                "Training pair has mismatched geometry: input={} target={}".format(
                    image_input.size, image_target.size
                )
            )

        image_input = self._pad_to_patch(image_input, self.patch_size)
        image_target = self._pad_to_patch(image_target, self.patch_size)
        width, height = image_input.size
        top = torch.randint(0, height - self.patch_size + 1, (1,)).item()
        left = torch.randint(0, width - self.patch_size + 1, (1,)).item()
        image_input = TF.crop(
            image_input, top, left, self.patch_size, self.patch_size
        )
        image_target = TF.crop(
            image_target, top, left, self.patch_size, self.patch_size
        )

        if random.random() < 0.5:
            image_input = TF.hflip(image_input)
            image_target = TF.hflip(image_target)
        if random.random() < 0.5:
            image_input = TF.vflip(image_input)
            image_target = TF.vflip(image_target)
        if random.random() < 0.5:
            image_input = TF.rotate(image_input, 90)
            image_target = TF.rotate(image_target, 90)
        return image_input, image_target

    @staticmethod
    def _pad_tensor_to_patch(image, patch_size):
        """Pad a CHW tensor safely, including one-pixel-wide images."""
        height, width = image.shape[-2:]
        pad_width = max(0, patch_size - width)
        pad_height = max(0, patch_size - height)
        if not pad_width and not pad_height:
            return image
        left, top = pad_width // 2, pad_height // 2
        right, bottom = pad_width - left, pad_height - top
        mode = (
            "reflect"
            if width > max(left, right) and height > max(top, bottom)
            else "replicate"
        )
        return F.pad(image, (left, right, top, bottom), mode=mode)

    def _getitem_with_full_input(self, index):
        """Build crop and context from one degraded, original-resolution image.

        The legacy three-item crop path below is intentionally left unchanged.
        In this opt-in path, synthetic noise is generated *before* cropping so
        global routing never sees either clean ground truth or a second noise
        realization. Full-image validation uses the same seeded realization.
        """
        task_id, input_name, target_name, fixed_sigma = self.samples[index]
        task_dir = self.task_dirs[task_id]
        target_path = os.path.join(task_dir, "target", target_name)
        with Image.open(target_path) as image:
            full_target = TF.to_tensor(image.convert("RGB"))

        input_path = os.path.join(task_dir, "input", input_name)
        synthetic_denoising = task_id == 2 and (
            self.train or not input_name or not os.path.isfile(input_path)
        )
        if synthetic_denoising:
            sigma_value = fixed_sigma
            if sigma_value is None:
                sigma_value = (
                    random.choice(self.noise_levels)
                    if self.train
                    else self.noise_levels[
                        zlib.crc32(target_name.encode("utf-8"))
                        % len(self.noise_levels)
                    ]
                )
            if self.train:
                noise = torch.randn_like(full_target)
            else:
                generator = torch.Generator()
                sample_seed = (
                    self.noise_seed
                    + zlib.crc32(
                        "{}:{}".format(target_name, sigma_value).encode("utf-8")
                    )
                ) % (2**63 - 1)
                generator.manual_seed(sample_seed)
                noise = torch.randn(
                    full_target.shape,
                    dtype=full_target.dtype,
                    generator=generator,
                )
            full_input = torch.clamp(
                full_target + noise * (sigma_value / 255.0), 0.0, 1.0
            )
        else:
            with Image.open(input_path) as image:
                full_input = TF.to_tensor(image.convert("RGB"))

        if full_input.shape != full_target.shape:
            raise ValueError(
                "Image pair has mismatched geometry: input={} target={} ({})".format(
                    tuple(full_input.shape), tuple(full_target.shape), target_path
                )
            )

        if self.full_image:
            input_tensor, target_tensor = full_input, full_target
        else:
            input_tensor = self._pad_tensor_to_patch(full_input, self.patch_size)
            target_tensor = self._pad_tensor_to_patch(full_target, self.patch_size)
            if self.train:
                height, width = input_tensor.shape[-2:]
                top = torch.randint(0, height - self.patch_size + 1, (1,)).item()
                left = torch.randint(0, width - self.patch_size + 1, (1,)).item()
                input_tensor = input_tensor[
                    :, top : top + self.patch_size, left : left + self.patch_size
                ]
                target_tensor = target_tensor[
                    :, top : top + self.patch_size, left : left + self.patch_size
                ]
            else:
                input_tensor = TF.center_crop(input_tensor, self.patch_size)
                target_tensor = TF.center_crop(target_tensor, self.patch_size)

        if self.train:
            # Apply the same sampled orientation to local input, local target
            # and original full degraded view. rot90 swaps H/W without cropping.
            for dimension in (-1, -2):
                if random.random() < 0.5:
                    input_tensor = torch.flip(input_tensor, (dimension,))
                    target_tensor = torch.flip(target_tensor, (dimension,))
                    full_input = torch.flip(full_input, (dimension,))
            if random.random() < 0.5:
                input_tensor = torch.rot90(input_tensor, 1, (-2, -1))
                target_tensor = torch.rot90(target_tensor, 1, (-2, -1))
                full_input = torch.rot90(full_input, 1, (-2, -1))

        sample = (
            input_tensor,
            target_tensor,
            torch.tensor(task_id, dtype=torch.long),
        )
        if self.return_context:
            context = prepare_routing_context(
                full_input.unsqueeze(0), self.context_size
            ).squeeze(0)
            return sample + (context,)
        return sample

    def __getitem__(self, index):
        if self.return_context or self.full_image:
            return self._getitem_with_full_input(index)

        task_id, input_name, target_name, fixed_sigma = self.samples[index]
        task_dir = self.task_dirs[task_id]
        target_path = os.path.join(task_dir, "target", target_name)
        with Image.open(target_path) as image:
            image_target = image.convert("RGB")

        input_path = os.path.join(task_dir, "input", input_name)
        synthetic_denoising = task_id == 2 and (
            self.train or not input_name or not os.path.isfile(input_path)
        )
        if synthetic_denoising:
            image_input = image_target.copy()
        else:
            with Image.open(input_path) as image:
                image_input = image.convert("RGB")

        if self.train:
            image_input, image_target = self._paired_crop_and_augment(
                image_input, image_target
            )
        else:
            image_input = self._pad_to_patch(image_input, self.patch_size)
            image_target = self._pad_to_patch(image_target, self.patch_size)
            image_input = TF.center_crop(
                image_input, [self.patch_size, self.patch_size]
            )
            image_target = TF.center_crop(
                image_target, [self.patch_size, self.patch_size]
            )

        input_tensor = TF.to_tensor(image_input)
        target_tensor = TF.to_tensor(image_target)
        if synthetic_denoising:
            sigma_value = fixed_sigma
            if sigma_value is None:
                if self.train:
                    sigma_value = random.choice(self.noise_levels)
                else:
                    sigma_value = self.noise_levels[
                        zlib.crc32(target_name.encode("utf-8"))
                        % len(self.noise_levels)
                    ]
            if self.train:
                noise = torch.randn_like(input_tensor)
            else:
                generator = torch.Generator()
                sample_seed = (
                    self.noise_seed
                    + zlib.crc32(
                        "{}:{}".format(target_name, sigma_value).encode("utf-8")
                    )
                ) % (2**63 - 1)
                generator.manual_seed(sample_seed)
                noise = torch.randn(
                    input_tensor.shape,
                    dtype=input_tensor.dtype,
                    generator=generator,
                )
            sigma = sigma_value / 255.0
            input_tensor = torch.clamp(
                input_tensor + noise * sigma, 0.0, 1.0
            )

        return input_tensor, target_tensor, torch.tensor(task_id, dtype=torch.long)


class BalancedTaskBatchSampler(Sampler):
    """Yield deterministic, balanced batches while cycling smaller tasks."""

    def __init__(
        self,
        dataset,
        batch_size,
        seed=42,
        drop_last=False,
        samples_per_epoch=None,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        self.task_ids = sorted(
            task_id
            for task_id, indices in dataset.task_indices.items()
            if indices
        )
        if not self.task_ids:
            raise ValueError("Dataset contains no non-empty tasks")
        if samples_per_epoch is None:
            samples_per_epoch = (
                max(len(dataset.task_indices[task_id]) for task_id in self.task_ids)
                * len(self.task_ids)
            )
        if samples_per_epoch <= 0:
            raise ValueError("samples_per_epoch must be positive")
        self.samples_per_epoch = int(samples_per_epoch)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        if self.drop_last:
            return self.samples_per_epoch // self.batch_size
        return int(math.ceil(self.samples_per_epoch / self.batch_size))

    def __iter__(self):
        generator = random.Random(self.seed + self.epoch)
        pools = {}
        positions = {}
        for task_id in self.task_ids:
            pools[task_id] = list(self.dataset.task_indices[task_id])
            generator.shuffle(pools[task_id])
            positions[task_id] = 0

        task_cycle = list(self.task_ids)
        generator.shuffle(task_cycle)
        offset = self.epoch % len(task_cycle)
        schedule = [
            task_cycle[(index + offset) % len(task_cycle)]
            for index in range(self.samples_per_epoch)
        ]

        batch = []
        for task_id in schedule:
            if positions[task_id] >= len(pools[task_id]):
                generator.shuffle(pools[task_id])
                positions[task_id] = 0
            batch.append(pools[task_id][positions[task_id]])
            positions[task_id] += 1
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch
