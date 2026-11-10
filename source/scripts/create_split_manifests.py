"""Create scene-disjoint train/validation manifests without moving images."""

import argparse
import csv
import json
import os
import random
import sys
import zlib
from collections import defaultdict
from datetime import datetime


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset import AllInOneDataset, TASK_NAMES  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Split every task by clean target so degradation variants of the "
            "same scene never cross train/validation boundaries."
        )
    )
    parser.add_argument("--data_root", default="data/train")
    parser.add_argument("--output_dir", default="splits/aio3_seed42")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--denoise_sigmas", type=int, nargs="+", default=(15, 25, 50))
    parser.add_argument(
        "--max_val_variants_per_target",
        type=int,
        default=3,
        help="Cap paired degradation variants per validation clean target.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _split_targets(samples, val_ratio, seed):
    target_groups = defaultdict(list)
    for sample in samples:
        task_id, input_name, target_name, _ = sample
        target_groups[(task_id, target_name)].append(
            (task_id, input_name, target_name)
        )

    train_rows = []
    validation_rows = []
    group_summary = {}
    for task_id, task_name in TASK_NAMES.items():
        keys = sorted(key for key in target_groups if key[0] == task_id)
        generator = random.Random(seed + task_id)
        generator.shuffle(keys)
        if len(keys) <= 1:
            validation_count = 0
        else:
            validation_count = min(
                len(keys) - 1,
                max(1, round(len(keys) * val_ratio)),
            )
        validation_keys = set(keys[:validation_count])
        for key in keys:
            destination = validation_rows if key in validation_keys else train_rows
            destination.extend(target_groups[key])
        group_summary[task_name] = {
            "total_clean_targets": len(keys),
            "train_clean_targets": len(keys) - validation_count,
            "val_clean_targets": validation_count,
        }
    return train_rows, validation_rows, group_summary


def _manifest_rows(rows, validation, denoise_sigmas):
    output = []
    for task_id, input_name, target_name in sorted(
        rows,
        key=lambda row: (row[0], row[2], row[1]),
    ):
        if task_id == 2:
            sigmas = denoise_sigmas if validation else (None,)
            for sigma in sigmas:
                output.append(
                    {
                        "task": TASK_NAMES[task_id],
                        "input": "",
                        "target": target_name,
                        "sigma": "" if sigma is None else sigma,
                    }
                )
        else:
            output.append(
                {
                    "task": TASK_NAMES[task_id],
                    "input": input_name,
                    "target": target_name,
                    "sigma": "",
                }
            )
    return output


def _limit_validation_variants(rows, maximum, seed):
    if maximum <= 0:
        return rows
    groups = defaultdict(list)
    for row in rows:
        groups[(row[0], row[2])].append(row)
    limited = []
    for key in sorted(groups):
        candidates = sorted(groups[key], key=lambda row: row[1])
        if key[0] != 2 and len(candidates) > maximum:
            group_seed = seed + zlib.crc32(
                "{}:{}".format(key[0], key[1]).encode("utf-8")
            )
            generator = random.Random(group_seed)
            candidates = sorted(generator.sample(candidates, maximum))
        limited.extend(candidates)
    return limited


def _write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("task", "input", "target", "sigma"),
        )
        writer.writeheader()
        writer.writerows(rows)


def _count_rows(rows):
    counts = {name: 0 for name in TASK_NAMES.values()}
    for row in rows:
        counts[row["task"]] += 1
    return counts


def main():
    args = parse_args()
    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val_ratio must be between 0 and 1")
    if any(sigma <= 0 for sigma in args.denoise_sigmas):
        raise ValueError("All denoising sigmas must be positive")
    output_dir = os.path.abspath(args.output_dir)
    train_path = os.path.join(output_dir, "train.csv")
    validation_path = os.path.join(output_dir, "val.csv")
    summary_path = os.path.join(output_dir, "summary.json")
    existing = [path for path in (train_path, validation_path, summary_path) if os.path.exists(path)]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Split outputs already exist; pass --overwrite to replace them: {}".format(
                existing
            )
        )

    dataset = AllInOneDataset(args.data_root, train=True)
    train_samples, validation_samples, group_summary = _split_targets(
        dataset.samples,
        args.val_ratio,
        args.seed,
    )
    train_rows = _manifest_rows(
        train_samples,
        validation=False,
        denoise_sigmas=args.denoise_sigmas,
    )
    validation_rows = _manifest_rows(
        _limit_validation_variants(
            validation_samples,
            args.max_val_variants_per_target,
            args.seed,
        ),
        validation=True,
        denoise_sigmas=args.denoise_sigmas,
    )
    train_targets = {(row["task"], row["target"]) for row in train_rows}
    validation_targets = {
        (row["task"], row["target"]) for row in validation_rows
    }
    overlap = train_targets & validation_targets
    if overlap:
        raise RuntimeError("Target leakage detected: {}".format(sorted(overlap)[:5]))

    os.makedirs(output_dir, exist_ok=True)
    _write_csv(train_path, train_rows)
    _write_csv(validation_path, validation_rows)
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_root": os.path.abspath(args.data_root),
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "denoise_sigmas": list(args.denoise_sigmas),
        "max_val_variants_per_target": args.max_val_variants_per_target,
        "target_overlap_count": 0,
        "clean_target_groups": group_summary,
        "train_rows": _count_rows(train_rows),
        "val_rows": _count_rows(validation_rows),
        "train_manifest": train_path,
        "val_manifest": validation_path,
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
