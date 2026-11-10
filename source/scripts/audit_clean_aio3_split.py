"""Bind scene-disjoint AIO-3 manifests to a passed dataset audit.

The parent data root has already been fully hash-audited by
``audit_clean_aio3_dataset.py``.  This tool verifies that two manifests are
valid disjoint subsets of that root, then records the immutable hashes of the
dataset audit and both manifests in one compact split report.
"""

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


TASKS = {"derain", "dehaze", "denoise"}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scene_id(value):
    return Path(value).stem.split("_", 1)[0]


def load_manifest(path, data_root, require_denoise_sigma):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError("Manifest not found: {}".format(path))
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"task", "input", "target", "sigma"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("Invalid manifest columns in {}".format(path))
        for number, row in enumerate(reader, start=2):
            task = (row.get("task") or "").strip().lower()
            input_name = (row.get("input") or "").strip().replace("\\", "/")
            target_name = (row.get("target") or "").strip().replace("\\", "/")
            sigma = (row.get("sigma") or "").strip()
            if task not in TASKS or not target_name:
                raise ValueError("Invalid row at {}:{}".format(path, number))
            target_path = data_root / task / "target" / target_name
            if not target_path.is_file():
                raise FileNotFoundError("Missing target at {}:{}".format(path, number))
            if task == "denoise":
                if input_name:
                    raise ValueError("Denoise input must be blank at {}:{}".format(path, number))
                if require_denoise_sigma and not sigma:
                    raise ValueError("Denoise validation/training row needs sigma at {}:{}".format(path, number))
            else:
                if not input_name:
                    raise ValueError("Missing input at {}:{}".format(path, number))
                input_path = data_root / task / "input" / input_name
                if not input_path.is_file():
                    raise FileNotFoundError("Missing input at {}:{}".format(path, number))
            rows.append(
                {
                    "task": task,
                    "input": input_name,
                    "target": target_name,
                    "sigma": sigma,
                }
            )
    return rows, sha256_file(path)


def test_haze_scene_ids(test_root):
    folder = test_root / "dehaze" / "input"
    if not folder.is_dir():
        raise FileNotFoundError("Missing dehaze test input folder: {}".format(folder))
    return {
        scene_id(path.name)
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }


def count_tasks(rows):
    return dict(sorted(Counter(row["task"] for row in rows).items()))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify clean AIO-3 train/validation manifests and record their audit chain."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--test-root", required=True)
    parser.add_argument("--dataset-audit", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    test_root = Path(args.test_root).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("Split audit output already exists: {}".format(output))

    dataset_audit_path = Path(args.dataset_audit).resolve()
    dataset_audit = json.loads(dataset_audit_path.read_text(encoding="utf-8"))
    if dataset_audit.get("status") != "passed":
        raise RuntimeError("Dataset audit did not pass: {}".format(dataset_audit_path))
    if Path(dataset_audit.get("data_root", "")).resolve() != data_root:
        raise RuntimeError("Dataset audit belongs to a different data root.")
    if Path(dataset_audit.get("test_root", "")).resolve() != test_root:
        raise RuntimeError("Dataset audit belongs to a different test root.")

    train_rows, train_hash = load_manifest(
        args.train_manifest, data_root, require_denoise_sigma=False
    )
    val_rows, val_hash = load_manifest(
        args.val_manifest, data_root, require_denoise_sigma=True
    )
    train_targets = {(row["task"], row["target"]) for row in train_rows}
    val_targets = {(row["task"], row["target"]) for row in val_rows}
    target_overlap = train_targets & val_targets
    train_haze_scenes = {scene_id(row["input"]) for row in train_rows if row["task"] == "dehaze"}
    val_haze_scenes = {scene_id(row["input"]) for row in val_rows if row["task"] == "dehaze"}
    final_test_scenes = test_haze_scene_ids(test_root)
    final_test_overlap = (train_haze_scenes | val_haze_scenes) & final_test_scenes
    duplicate_rows = set()
    seen_rows = set()
    for label, rows in (("train", train_rows), ("val", val_rows)):
        for row in rows:
            key = (label, row["task"], row["input"], row["target"], row["sigma"])
            if key in seen_rows:
                duplicate_rows.add(key)
            seen_rows.add(key)

    passed = not target_overlap and not final_test_overlap and not duplicate_rows
    report = {
        "format_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if passed else "failed",
        "data_root": str(data_root),
        "test_root": str(test_root),
        "dataset_audit": {
            "path": str(dataset_audit_path),
            "sha256": sha256_file(dataset_audit_path),
            "status": dataset_audit.get("status"),
        },
        "train_manifest": {"path": str(Path(args.train_manifest).resolve()), "sha256": train_hash, "rows": count_tasks(train_rows)},
        "val_manifest": {"path": str(Path(args.val_manifest).resolve()), "sha256": val_hash, "rows": count_tasks(val_rows)},
        "target_overlap_count": len(target_overlap),
        "target_overlap_examples": sorted(target_overlap)[:20],
        "dehaze_train_scene_count": len(train_haze_scenes),
        "dehaze_val_scene_count": len(val_haze_scenes),
        "dehaze_final_test_scene_overlap_count": len(final_test_overlap),
        "dehaze_final_test_scene_overlap_examples": sorted(final_test_overlap)[:20],
        "duplicate_row_count": len(duplicate_rows),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
