"""Audit a clean AIO-3 data root before a new training run.

This is intentionally independent of the historical benchmark checker, whose
fixed 72,135-image expectation is inappropriate for a scene-disjoint cleaned
subset.  It checks three things that must all hold: valid pair mapping and
geometry, no RESIDE test scene IDs in dehazing training, and no exact bytes
shared between the new training root and final test root.
"""

import argparse
import hashlib
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


def image_paths(root):
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def relative_paths(root):
    root = Path(root)
    return [str(path.relative_to(root)).replace(os.sep, "/") for path in image_paths(root)]


def scene_id(path):
    return Path(path).stem.split("_", 1)[0]


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_status(path):
    try:
        with Image.open(path) as image:
            size = image.size
            image.verify()
        return size, None
    except Exception as error:  # Pillow can emit several decode exception types.
        return None, str(error)


def build_target_map(target_root):
    values = {}
    duplicate_stems = []
    for path in image_paths(target_root):
        if path.stem in values:
            duplicate_stems.append(path.stem)
        else:
            values[path.stem] = path
    return values, duplicate_stems


def dehaze_pairs(data_root):
    input_root = data_root / "dehaze" / "input"
    target_root = data_root / "dehaze" / "target"
    targets, duplicates = build_target_map(target_root)
    pairs, missing = [], []
    for source in image_paths(input_root):
        target = targets.get(scene_id(source))
        if target is None:
            missing.append(source)
        else:
            pairs.append((source, target))
    return pairs, missing, duplicates


def derain_pairs(data_root):
    input_root = data_root / "derain" / "input"
    target_root = data_root / "derain" / "target"
    targets = {path.name: path for path in image_paths(target_root)}
    pairs, missing = [], []
    for source in image_paths(input_root):
        name = source.name
        if name.startswith("rain-"):
            target_name = "norain-" + name[len("rain-") :]
        elif name.startswith("rain"):
            target_name = "norain" + name[len("rain") :]
        else:
            target_name = name
        target = targets.get(target_name)
        if target is None:
            missing.append(source)
        else:
            pairs.append((source, target))
    return pairs, missing


def audit_pair_geometry(pairs, verify_images, max_errors=20):
    errors = []
    if not verify_images:
        return errors
    for index, (input_path, target_path) in enumerate(pairs, start=1):
        input_size, input_error = image_status(input_path)
        target_size, target_error = image_status(target_path)
        if input_error or target_error or input_size != target_size:
            errors.append(
                {
                    "input": str(input_path),
                    "target": str(target_path),
                    "input_size": input_size,
                    "target_size": target_size,
                    "input_error": input_error,
                    "target_error": target_error,
                }
            )
            if len(errors) >= max_errors:
                break
        if index % 5000 == 0:
            print("verified paired images: {}".format(index), flush=True)
    return errors


def task_files(root, task):
    task_root = root / task
    return image_paths(task_root)


def audit_hash_overlap(train_root, test_root, task, max_examples=20):
    """Return exact byte overlaps for one task using SHA-256.

    Test hashes are indexed first. Training files are then streamed, so the
    routine remains bounded even when the OTS subset is large.
    """
    test_files = task_files(test_root, task)
    test_hashes = defaultdict(list)
    for path in test_files:
        test_hashes[sha256_file(path)].append(path)

    overlaps = []
    overlap_hashes = set()
    train_files = task_files(train_root, task)
    for index, path in enumerate(train_files, start=1):
        digest = sha256_file(path)
        if digest in test_hashes:
            overlap_hashes.add(digest)
            if len(overlaps) < max_examples:
                overlaps.append(
                    {
                        "train": str(path.relative_to(train_root)),
                        "test": [
                            str(test_path.relative_to(test_root))
                            for test_path in test_hashes[digest]
                        ],
                        "sha256": digest,
                    }
                )
        if index % 5000 == 0:
            print("hashed {} training files for {}".format(index, task), flush=True)
    return {
        "train_file_count": len(train_files),
        "test_file_count": len(test_files),
        "overlap_hash_count": len(overlap_hashes),
        "examples": overlaps,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify pair integrity and final-test independence of a clean AIO-3 root."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--test-root", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--verify-images", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    test_root = Path(args.test_root).resolve()
    output = Path(args.output).resolve() if args.output else data_root / "audit_report.json"
    if output.exists() and not args.overwrite:
        raise FileExistsError("Audit output already exists: {}".format(output))
    if not data_root.is_dir() or not test_root.is_dir():
        raise FileNotFoundError("Both --data-root and --test-root must exist.")

    haze_pairs, haze_missing, haze_target_duplicates = dehaze_pairs(data_root)
    rain_pairs, rain_missing = derain_pairs(data_root)
    denoise_targets = image_paths(data_root / "denoise" / "target")
    test_hazy = image_paths(test_root / "dehaze" / "input")
    test_scene_ids = {scene_id(path) for path in test_hazy}
    train_scene_ids = {scene_id(path) for path, _ in haze_pairs}

    geometry_errors = {
        "dehaze": audit_pair_geometry(haze_pairs, args.verify_images),
        "derain": audit_pair_geometry(rain_pairs, args.verify_images),
        "denoise": [],
    }
    if args.verify_images:
        for index, path in enumerate(denoise_targets, start=1):
            _, error = image_status(path)
            if error:
                geometry_errors["denoise"].append({"target": str(path), "error": error})
                if len(geometry_errors["denoise"]) >= 20:
                    break
            if index % 5000 == 0:
                print("verified denoise images: {}".format(index), flush=True)

    hash_overlap = {}
    for task in ("dehaze", "derain", "denoise"):
        hash_overlap[task] = audit_hash_overlap(data_root, test_root, task)

    errors_present = (
        bool(haze_missing)
        or bool(haze_target_duplicates)
        or bool(rain_missing)
        or bool(train_scene_ids & test_scene_ids)
        or any(geometry_errors.values())
        or any(record["overlap_hash_count"] for record in hash_overlap.values())
    )
    report = {
        "format_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root),
        "test_root": str(test_root),
        "verify_images": bool(args.verify_images),
        "status": "failed" if errors_present else "passed",
        "counts": {
            "dehaze_pairs": len(haze_pairs),
            "dehaze_clean_targets": len({target for _, target in haze_pairs}),
            "derain_pairs": len(rain_pairs),
            "denoise_clean_targets": len(denoise_targets),
            "test_dehaze_images": len(test_hazy),
            "test_dehaze_scenes": len(test_scene_ids),
        },
        "scene_overlap": {
            "count": len(train_scene_ids & test_scene_ids),
            "examples": sorted(train_scene_ids & test_scene_ids)[:20],
        },
        "pairing": {
            "dehaze_missing_targets": [str(path.relative_to(data_root)) for path in haze_missing[:20]],
            "dehaze_missing_target_count": len(haze_missing),
            "dehaze_duplicate_target_stems": haze_target_duplicates[:20],
            "derain_missing_targets": [str(path.relative_to(data_root)) for path in rain_missing[:20]],
            "derain_missing_target_count": len(rain_missing),
        },
        "geometry_or_decode_errors": geometry_errors,
        "exact_sha256_overlap": hash_overlap,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors_present:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
