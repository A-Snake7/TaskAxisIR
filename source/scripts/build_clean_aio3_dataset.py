"""Build a new, scene-disjoint AIO-3 dataset without touching old data.

The original ``prepare_training_data.py`` was written for a count-based OTS
subset.  It must not be used to create a benchmark split: numerical scene IDs
are not a valid RESIDE training-list definition and some test scenes can occur
in an archive's train directory.  This tool writes a separate data root using
links, excludes every SOTS-Outdoor scene ID, and leaves all original data and
previous runs unchanged.

The resulting root is suitable for ``dataset.AllInOneDataset`` and can be
split further with ``create_split_manifests.py``.
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path


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
    """Return stable, relative-path-safe image paths beneath ``root``."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError("Image directory does not exist: {}".format(root))
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scene_id(path):
    """Extract the clear-scene identifier used by RESIDE outdoor filenames."""
    value = Path(path).stem.split("_", 1)[0]
    if not value:
        raise ValueError("Cannot infer a RESIDE scene ID from: {}".format(path))
    return value


def link_file(source, destination, mode):
    source = Path(source).resolve()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Refusing to replace an existing path: {}".format(destination))
    if mode == "symlink":
        destination.symlink_to(source)
    elif mode == "hardlink":
        os.link(str(source), str(destination))
    elif mode == "copy":
        shutil.copy2(source, destination)
    else:
        raise ValueError("Unsupported link mode: {}".format(mode))


def normalize_list_name(value):
    return Path(value.strip().replace("\\", "/")).name


def read_official_list(path):
    """Read an optional official hazy-image list without downloading data."""
    if not path:
        return None, None
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("Official list not found: {}".format(path))
    names = set()
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                names.add(normalize_list_name(line))
    if not names:
        raise ValueError("Official list contains no image entries: {}".format(path))
    return names, sha256_file(path)


def inventory(root):
    rows = image_paths(root)
    return {
        "root": str(Path(root).resolve()),
        "image_count": len(rows),
        "examples": [str(path.relative_to(root)) for path in rows[:3]],
    }


def target_map(target_root):
    result = {}
    duplicates = []
    for path in image_paths(target_root):
        stem = path.stem
        if stem in result:
            duplicates.append((stem, result[stem], path))
        else:
            result[stem] = path
    if duplicates:
        preview = ", ".join(str(item[0]) for item in duplicates[:5])
        raise RuntimeError(
            "Clean target directory has duplicate scene IDs (for example: {})".format(
                preview
            )
        )
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create an independent AIO-3 data root and remove every SOTS-Outdoor "
            "test scene from the dehazing training source."
        )
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--hazy-source", required=True)
    parser.add_argument("--hazy-target-source", required=True)
    parser.add_argument("--test-hazy-root", required=True)
    parser.add_argument("--test-target-root", required=True)
    parser.add_argument("--derain-source", required=True)
    parser.add_argument("--denoise-source", required=True)
    parser.add_argument(
        "--official-hazy-list",
        default="",
        help=(
            "Optional official OTS image list. When supplied, only files named in "
            "that list are eligible before the test-scene exclusion is applied."
        ),
    )
    parser.add_argument(
        "--selection-label",
        default="available_ots_scene_disjoint",
        help="Human-readable protocol label recorded in the provenance file.",
    )
    parser.add_argument(
        "--mode",
        choices=("symlink", "hardlink", "copy"),
        default="symlink",
    )
    return parser.parse_args()


def link_tree(source_root, destination_root, mode, manifest_rows, task, role):
    source_root = Path(source_root).resolve()
    count = 0
    for source in image_paths(source_root):
        relative = source.relative_to(source_root)
        link_file(source, Path(destination_root) / relative, mode)
        manifest_rows.append(
            {
                "task": task,
                "role": role,
                "relative_path": str(relative).replace(os.sep, "/"),
                "source_path": str(source),
            }
        )
        count += 1
    return count


def main():
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(
            "Output root already exists; this tool never overwrites data: {}".format(
                output_root
            )
        )
    output_root.parent.mkdir(parents=True, exist_ok=True)

    hazy_source = Path(args.hazy_source).resolve()
    hazy_target_source = Path(args.hazy_target_source).resolve()
    test_hazy_root = Path(args.test_hazy_root).resolve()
    test_target_root = Path(args.test_target_root).resolve()
    derain_source = Path(args.derain_source).resolve()
    denoise_source = Path(args.denoise_source).resolve()
    official_names, official_list_hash = read_official_list(args.official_hazy_list)

    test_hazy = image_paths(test_hazy_root)
    test_target = image_paths(test_target_root)
    test_scene_ids = {scene_id(path) for path in test_hazy}
    source_targets = target_map(hazy_target_source)
    all_hazy = image_paths(hazy_source)

    eligible = []
    skipped_not_in_list = []
    skipped_test_scene = []
    missing_target = []
    for hazy in all_hazy:
        if official_names is not None and hazy.name not in official_names:
            skipped_not_in_list.append(hazy)
            continue
        clean_id = scene_id(hazy)
        if clean_id in test_scene_ids:
            skipped_test_scene.append(hazy)
            continue
        target = source_targets.get(clean_id)
        if target is None:
            missing_target.append(hazy)
            continue
        eligible.append((hazy, target))

    if missing_target:
        preview = ", ".join(path.name for path in missing_target[:5])
        raise RuntimeError(
            "{} selected hazy files do not have a clean target (for example: {}). "
            "No output was created.".format(len(missing_target), preview)
        )
    if not eligible:
        raise RuntimeError("No dehazing training pairs remain after safe filtering.")

    stage_root = Path(
        tempfile.mkdtemp(prefix=".{}-staging-".format(output_root.name), dir=str(output_root.parent))
    )
    manifest_rows = []
    try:
        selected_target_ids = set()
        for hazy, target in eligible:
            input_relative = hazy.relative_to(hazy_source)
            target_relative = target.relative_to(hazy_target_source)
            link_file(hazy, stage_root / "dehaze" / "input" / input_relative, args.mode)
            manifest_rows.append(
                {
                    "task": "dehaze",
                    "role": "input",
                    "relative_path": str(input_relative).replace(os.sep, "/"),
                    "source_path": str(hazy),
                }
            )
            clean_id = scene_id(hazy)
            if clean_id not in selected_target_ids:
                link_file(
                    target,
                    stage_root / "dehaze" / "target" / target_relative,
                    args.mode,
                )
                manifest_rows.append(
                    {
                        "task": "dehaze",
                        "role": "target",
                        "relative_path": str(target_relative).replace(os.sep, "/"),
                        "source_path": str(target),
                    }
                )
                selected_target_ids.add(clean_id)

        for role in ("input", "target"):
            source = derain_source / role
            if not source.is_dir():
                raise FileNotFoundError("Missing deraining source directory: {}".format(source))
            link_tree(
                source,
                stage_root / "derain" / role,
                args.mode,
                manifest_rows,
                "derain",
                role,
            )

        denoise_target = denoise_source / "target"
        if not denoise_target.is_dir():
            raise FileNotFoundError("Missing denoising target directory: {}".format(denoise_target))
        link_tree(
            denoise_target,
            stage_root / "denoise" / "target",
            args.mode,
            manifest_rows,
            "denoise",
            "target",
        )
        optional_denoise_input = denoise_source / "input"
        if optional_denoise_input.is_dir():
            link_tree(
                optional_denoise_input,
                stage_root / "denoise" / "input",
                args.mode,
                manifest_rows,
                "denoise",
                "input",
            )

        with open(stage_root / "source_manifest.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("task", "role", "relative_path", "source_path"),
            )
            writer.writeheader()
            writer.writerows(manifest_rows)

        provenance = {
            "format_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "selection_label": args.selection_label,
            "link_mode": args.mode,
            "builder_script_sha256": sha256_file(__file__),
            "output_root": str(output_root),
            "sources": {
                "hazy_source": inventory(hazy_source),
                "hazy_target_source": inventory(hazy_target_source),
                "test_hazy_root": inventory(test_hazy_root),
                "test_target_root": inventory(test_target_root),
                "derain_source": str(derain_source),
                "denoise_source": str(denoise_source),
                "official_hazy_list": str(Path(args.official_hazy_list).resolve())
                if args.official_hazy_list
                else "",
                "official_hazy_list_sha256": official_list_hash or "",
            },
            "dehaze": {
                "raw_hazy_count": len(all_hazy),
                "test_hazy_count": len(test_hazy),
                "test_target_count": len(test_target),
                "test_scene_count": len(test_scene_ids),
                "official_list_entries": len(official_names) if official_names else None,
                "skipped_not_in_official_list": len(skipped_not_in_list),
                "skipped_test_scene_hazy": len(skipped_test_scene),
                "selected_hazy_count": len(eligible),
                "selected_clean_scene_count": len(selected_target_ids),
                "test_scene_ids_sha256": hashlib.sha256(
                    "\n".join(sorted(test_scene_ids)).encode("utf-8")
                ).hexdigest(),
                "scene_id_overlap_count": 0,
            },
            "output_inventory": {
                "derain_input_count": len(image_paths(stage_root / "derain" / "input")),
                "derain_target_count": len(image_paths(stage_root / "derain" / "target")),
                "dehaze_input_count": len(image_paths(stage_root / "dehaze" / "input")),
                "dehaze_target_count": len(image_paths(stage_root / "dehaze" / "target")),
                "denoise_target_count": len(image_paths(stage_root / "denoise" / "target")),
            },
            "limitations": [
                "This dataset is scene-disjoint from the supplied SOTS-Outdoor root.",
                "It is not an exact AirNet/PromptIR/AdaIR reproduction unless an official list is supplied and every required source image is present.",
                "Run audit_clean_aio3_dataset.py before training to verify exact file hashes and pair geometry.",
            ],
        }
        with open(stage_root / "provenance.json", "w", encoding="utf-8") as handle:
            json.dump(provenance, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        with open(stage_root / "README.md", "w", encoding="utf-8") as handle:
            handle.write(
                "# Clean AIO-3 data root\n\n"
                "This directory was created without modifying the original training data. "
                "For dehazing, every SOTS-Outdoor scene ID was excluded before "
                "links were created. See `provenance.json`, `source_manifest.csv`, "
                "and the later audit report before using it for training.\n"
            )
        os.replace(str(stage_root), str(output_root))
    except Exception:
        shutil.rmtree(stage_root, ignore_errors=True)
        raise

    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "dehaze_hazy": len(eligible),
                "dehaze_clean": len(selected_target_ids),
                "excluded_test_scene_hazy": len(skipped_test_scene),
                "selection_label": args.selection_label,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
