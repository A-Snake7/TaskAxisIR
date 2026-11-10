import argparse
import json
import os
import sys

from PIL import Image


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset import AllInOneDataset, TASK_NAMES  # noqa: E402


EXPECTED_PROFILES = {
    "train": {
        "derain": 200,
        "dehaze": 72135,
        "denoise": 5144,
    },
    "test": {
        "derain": 100,
        "dehaze": 500,
        "denoise": 68,
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="./data/train")
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument(
        "--profile",
        choices=tuple(EXPECTED_PROFILES),
        default="train",
    )
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--decode-all",
        action="store_true",
        help="Open and verify every unique input/target image.",
    )
    args = parser.parse_args()

    dataset = AllInOneDataset(
        args.data_root,
        patch_size=args.patch_size,
        train=args.profile == "train",
    )
    counts = {
        TASK_NAMES[task_id]: len(pairs)
        for task_id, pairs in dataset.task_pairs.items()
    }
    sample_shapes = {}
    for task_id, indices in dataset.task_indices.items():
        image_input, image_target, prompt = dataset[indices[0]]
        if image_input.shape != image_target.shape:
            raise RuntimeError(
                "{} sample geometry mismatch".format(TASK_NAMES[task_id])
            )
        sample_shapes[TASK_NAMES[task_id]] = list(image_input.shape)
        if prompt.item() != task_id:
            raise RuntimeError("Task prompt mismatch")

    report = {
        "counts": counts,
        "expected": EXPECTED_PROFILES[args.profile],
        "profile": args.profile,
        "sample_shapes": sample_shapes,
        "strict": args.strict,
    }
    if args.decode_all:
        unique_paths = set()
        for task_id, pairs in dataset.task_pairs.items():
            task_dir = dataset.task_dirs[task_id]
            for input_name, target_name in pairs:
                if not (task_id == 2 and args.profile == "train"):
                    unique_paths.add(
                        os.path.join(task_dir, "input", input_name)
                    )
                unique_paths.add(
                    os.path.join(task_dir, "target", target_name)
                )
        unique_paths = sorted(unique_paths)
        decode_errors = []
        for path in unique_paths:
            try:
                with Image.open(path) as image:
                    image.verify()
            except Exception as exc:
                decode_errors.append(
                    {
                        "path": path,
                        "error": "{}: {}".format(type(exc).__name__, exc),
                    }
                )
        report["decoded_unique_images"] = len(unique_paths)
        report["decode_errors"] = decode_errors
    print(json.dumps(report, indent=2))
    expected = EXPECTED_PROFILES[args.profile]
    if args.strict and counts != expected:
        raise SystemExit(
            "Dataset count mismatch: expected={} actual={}".format(
                expected,
                counts,
            )
        )
    if args.strict and report.get("decode_errors"):
        raise SystemExit(
            "Image decode validation failed for {} file(s)".format(
                len(report["decode_errors"])
            )
        )


if __name__ == "__main__":
    main()
