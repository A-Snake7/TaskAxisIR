import argparse
import os
import shutil
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
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def validate_images(paths, label):
    invalid = []
    for path in paths:
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception as error:  # Pillow exposes multiple decode exceptions.
            invalid.append((path, str(error)))
    if invalid:
        preview = "\n".join(
            "{}: {}".format(path, error) for path, error in invalid[:10]
        )
        raise RuntimeError(
            "{} contains {} invalid images:\n{}".format(
                label,
                len(invalid),
                preview,
            )
        )


def link_file(source, destination, mode):
    source = source.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.resolve() == source:
            return False
        raise FileExistsError(
            "Refusing to replace existing path: {}".format(destination)
        )
    if mode == "symlink":
        destination.symlink_to(source)
    elif mode == "hardlink":
        os.link(str(source), str(destination))
    elif mode == "copy":
        shutil.copy2(source, destination)
    else:
        raise ValueError("Unsupported mode: {}".format(mode))
    return True


def add_tree(
    source_root,
    destination_root,
    mode,
    verify=False,
    include=None,
):
    source_root = Path(source_root).resolve()
    paths = image_paths(source_root)
    if include is not None:
        paths = [path for path in paths if include(path)]
    if verify:
        validate_images(paths, str(source_root))
    created = 0
    for source in paths:
        relative = source.relative_to(source_root)
        created += int(
            link_file(source, Path(destination_root) / relative, mode)
        )
    return len(paths), created


def find_ots_subdirs(root):
    root = Path(root)
    directories = [path for path in root.rglob("*") if path.is_dir()]

    def score(path, keywords):
        name = path.name.lower()
        count = len(image_paths(path))
        keyword_score = max(
            (10 if name == keyword else 1)
            for keyword in keywords
            if keyword in name
        ) if any(keyword in name for keyword in keywords) else 0
        return keyword_score, count

    hazy_keywords = ("hazy", "haze", "synthetic")
    clean_keywords = ("clear", "clean", "gt", "original")
    hazy_candidates = [
        (score(path, hazy_keywords), path)
        for path in directories
        if any(word in path.name.lower() for word in hazy_keywords)
    ]
    clean_candidates = [
        (score(path, clean_keywords), path)
        for path in directories
        if any(word in path.name.lower() for word in clean_keywords)
    ]
    if not hazy_candidates or not clean_candidates:
        raise RuntimeError(
            "Could not automatically locate OTS hazy/clear directories under "
            "{}".format(root)
        )
    hazy_root = max(hazy_candidates, key=lambda item: item[0])[1]
    clean_root = max(clean_candidates, key=lambda item: item[0])[1]
    return hazy_root, clean_root


def parse_args():
    parser = argparse.ArgumentParser(
        description="Align WED and RESIDE-OTS with the ProMamba data layout."
    )
    parser.add_argument("--data-root", default="./data/train")
    parser.add_argument("--wed-root", default="")
    parser.add_argument("--ots-root", default="")
    parser.add_argument("--ots-hazy-root", default="")
    parser.add_argument("--ots-clean-root", default="")
    parser.add_argument(
        "--mode",
        choices=("symlink", "hardlink", "copy"),
        default="symlink",
    )
    parser.add_argument(
        "--verify-images",
        action="store_true",
        help="Decode every source image before alignment (slow for OTS).",
    )
    parser.add_argument(
        "--ots-max-scene-id",
        type=int,
        default=None,
        help=(
            "Historical count-based selector. This is not an official RESIDE "
            "training split and is rejected unless --unsafe-legacy-ots-range is "
            "also supplied."
        ),
    )
    parser.add_argument(
        "--unsafe-legacy-ots-range",
        action="store_true",
        help=(
            "Acknowledge that --ots-max-scene-id is only for reproducing an "
            "old experiment. Use build_clean_aio3_dataset.py for any new "
            "benchmark or paper result."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    if not args.wed_root and not args.ots_root and not args.ots_hazy_root:
        raise ValueError("Provide --wed-root and/or an OTS root")
    legacy_ots_root = (
        bool(args.ots_root)
        and not args.ots_hazy_root
        and not args.ots_clean_root
    )
    if legacy_ots_root and not args.unsafe_legacy_ots_range:
        raise RuntimeError(
            "Refusing count-based OTS preparation because scene IDs are not an "
            "official split and can overlap SOTS. Use "
            "build_clean_aio3_dataset.py for a new clean dataset, or pass both "
            "--unsafe-legacy-ots-range and --ots-max-scene-id only to reproduce "
            "an archived experiment."
        )
    if args.unsafe_legacy_ots_range and args.ots_max_scene_id is None:
        raise ValueError(
            "--unsafe-legacy-ots-range requires an explicit --ots-max-scene-id"
        )

    if args.wed_root:
        total, created = add_tree(
            args.wed_root,
            data_root / "denoise" / "target" / "wed",
            args.mode,
            verify=args.verify_images,
        )
        print("WED aligned: total={} new={}".format(total, created))

    if args.ots_hazy_root or args.ots_clean_root:
        if not args.ots_hazy_root or not args.ots_clean_root:
            raise ValueError(
                "--ots-hazy-root and --ots-clean-root must be used together"
            )
        hazy_root = Path(args.ots_hazy_root)
        clean_root = Path(args.ots_clean_root)
    elif args.ots_root:
        ots_root = Path(args.ots_root)
        prepared_hazy = ots_root / "train" / "hazy"
        prepared_clean = ots_root / "train" / "gt"
        if prepared_hazy.is_dir() and prepared_clean.is_dir():
            def include_ots_beta(path):
                token = path.stem.split("_", 1)[0]
                return (
                    token.isdigit()
                    and 1 <= int(token) <= args.ots_max_scene_id
                )

            total_hazy, total_hazy_created = add_tree(
                prepared_hazy,
                data_root / "dehaze" / "input" / "ots",
                args.mode,
                verify=args.verify_images,
                include=include_ots_beta,
            )
            total_clean, total_clean_created = add_tree(
                prepared_clean,
                data_root / "dehaze" / "target" / "ots",
                args.mode,
                verify=args.verify_images,
                include=include_ots_beta,
            )
            print(
                "OTS train split aligned: hazy={} (new={}) "
                "clean={} (new={})".format(
                    total_hazy,
                    total_hazy_created,
                    total_clean,
                    total_clean_created,
                )
            )
            if (ots_root / "test").is_dir():
                print(
                    "OTS archive test split intentionally ignored; "
                    "benchmark tests must remain under data/test."
                )
            return
        hazy_root, clean_root = find_ots_subdirs(ots_root)
        print(
            "Detected OTS directories: hazy={} clean={}".format(
                hazy_root,
                clean_root,
            )
        )
    else:
        return

    hazy_total, hazy_created = add_tree(
        hazy_root,
        data_root / "dehaze" / "input" / "ots",
        args.mode,
        verify=args.verify_images,
    )
    clean_total, clean_created = add_tree(
        clean_root,
        data_root / "dehaze" / "target" / "ots",
        args.mode,
        verify=args.verify_images,
    )
    print(
        "OTS aligned: hazy={} (new={}) clean={} (new={})".format(
            hazy_total,
            hazy_created,
            clean_total,
            clean_created,
        )
    )


if __name__ == "__main__":
    main()
