import argparse
import os
import re


VALID_EXTENSIONS = (".png", ".jpg", ".jpeg")


def apply_rename(source, destination, execute):
    if source == destination:
        return False
    if os.path.exists(destination):
        raise FileExistsError(
            "Refusing to overwrite existing destination: {}".format(destination)
        )
    print(
        "{} RENAME {} -> {}".format(
            "APPLY" if execute else "DRY-RUN",
            source,
            destination,
        )
    )
    if execute:
        os.rename(source, destination)
    return True


def align_data(base_dir, mode="test", execute=False):
    changes = 0
    rain_input = os.path.join(base_dir, mode, "derain", "input")
    rain_target = os.path.join(base_dir, mode, "derain", "target")
    if os.path.isdir(rain_input) and os.path.isdir(rain_target):
        matched = 0
        unmatched = []
        for filename in sorted(os.listdir(rain_input)):
            if not filename.lower().endswith(VALID_EXTENSIONS):
                continue
            numbers = re.findall(r"\d+", filename)
            if not numbers:
                continue
            target = os.path.join(rain_target, "norain-{}.png".format(numbers[0]))
            if os.path.isfile(target):
                matched += 1
            else:
                unmatched.append(filename)
        print(
            "Derain mapping validation: matched={} unmatched={}".format(
                matched,
                len(unmatched),
            )
        )
        if unmatched:
            print("Unmatched examples: {}".format(unmatched[:10]))

    haze_input = os.path.join(base_dir, mode, "dehaze", "input")
    haze_target = os.path.join(base_dir, mode, "dehaze", "target")
    for directory, token in ((haze_input, "_hazy"), (haze_target, "_GT")):
        if not os.path.isdir(directory):
            continue
        for filename in sorted(os.listdir(directory)):
            if token not in filename:
                continue
            source = os.path.join(directory, filename)
            destination = os.path.join(directory, filename.replace(token, ""))
            changes += apply_rename(source, destination, execute)

    denoise_input = os.path.join(base_dir, mode, "denoise", "input")
    denoise_target = os.path.join(base_dir, mode, "denoise", "target")
    if os.path.isdir(denoise_input) and os.path.isdir(denoise_target):
        input_names = {
            name
            for name in os.listdir(denoise_input)
            if name.lower().endswith(VALID_EXTENSIONS)
        }
        target_names = {
            name
            for name in os.listdir(denoise_target)
            if name.lower().endswith(VALID_EXTENSIONS)
        }
        if input_names != target_names:
            print(
                "WARNING denoise filenames differ: input_only={} target_only={}".format(
                    len(input_names - target_names),
                    len(target_names - input_names),
                )
            )

    print("{} proposed change(s).".format(changes))
    if not execute:
        print("No files changed. Re-run with --apply after reviewing the plan.")


def main():
    parser = argparse.ArgumentParser(description="Safely align dataset filenames")
    parser.add_argument("--base_dir", default="data")
    parser.add_argument("--mode", choices=("train", "test"), default="test")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Perform changes. Default is dry-run.",
    )
    args = parser.parse_args()
    align_data(args.base_dir, mode=args.mode, execute=args.apply)


if __name__ == "__main__":
    main()
