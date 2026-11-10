import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Plot ProMamba training metrics")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--save_path", default="")
    args = parser.parse_args()

    csv_path = os.path.join(args.run_dir, "metrics.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(csv_path)
    metrics = pd.read_csv(csv_path)
    if metrics.empty:
        raise ValueError("Metrics file is empty")
    if metrics["epoch"].duplicated().any():
        duplicates = metrics.loc[metrics["epoch"].duplicated(), "epoch"].tolist()
        raise ValueError("Duplicate epochs found: {}".format(duplicates))

    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(metrics["epoch"], metrics["train_loss"], label="train loss")
    axes[0].set_title("Normalized weighted loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(True)
    axes[0].legend()

    for task in ("derain", "dehaze", "denoise"):
        column = "train_{}_psnr".format(task)
        if column in metrics:
            axes[1].plot(metrics["epoch"], metrics[column], label=task)
    if "val_macro_psnr" in metrics and metrics["val_macro_psnr"].notna().any():
        axes[1].plot(
            metrics["epoch"],
            metrics["val_macro_psnr"],
            label="validation macro",
            linewidth=2,
        )
    axes[1].set_title("PSNR by task")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("dB")
    axes[1].grid(True)
    axes[1].legend()
    figure.tight_layout()

    save_path = args.save_path or os.path.join(args.run_dir, "training_curves.png")
    figure.savefig(save_path, dpi=160)
    plt.close(figure)
    print("Saved {}".format(save_path))


if __name__ == "__main__":
    main()
