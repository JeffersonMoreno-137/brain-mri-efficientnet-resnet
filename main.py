"""
CLI entry point for Brain MRI Classification pipeline.
"""

import argparse
from pathlib import Path
from src.config import DATASET_DIR
from src.data import prepare_datasets
from src.train import train_efficientnet, train_resnet50
from src.evaluate import evaluate_model


def main():
    parser = argparse.ArgumentParser(
        description="Brain MRI Classification with EfficientNet-B0 and ResNet50"
    )
    parser.add_argument(
        "--model",
        choices=["efficientnet", "resnet50", "all"],
        default="all",
        help="Model architecture to run",
    )
    parser.add_argument(
        "--fine-tune",
        action="store_true",
        default=True,
        help="Run fine-tuning phase on ResNet50",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DATASET_DIR,
        help="Path to dataset directory containing class folders",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Brain MRI Classification Pipeline")
    print(f"Dataset path: {args.dataset_dir}")
    print(f"Selected model: {args.model}")
    print("=" * 60)

    if not args.dataset_dir.exists():
        print(f"Dataset directory '{args.dataset_dir}' not found.")
        print("Please download and place the dataset in the 'dataset/' folder.")
        return

    (train_ds, val_ds, test_ds), (train_t, val_t, test_t), _ = prepare_datasets(args.dataset_dir)

    if args.model in ["efficientnet", "all"]:
        print("\n--- Training EfficientNet-B0 ---")
        eff_model, eff_hist = train_efficientnet(train_ds, val_ds)
        evaluate_model(eff_model, test_ds, test_t, "EfficientNet-B0")

    if args.model in ["resnet50", "all"]:
        print("\n--- Training ResNet50 ---")
        res_model, res_hist, ft_hist = train_resnet50(
            train_ds, val_ds, run_fine_tuning=args.fine_tune
        )
        evaluate_model(res_model, test_ds, test_t, "ResNet50 + Fine-tuning")


if __name__ == "__main__":
    main()
