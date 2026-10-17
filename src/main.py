from argparse import ArgumentParser
from monai.losses import DiceLoss
import torch
import torch.optim as optim
import models
from utils import *
from data_load import get_data_path, get_dataset, get_loader
from trainer import Trainer


def parse_args():
    parser = ArgumentParser(
        description="Medical Image Segmentation Training & Evaluation Pipeline"
    )

    # Model and Data Arguments
    parser.add_argument(
        "--model",
        type=str,
        default="UNet",
        help="Model name inside models module (e.g., UNet)",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="./dataset",
        help="Root directory containing the dataset",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for training DataLoader"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker threads for DataLoader",
    )

    # Training Hyperparameters
    parser.add_argument(
        "--epochs", type=int, default=50, help="Total number of training epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-5, help="Weight decay for optimizer"
    )

    # Augmentation selection flags
    parser.add_argument(
        "--augment",
        type=str,
        default="none",
        choices=[
            "none",
            "flip",
            "rotate90",
            "affine",
            "elastic",
            "random_crop",
            "gaussian_noise",
            "gaussian_smooth",
            "contrast",
            "intensity_shift",
            "strong",
            "all",
        ],
        help="Augmentation pipeline to apply during training",
    )

    args = parser.parse_args()
    return args


def main(args):
    # 1. Dynamically retrieve the model class from the models module
    if not hasattr(models, args.model):
        raise AttributeError(f"Model '{args.model}' not found in the 'models' module.")

    model_class = getattr(models, args.model)
    model = model_class()

    # 2. Set augmentations
    train_augmentations = get_augmentations(args.augment)

    # 3. Data Pipeline Initialization
    print("Loading data paths...")
    train_path, val_path, test_path = get_data_path(data_root=args.data_root)

    print("Building datasets...")
    train_set, val_set, test_set = get_dataset(
        train_path,
        val_path,
        test_path,
        img_augment=train_augmentations,
        seg_augment=None,
    )

    print("Creating data loaders...")
    train_loader, val_loader, test_loader = get_loader(
        train_set,
        val_set,
        test_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # 4. Optimizer and Loss Setup
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    loss_function = DiceLoss(
        to_onehot_y=False, sigmoid=True, squared_pred=True, include_background=True
    )

    # 5. Initialize Learner
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = Trainer(
        model=model,
        model_name=args.model + "_" + args.augment,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_function=loss_function,
        test_loader=test_loader,
        target_batch_size=args.batch_size,  # Trainer figures out accum_steps internally
        device=device,
    )

    # 6. Execute Training Lifecycle
    trainer.fit(epochs=args.epochs)


if __name__ == "__main__":
    args = parse_args()
    main(args)
    print("Job done")
