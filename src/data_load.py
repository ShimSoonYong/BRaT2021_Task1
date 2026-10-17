import torch
import numpy as np
from pathlib import Path
import monai.transforms as mt
from monai.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import ShuffleSplit


def get_data_path(data_root: str = "./dataset") -> tuple:
    test_hint = [
        "00014",
        "00024",
        "00044",
        "00056",
        "00070",
        "00090",
        "00104",
        "00111",
        "00128",
        "00138",
        "00150",
        "00162",
    ]

    root = Path(data_root)

    image_path = sorted(
        [
            p
            for p in root.rglob("*_t1ce.nii.gz")
            if not any(hint in part for part in p.parts for hint in test_hint)
        ]
    )
    label_path = sorted(
        [
            p
            for p in root.rglob("*_seg.nii.gz")
            if not any(hint in part for part in p.parts for hint in test_hint)
        ]
    )

    data_path = {
        "image": np.array(image_path, dtype=object),
        "label": np.array(label_path, dtype=object),
    }

    test_image_path = sorted(
        [
            p
            for p in root.rglob("*_t1ce.nii.gz")
            if any(hint in part for part in p.parts for hint in test_hint)
        ]
    )
    test_label_path = sorted(
        [
            p
            for p in root.rglob("*_seg.nii.gz")
            if any(hint in part for part in p.parts for hint in test_hint)
        ]
    )
    test_path = {
        "image": np.array(test_image_path, dtype=object),
        "label": np.array(test_label_path, dtype=object),
    }

    n_samples = len(data_path["image"])
    ss = ShuffleSplit(n_splits=1, test_size=0.2, train_size=0.8, random_state=42)
    train_path, val_path = {}, {}

    for train_idx, val_idx in ss.split(np.arange(n_samples)):
        print(f"{len(train_idx)} of train samples")
        print(f"{len(val_idx)} of validation samples")

        train_path["image"] = data_path["image"][train_idx].tolist()
        train_path["label"] = data_path["label"][train_idx].tolist()

        val_path["image"] = data_path["image"][val_idx].tolist()
        val_path["label"] = data_path["label"][val_idx].tolist()

    test_path["image"] = test_path["image"].tolist()
    test_path["label"] = test_path["label"].tolist()

    return train_path, val_path, test_path


def get_dataset(
    train_path: dict,
    val_path: dict,
    test_path: dict,
    img_augment: list | None = None,
    seg_augment: list | None = None,
) -> tuple:

    base_train_transforms = [
        mt.LoadImaged(keys=["image", "label"]),
        mt.EnsureChannelFirstd(keys=["image", "label"]),
        mt.ScaleIntensityd(keys=["image"]),
        # Convert labels to binary {0, 1} (threshold at 0: anything > 0 becomes 1)
        mt.Lambdad(keys=["label"], func=lambda x: (x > 0).float()),
        mt.CastToTyped(keys=["image", "label"], dtype=[torch.float32, torch.float32]),
    ]

    if img_augment:
        base_train_transforms.extend(img_augment)

    train_transforms = mt.Compose(base_train_transforms)

    val_transforms = mt.Compose(
        [
            mt.LoadImaged(keys=["image", "label"]),
            mt.EnsureChannelFirstd(keys=["image", "label"]),
            mt.ScaleIntensityd(keys=["image"]),
            mt.Lambdad(keys=["label"], func=lambda x: (x > 0).float()),
            mt.CenterSpatialCropd(keys=["image", "label"], roi_size=(-1, -1, 1)),
            mt.SqueezeDimd(keys=["image", "label"], dim=-1),
            mt.CastToTyped(
                keys=["image", "label"], dtype=[torch.float32, torch.float32]
            ),
        ]
    )

    # Convert paths dictionary lists to MONAI dict-format data lists
    def make_dicts(path_dict):
        return [
            {"image": img, "label": lbl}
            for img, lbl in zip(path_dict["image"], path_dict["label"])
        ]

    train_files = make_dicts(train_path)
    val_files = make_dicts(val_path)
    test_files = make_dicts(test_path)

    train_set = Dataset(data=train_files, transform=train_transforms)
    val_set = Dataset(data=val_files, transform=val_transforms)
    test_set = Dataset(data=test_files, transform=val_transforms)

    return train_set, val_set, test_set


def get_loader(
    train_set: Dataset,
    val_set: Dataset,
    test_set: Dataset,
    batch_size: int,
    num_workers: int,
) -> tuple:

    physical_batch_size = min(30, batch_size)

    train_loader = DataLoader(
        train_set,
        batch_size=physical_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=10,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=10,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=10,
    )

    return train_loader, val_loader, test_loader
