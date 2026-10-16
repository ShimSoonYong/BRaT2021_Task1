import numpy as np
import monai.transforms as mt


def get_augmentations(aug_name: str | None) -> list:
    """Builds 2D slice extraction and 2D augmentation transforms"""

    # Base transforms: Dynamically extract a 2D slice from the 3D volume and squeeze the Z-axis
    base_transforms = [
        mt.RandSpatialCropd(
            keys=["image", "label"],
            roi_size=(
                -1,
                -1,
                1,
            ),  # Keep full Height/Width, take exactly 1 slice in Depth
            random_size=False,
        ),
        mt.SqueezeDimd(
            keys=["image", "label"],
            dim=-1,  # Remove the depth dimension: (C, H, W, 1) -> (C, H, W)
        ),
    ]

    if not aug_name or aug_name.lower() == "none":
        return base_transforms

    aug_name = aug_name.lower()

    # 1. Individual Intensity Augmentations (Image Only)
    if aug_name == "gaussian_noise":
        aug_transforms = [
            mt.RandGaussianNoised(keys=["image"], prob=0.5, mean=0.0, std=0.1)
        ]

    elif aug_name == "gaussian_smooth":
        aug_transforms = [
            mt.RandGaussianSmoothd(
                keys=["image"],
                sigma_x=(0.5, 1.15),
                sigma_y=(0.5, 1.15),
                prob=0.3,
            )
        ]

    elif aug_name == "contrast":
        aug_transforms = [
            mt.RandAdjustContrastd(keys=["image"], prob=0.5, gamma=(0.7, 1.5))
        ]

    elif aug_name == "intensity_shift":
        aug_transforms = [
            mt.RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
            mt.RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
        ]

    # 2. Individual Spatial Augmentations (Synchronized: image + label)
    elif aug_name == "flip":
        aug_transforms = [
            mt.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            mt.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        ]

    elif aug_name == "rotate90":
        aug_transforms = [
            mt.RandRotate90d(
                keys=["image", "label"], prob=0.5, max_k=3, spatial_axes=(0, 1)
            )
        ]

    elif aug_name == "affine":
        aug_transforms = [
            mt.RandAffined(
                keys=["image", "label"],
                prob=0.5,
                rotate_range=(np.pi / 12,),  # 2D rotation angle
                translate_range=(10, 10),
                scale_range=(0.1, 0.1),
                mode=("bilinear", "nearest"),  # keep label discrete
            )
        ]

    elif aug_name == "elastic":
        aug_transforms = [
            mt.Rand2DElasticd(
                keys=["image", "label"],
                prob=0.4,
                spacing=(20, 20),
                magnitude_range=(1.0, 2.0),
                mode=("bilinear", "nearest"),
            )
        ]

    elif aug_name == "random_crop":
        # Additional 2D crop applied AFTER the 2D slice extraction
        aug_transforms = [
            mt.RandSpatialCropd(
                keys=["image", "label"],
                roi_size=(224, 224),
                random_size=False,
            )
        ]

    # 3. Comprehensive Presets (Recommended for Training)
    elif aug_name in ["strong", "all"]:
        aug_transforms = [
            # 2D Spatial transformations
            mt.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            mt.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            mt.RandRotated(
                keys=["image", "label"],
                range_x=np.pi / 8,  # 2D rotation
                mode=("bilinear", "nearest"),
                prob=0.4,
            ),
            mt.RandZoomd(
                keys=["image", "label"],
                min_zoom=0.85,
                max_zoom=1.15,
                mode=("bilinear", "nearest"),
                prob=0.4,
            ),
            # Intensity transformations
            mt.RandGaussianNoised(keys=["image"], prob=0.3, mean=0.0, std=0.05),
            mt.RandAdjustContrastd(keys=["image"], prob=0.4, gamma=(0.75, 1.35)),
            mt.RandGaussianSmoothd(
                keys=["image"],
                sigma_x=(0.5, 1.0),
                sigma_y=(0.5, 1.0),
                prob=0.2,
            ),
        ]

    else:
        raise ValueError(f"Unknown augmentation type: '{aug_name}'")

    return base_transforms + aug_transforms
