import torch
from monai.networks.nets import UNet
from monai.networks.nets import DynUNet


class AutoUNet(DynUNet):
    def __init__(
        self,
        spatial_dims: int = 2,
        in_channels: int = 1,
        out_channels: int = 1,
        spatial_shape: tuple = (240, 240),  # (H, W)
    ):
        strides = []
        kernel_size = []
        cur_shape = list(spatial_shape)

        while min(cur_shape) >= 8:
            strides.append([2] * spatial_dims)
            kernel_size.append([3] * spatial_dims)
            cur_shape = [s // 2 for s in cur_shape]

        if not strides:
            strides = [[1] * spatial_dims]
            kernel_size = [[3] * spatial_dims]

        strides[0] = [1] * spatial_dims

        upsample_kernel_size = strides[1:]
        filters = [min(2 ** (5 + i), 512) for i in range(len(strides))]

        super().__init__(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            upsample_kernel_size=upsample_kernel_size,
            filters=filters,
            norm_name="instance",
            deep_supervision=False,
        )

    def forward(self, x: torch.Tensor):
        return super().forward(x)


class MonaiUNet(UNet):
    def __init__(self):
        super().__init__(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128, 256, 512, 1024),
            strides=(2, 2, 2, 2, 2, 2, 2),
            num_res_units=2,
        )


from unet.unet_model import UNet
