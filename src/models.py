from monai.networks.nets import UNet


class MonaiUNet(UNet):
    def __init__(self):
        super().__init__(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )


from unet.unet_model import UNet
