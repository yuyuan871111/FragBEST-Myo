"""Modified by Yu-Yuab (Stuart) Yang - 2024
from ignite generator template - 2024

"""

from ...thirdparty.unet3d_model.unet3d import UnetModel


def setup_model(config):
    return UnetModel(
        in_channels=config.in_channels,
        out_channels=config.num_classes,
        final_activation="softmax",
    )
