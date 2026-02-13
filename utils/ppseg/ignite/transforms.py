import numpy as np
from monai.transforms import RandRotated


def setup_transforms(config):
    if config.transform == "RandRotated":
        assert config.rotated_range is not None, (
            "rotated_range must be provided (in degree)"
        )
        range_x = range_y = range_z = np.deg2rad(config.rotated_range)
        return RandRotated(
            keys=["data", "label"],
            range_x=range_x,
            range_y=range_y,
            range_z=range_z,
            prob=0.5,
            keep_size=True,
            mode="nearest",
            padding_mode="zeros",
            allow_missing_keys=False,
        )
    else:
        raise NotImplementedError(f"Transform {config.transform} not implemented")
