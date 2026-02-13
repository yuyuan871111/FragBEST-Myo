from typing import Any

from monai.losses import (
    FocalLoss,
    GeneralizedDiceFocalLoss,
    GeneralizedDiceLoss,
    MaskedDiceLoss,
)
from monai.losses.spatial_mask import MaskedLoss
from torch import Tensor
from torch.nn import CrossEntropyLoss


def setup_loss(config):
    weights = Tensor(config.weights)
    if config.loss_fn == "cross_entropy":
        return CrossEntropyLoss(weight=weights)
    elif config.loss_fn == "generalized_dice":
        return GeneralizedDiceLoss(to_onehot_y=True, softmax=True)
    elif config.loss_fn == "focal":
        return FocalLoss(weight=weights, to_onehot_y=True, use_softmax=True)
    elif config.loss_fn == "generalized_dice_focal":
        return GeneralizedDiceFocalLoss(weight=weights, to_onehot_y=True, softmax=True)
    elif config.loss_fn == "masked_generalized_dice_focal":
        return MaskedGenealizedDiceFocalLoss(
            weight=weights, to_onehot_y=True, softmax=True
        )
    elif config.loss_fn == "masked_dice":
        return MaskedDiceLoss(to_onehot_y=True, softmax=True)
    else:
        raise ValueError(f"Invalid loss: {config.loss_fn}.")


class MaskedGenealizedDiceFocalLoss(GeneralizedDiceFocalLoss):
    """Add an additional `masking` process before `GeneralizedDiceLoss`, accept a
    binary mask ([0, 1]) indicating a region,
    `input` and `target` will be masked by the region: region with mask `1`
    will keep the original value,
    region with `0` mask will be converted to `0`. Then feed `input` and `target`
    to normal `DiceLoss` computation.
    This has the effect of ensuring only the masked region contributes to the loss
    computation and hence gradient calculation.

    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Args follow :py:class:`monai.losses.GenealizedDiceLoss`."""
        super().__init__(*args, **kwargs)
        self.spatial_weighted = MaskedLoss(loss=super().forward)

    def forward(
        self, input: Tensor, target: Tensor, mask: Tensor | None = None
    ) -> Tensor:
        """Args:
        input: the shape should be BNH[WD].
        target: the shape should be BNH[WD].
        mask: the shape should B1H[WD] or 11H[WD].

        """
        return self.spatial_weighted(input=input, target=target, mask=mask)  # type: ignore[no-any-return]
