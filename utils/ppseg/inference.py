import numpy as np
import torch
from ignite.engine import Engine
from ignite.metrics.confusion_matrix import ConfusionMatrix, IoU, cmAccuracy, mIoU

from .dataset import get_mask


def inference(model, data, device="cpu", return_mask=True):
    """Inference function for a given model and data.
    Input:
        - model: PyTorch model.
        - data: Input data.
        - device: Device to use.
        - return_mask: Return the mask of the input data.

    Output:
        if return_mask is True:
            return outputs, pred, probs, mask
        else:
            return outputs, pred, probs
        - pred: Predicted label.
        - probs: Probability of the predicted label.
        - mask: Mask of the input data.
    """
    model.to(device)
    model.eval()
    data = torch.Tensor(data)

    with torch.no_grad():
        data = data.unsqueeze(0).to(device)
        outputs = model(data)
        outputs = outputs.cpu().squeeze(0).numpy()
        pred = outputs.argmax(axis=0)
        probs = outputs.max(axis=0)
        data = data.squeeze(0).cpu().numpy()

    if return_mask:
        return outputs, pred, probs, get_mask(data)
    else:
        return outputs, pred, probs


def mask_on_label(label, mask):
    """Apply a mask on a label.
    Input:
        - label (1, X, Y, Z): Label to apply the mask.
        - mask (X, Y, Z): Mask to apply on the label.

    Output:
        - Return the masked label ((1, N), in Tensor format,
        where N is the number of the points).
    """
    label = np.array(label).squeeze(0)
    mask = np.array(mask)

    return torch.Tensor(label[mask]).long().unsqueeze(0)


def mask_on_data(data, mask):
    """Apply a mask on data.
    Input:
        - data (C, X, Y, Z): data to apply the mask.
        - mask (X, Y, Z): Mask to apply on the data.

    Output:
        - Return the masked data ((1, C, N), in Tensor format,
        where N is the number of the points).
    """
    data = np.array(data)
    mask = np.array(mask)

    return torch.Tensor(data.transpose(1, 2, 3, 0)[mask].transpose(1, 0)).unsqueeze(0)


def evaluation_by_ignite(logits, gt_label):
    """Evaluation the result by PyTorch Ignite.
    Input:
        - logits: Predicted logits.
        - gt_label: Ground truth label.

    Output:
        - Print the evaluation results.
    """

    def eval_step(engine, batch):
        return batch

    default_evaluator = Engine(eval_step)

    metric = ConfusionMatrix(num_classes=7, device="cpu")
    metric.attach(default_evaluator, "cm")

    cm = ConfusionMatrix(num_classes=7, device="cpu")
    metric = mIoU(cm)
    metric.attach(default_evaluator, "mIoU")
    metric = IoU(cm)
    metric.attach(default_evaluator, "IoU")
    metric = cmAccuracy(cm)
    metric.attach(default_evaluator, "Accuracy")

    state = default_evaluator.run([[logits, gt_label]])
    print(f"Accuracy: {state.metrics['Accuracy']:.4f}")
    print(f"mIoU: {state.metrics['mIoU']:.4f}")
    print("IoU: ", [f"{each:.4f}" for each in state.metrics["IoU"]])
    return state
