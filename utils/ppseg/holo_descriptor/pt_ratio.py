import numpy as np
import pymesh


def get_nonbck_class_pt_ratio(class_pt_ratio: list | np.ndarray):
    class_pt_ratio = np.array(class_pt_ratio)
    if len(class_pt_ratio.shape) == 1:
        mask = class_pt_ratio[1:] != 0
        filtered = np.where(mask, class_pt_ratio[1:], np.nan)
        nonbck_class_pt_ratio = np.nanmean(filtered)
        nonbck_class_pt_ratio = np.nan_to_num(nonbck_class_pt_ratio)
        return nonbck_class_pt_ratio
    elif len(class_pt_ratio.shape) == 2:
        mask = class_pt_ratio[:, 1:] != 0
        filtered = np.where(mask, class_pt_ratio[:, 1:], np.nan)
        nonbck_class_pt_ratio = np.nanmean(filtered, axis=1)
        nonbck_class_pt_ratio = np.nan_to_num(nonbck_class_pt_ratio)
        return nonbck_class_pt_ratio
    else:
        raise ValueError("Invalid shape of class_pt_ratio")


def get_class_predprobs(ply_path, classes=7):
    regular_mesh = pymesh.load_mesh(ply_path)

    mask = regular_mesh.get_attribute("vertex_interest")
    pred = regular_mesh.get_attribute("vertex_pred")
    pred = pred[mask == 1]
    predprobs = regular_mesh.get_attribute("vertex_predprobs")
    predprobs = predprobs[mask == 1]

    class_probs = []
    class_pt_ratio = []
    for i in range(classes):
        temp = predprobs[pred == i]
        class_pt_ratio.append(len(temp) / len(predprobs) if len(temp) > 0 else 0)
        mean_probs = temp.mean() if len(temp) > 0 else 0
        class_probs.append(mean_probs)

    overall_predprobs = predprobs.mean()
    return class_probs, overall_predprobs, class_pt_ratio


def get_nonbck_ratio(class_pt_ratio: list | np.ndarray):
    class_pt_ratio = np.array(class_pt_ratio)
    if len(class_pt_ratio.shape) == 1:
        return 1 - class_pt_ratio[0]
    elif len(class_pt_ratio.shape) == 2:
        return 1 - class_pt_ratio[:, 0]
    else:
        raise ValueError("Invalid shape of class_pt_ratio")


def get_num_of_class_per_frame(class_predprobs: list | np.ndarray):
    class_predprobs = np.array(class_predprobs)
    if len(class_predprobs.shape) == 1:
        return np.sum(class_predprobs != 0)
    elif len(class_predprobs.shape) == 2:
        return np.sum(class_predprobs != 0, axis=1)
    else:
        raise ValueError("Invalid shape of class_predprobs")


def get_num_interest_points(ply_path):
    regular_mesh = pymesh.load_mesh(ply_path)
    mask = regular_mesh.get_attribute("vertex_interest")
    return len(mask[mask == 1])
