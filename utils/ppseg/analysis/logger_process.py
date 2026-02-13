import os

import numpy as np
import pandas as pd
from natsort import natsorted
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def tabulate_events(d, columns=None):
    df = pd.DataFrame(d)
    if columns is not None:
        for each in columns:
            assert each in df.columns, f"Column {each} not in dataframe"
        df = df[columns]
        return df
    else:
        return df


def best_results(results_list, which="validation_mIoU"):
    """results_list: list of dataframes"""
    best_results = []
    for results in results_list:
        best_results.append(results[which].max())
    _print_best_results(best_results, which)
    return best_results


def _print_best_results(best_results, which="validation_mIoU"):
    print(f"{which}: {np.mean(best_results):.3f} ± {np.std(best_results):.3f}")
    return None


def read_results(path):
    acc = EventAccumulator(path)

    acc.Reload()
    # acc.Tags()

    results = pd.DataFrame(
        [
            tabulate_events(acc.Scalars("training/loss"), columns=["value"])[
                "value"
            ].values,
            tabulate_events(acc.Scalars("validation/loss"), columns=["value"])[
                "value"
            ].values,
            tabulate_events(acc.Scalars("training/accuracy"), columns=["value"])[
                "value"
            ].values,
            tabulate_events(acc.Scalars("validation/accuracy"), columns=["value"])[
                "value"
            ].values,
            tabulate_events(acc.Scalars("training/miou"), columns=["value"])[
                "value"
            ].values,
            tabulate_events(acc.Scalars("validation/miou"), columns=["value"])[
                "value"
            ].values,
        ]
    ).T
    results.columns = [
        "training_loss",
        "validation_loss",
        "training_accuracy",
        "validation_accuracy",
        "training_mIoU",
        "validation_mIoU",
    ]

    return results


def read_cv_results(dir_path, machine="marvin"):
    """Read the results of a cross-validation experiment"""
    filenames = [each for each in os.listdir(dir_path) if each.endswith(f".{machine}")]
    filenames = natsorted(filenames)
    results_list = []
    for filename in filenames:
        results = read_results(f"{dir_path}/{filename}")
        results_list.append(results)
    return results_list


def avg_training_process(data_list):
    data_cv_avg, data_cv_std = {}, {}
    for data in data_list:
        for col in data.columns:
            if col not in data_cv_avg:
                data_cv_avg[col] = []
            data_cv_avg[col].append(data[col].values)

    for key in data_cv_avg:
        data_cv_std[key] = np.std(data_cv_avg[key], axis=0)
        data_cv_avg[key] = np.mean(data_cv_avg[key], axis=0)

    return pd.DataFrame(data_cv_avg), pd.DataFrame(data_cv_std)
