from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def get_client_map(dataset):
    if dataset in ("mnist", "fashion"):
        clients_class_map = {
            "client_0": [0],
            "client_1": [1],
            "client_2": [2],
            "client_3": [3],
            "client_4": [4],
            "client_5": [5],
            "client_6": [6],
            "client_7": [7],
            "client_8": [8],
            "client_9": [9],
        }
    elif dataset == "cifar100":
        clients_class_map = {
            "client_0": [0, 1, 2, 3, 4],
            "client_1": [5, 6, 7, 8, 9],
            "client_2": [10, 11, 12, 13, 14],
            "client_3": [15, 16, 17, 18, 19],
            "client_4": [20, 21, 22, 23, 24],
            "client_5": [25, 26, 27, 28, 29],
            "client_6": [30, 31, 32, 33, 34],
            "client_7": [35, 36, 37, 38, 39],
            "client_8": [40, 41, 42, 43, 44],
            "client_9": [45, 46, 47, 48, 49],
            "client_10": [50, 51, 52, 53, 54],
            "client_11": [55, 56, 57, 58, 59],
            "client_12": [60, 61, 62, 63, 64],
            "client_13": [65, 66, 67, 68, 69],
            "client_14": [70, 71, 72, 73, 74],
            "client_15": [75, 76, 77, 78, 79],
            "client_16": [80, 81, 82, 83, 84],
            "client_17": [85, 86, 87, 88, 89],
            "client_18": [90, 91, 92, 93, 94],
            "client_19": [95, 96, 97, 98, 99],
        }
        # clients_class_map = {
        #     "client_0": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        #     "client_1": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        #     "client_2": [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
        #     "client_3": [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
        #     "client_4": [40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
        #     "client_5": [50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
        #     "client_6": [60, 61, 62, 63, 64, 65, 66, 67, 68, 69],
        #     "client_7": [70, 71, 72, 73, 74, 75, 76, 77, 78, 79],
        #     "client_8": [80, 81, 82, 83, 84, 85, 86, 87, 88, 89],
        #     "client_9": [90, 91, 92, 93, 94, 95, 96, 97, 98, 99],
        # }
    elif dataset in ("cifar10", "svhn"):
        clients_class_map = {
            "client_0": [0, 1],
            "client_1": [2, 3],
            "client_2": [4, 5],
            "client_3": [6, 7],
            "client_4": [8, 9],
        }
    elif dataset == "cifar10-2":
        client_class_map = {
            "client_0": [0, 1, 2],
            "client_1": [2, 3, 4],
            "client_2": [4, 5, 6],
            "client_3": [6, 7, 8],
            "client_4": [8, 9, 0],
        }
        num_clients = len(client_label_map)
    elif dataset == "cifar10-3":
        client_class_map = {
            "client_0": [0, 1, 2, 3],
            "client_1": [2, 3, 4, 5],
            "client_2": [4, 5, 6, 7],
            "client_3": [6, 7, 8, 9],
            "client_4": [8, 9, 0, 1],
        }
    elif dataset in ("bloodmnist", "tissuemnist"):
        clients_class_map = {
            "client_0": [0, 1],
            "client_1": [2, 3],
            "client_2": [4, 5],
            "client_3": [6, 7],
        }
    elif dataset == "dermamnist":
        clients_class_map = {
            "client_0": [0, 1],
            "client_1": [2, 3],
            "client_2": [4, 5, 6],
        }
    elif dataset == "pathmnist":
        clients_class_map = {
            "client_0": [0, 1, 2],
            "client_1": [3, 4, 5],
            "client_2": [6, 7, 8],
        }

    return clients_class_map


def load_scores(experiment: Path = None, dataset=None) -> dict:
    try:
        load = lambda p: torch.load(p, map_location=torch.device("cpu"))
        stage = "stage2" if "tct" in experiment.name else "stage1"

        val_scores = load(*(experiment / "scores").glob(f"*_{stage}_val_scores.pth"))
        val_targets = load(*(experiment / "scores").glob(f"*_{stage}_val_targets.pth"))
        test_scores = load(*(experiment / "scores").glob(f"*_{stage}_test_scores.pth"))
        test_targets = load(
            *(experiment / "scores").glob(f"*_{stage}_test_targets.pth")
        )
        return dict(
            val_scores=val_scores,
            val_targets=val_targets,
            test_scores=test_scores,
            test_targets=test_targets,
        )
    except Exception as e:
        print(e)
        return None


def get_new_trial(experiments, frac=0.5, fitzpatrick_df=None):
    # orig_val_scores = experiments["tct"]["val_scores"]
    # orig_val_targets = experiments["tct"]["val_targets"]
    # orig_test_scores = experiments["tct"]["test_scores"]
    # orig_test_targets = experiments["tct"]["test_targets"]
    orig_val_scores = experiments["fedavg"]["val_scores"]
    orig_val_targets = experiments["fedavg"]["val_targets"]
    orig_test_scores = experiments["fedavg"]["test_scores"]
    orig_test_targets = experiments["fedavg"]["test_targets"]
    orig_comb_scores = torch.concat([orig_val_scores, orig_test_scores])
    orig_comb_targets = torch.concat([orig_val_targets, orig_test_targets])
    assert orig_comb_scores.size(0) == orig_comb_targets.size(0)
    n = orig_comb_scores.size(0)
    rand_index = torch.randperm(n)
    k = int(frac * n)
    val_index = rand_index[:k]
    test_index = rand_index[k:]
    assert val_index.shape[0] + test_index.shape[0] == n
    new_experiments = {}
    for exp, v in experiments.items():
        # print(exp)
        val_scores = v["val_scores"]
        val_targets = v["val_targets"]
        test_scores = v["test_scores"]
        test_targets = v["test_targets"]
        comb_scores = torch.concat([val_scores, test_scores])
        comb_targets = torch.concat([val_targets, test_targets])
        assert (comb_targets == orig_comb_targets).all(), exp
        assert comb_targets.sum() == orig_comb_targets.sum(), exp
        new_experiments[exp] = {
            "val_scores": comb_scores[val_index],
            "val_targets": comb_targets[val_index],
            "test_scores": comb_scores[test_index],
            "test_targets": comb_targets[test_index],
        }
    if fitzpatrick_df is not None:
        val_df = fitzpatrick_df.copy().loc[val_index]
        test_df = fitzpatrick_df.copy().loc[test_index]
        return dict(experiments=new_experiments, val_df=val_df, test_df=test_df)
    else:
        return dict(experiments=new_experiments, val_df=None, test_df=None)


def combine_trials(trials):
    metrics = set(list(trials.values())[0].keys())
    mean_metrics = {met: defaultdict(list) for met in metrics}
    std_metrics = {met: defaultdict(list) for met in metrics}

    for trial in trials.values():
        for met, res in trial.items():
            # print(res)
            for alpha, val in res.items():
                mean_metrics[met][alpha].append(val)
                std_metrics[met][alpha].append(val)
        # break

    for met, dd in mean_metrics.items():
        mean_metrics[met] = {alpha: np.mean(values) for alpha, values in dd.items()}
    for met, dd in std_metrics.items():
        std_metrics[met] = {alpha: np.std(values) for alpha, values in dd.items()}

    return dict(mean=mean_metrics, std=std_metrics)
