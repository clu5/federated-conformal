import argparse
import collections
import copy
import json
import logging
import os
import sys
import time
from pathlib import Path
from statistics import mean, stdev

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import matplotlib.pyplot as plt
import medmnist
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import torchvision
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, models, transforms
from tqdm import tqdm

from skin_dataset import (TEST_TRANSFORM, TRAIN_TRANSFORM, SkinDataset,
                          get_weighted_sampler)
from utils import (Net, Net_eNTK, average_models, client_update, compute_eNTK,
                   evaluate_many_models, evaluate_model, get_datasets,
                   make_model, partition_dataset, replace_last_layer,
                   scaffold_update)

plt.style.use("seaborn")


def main():
    """Main script for TCT, FedAvg, and Centrally hosted experiments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_client", default=5, type=int)
    parser.add_argument("--seed", default=123, type=int)
    parser.add_argument("--samples_per_client", default=1000, type=int)
    parser.add_argument("--rounds_stage1", default=50, type=int)
    parser.add_argument("--local_epochs_stage1", default=5, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--local_lr_stage1", default=0.01, type=float)
    parser.add_argument("--rounds_stage2", default=100, type=int)
    parser.add_argument("--local_steps_stage2", default=200, type=int)
    parser.add_argument("--local_lr_stage2", default=0.00001, type=float)
    parser.add_argument("--data_dir", default="data", type=str)
    parser.add_argument("--save_dir", default="experiments", type=str)
    parser.add_argument("--dataset", default="fashion", type=str)
    parser.add_argument("--architecture", default="cnn", type=str)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--start_from_stage2", action="store_true")
    parser.add_argument("--num_workers", default=16, type=int)
    parser.add_argument("--num_test_samples", default=10000, type=int)
    parser.add_argument("--use_iid_partition", action="store_true")
    parser.add_argument("--central", action="store_true")
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--use_data_augmentation", action="store_true")
    parser.add_argument("--fitzpatrick_csv", default="csv/fitzpatrick.csv", type=str)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--num_random_grad", default=100000, type=int)
    parser.add_argument("--start_from_stage1", action="store_true")
    parser.add_argument(
        "--fitzpatrick_image_dir",
        default="/u/luchar/data/fitzpatrick17k/images",
        type=str,
    )
    args = vars(parser.parse_args())

    start_time = time.perf_counter()

    # Set random seed for Repeatability
    seed = args["seed"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Hyperparameters
    num_clients = args["num_client"]
    num_rounds_stage1 = args["rounds_stage1"]
    num_rounds_stage2 = args["rounds_stage2"]
    local_epochs_stage1 = args["local_epochs_stage1"]
    lr_stage1 = args["local_lr_stage1"]
    samples_per_client = args["samples_per_client"]
    batch_size = args["batch_size"]
    num_local_steps = args["local_steps_stage2"]
    lr_stage2 = args["local_lr_stage2"]
    architecture = args["architecture"]
    start_from_stage1 = args["start_from_stage1"]
    start_from_stage2 = args["start_from_stage2"]
    num_workers = args["num_workers"]
    num_test_samples = args["num_test_samples"]
    use_iid_partition = args["use_iid_partition"]
    central = args["central"]
    momentum = args["momentum"]
    use_data_augmentation = args["use_data_augmentation"]
    fitzpatrick_csv = args["fitzpatrick_csv"]
    fitzpatrick_image_dir = args["fitzpatrick_image_dir"]
    pretrained = args["pretrained"]
    num_random_grad = args["num_random_grad"]

    debug = args["debug"]  # debugging mode
    dataset_name = args["dataset"]

    if dataset_name == "mnist":
        in_channels = 1
        num_classes = 10
        num_clients = 5
        client_label_map = {
            "client_1": [0, 1],
            "client_2": [2, 3],
            "client_3": [4, 5],
            "client_4": [6, 7],
            "client_5": [8, 9],
        }
    elif dataset_name == "svhn":
        in_channels = 3
        num_classes = 10
        num_clients = 5
        client_label_map = {
            "client_1": [0, 1],
            "client_2": [2, 3],
            "client_3": [4, 5],
            "client_4": [6, 7],
            "client_5": [8, 9],
        }
    elif dataset_name == "fashion":
        in_channels = 1
        num_classes = 10
        num_clients = 5
        client_label_map = {
            "client_1": [0, 1],
            "client_2": [2, 3],
            "client_3": [4, 5],
            "client_4": [6, 7],
            "client_5": [8, 9],
        }
    elif dataset_name == "cifar10":
        in_channels = 3
        num_classes = 10
        num_clients = 5
        client_label_map = {
            "client_1": [0, 1],
            "client_2": [2, 3],
            "client_3": [4, 5],
            "client_4": [6, 7],
            "client_5": [8, 9],
        }
    elif dataset_name == "cifar100":
        in_channels = 3
        num_classes = 100
        num_clients = 10
        client_label_map = {
            "client_1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "client_2": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            "client_3": [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
            "client_4": [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
            "client_5": [40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
            "client_6": [50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
            "client_7": [60, 61, 62, 63, 64, 65, 66, 67, 68, 69],
            "client_8": [70, 71, 72, 73, 74, 75, 76, 77, 78, 79],
            "client_9": [80, 81, 82, 83, 84, 85, 86, 87, 88, 89],
            "client_10": [90, 91, 92, 93, 94, 95, 96, 97, 98, 99],
        }
    elif dataset_name == "bloodmnist":
        in_channels = 3
        num_classes = 8
        num_clients = 4
        client_label_map = {
            "client_1": [0, 1],
            "client_2": [2, 3],
            "client_3": [4, 5],
            "client_4": [6, 7],
        }
    elif dataset_name == "dermamnist":
        in_channels = 3
        num_classes = 7
        num_clients = 3
        client_label_map = {
            "client_1": [0, 1],
            "client_2": [2, 3],
            "client_3": [4, 5, 6],
        }
    elif dataset_name == "pathmnist":
        in_channels = 3
        num_classes = 9
        num_clients = 4
        client_label_map = {
            "client_1": [0, 1],
            "client_2": [2, 3],
            "client_3": [4, 5],
            "client_4": [6, 7, 8],
        }
    elif dataset_name == "tissuemnist":
        in_channels = 1
        num_classes = 8
        num_clients = 4
        client_label_map = {
            "client_1": [0, 1],
            "client_2": [2, 3],
            "client_3": [4, 5],
            "client_4": [6, 7],
        }
    elif dataset_name == "fitzpatrick":
        in_channels = 3
        num_classes = 114
        num_clients = 12
        client_label_map = None
    else:
        raise ValueError(f'dataset "{dataset_name}" not supported')

    if central:
        experiment = "central"
        num_clients = 1
        local_epochs_stage1 = 1
        num_rounds_stage2 = 0
        client_label_map = {"central_server": list(range(num_classes))}
    elif num_rounds_stage2 == 0:
        experiment = "fedavg"
    else:
        experiment = "tct"

    save_name = f"{dataset_name}_{experiment}_{architecture}"

    if use_iid_partition:
        save_name = save_name + "_iid_partition"

    if pretrained:
        save_name = save_name + "_pretrained"

    if debug:
        save_name = "debug_" + save_name
        num_rounds_stage1 = 5
        num_rounds_stage2 = 5
        samples_per_client = 100
        batch_size = 8
        num_test_samples = 100

    # Make directory to write outputs
    save_dir = Path(args["save_dir"]) / save_name
    data_dir = Path(args["data_dir"])

    if (save_dir / "finished.txt").exists():
        exit()

    save_dir.mkdir(exist_ok=True, parents=True)
    data_dir.mkdir(exist_ok=True, parents=True)

    checkpoint_dir = save_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    score_dir = save_dir / "scores"
    score_dir.mkdir(exist_ok=True, parents=True)

    figure_dir = save_dir / "figures"
    figure_dir.mkdir(exist_ok=True, parents=True)

    if not debug:
        with open(save_dir / "commands.txt", "w") as f:
            json.dump(args, f, indent=4)

    # Setup logging to write console output to file
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # print to stdout
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    # write to file
    file_handler = logging.FileHandler(save_dir / "console.log")
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    logger.info(f" {save_name.upper()} ".center(50, "="))

    if dataset_name == "fitzpatrick":
        df = pd.read_csv(fitzpatrick_csv)
        skin_types = sorted(df.aggregated_fitzpatrick_scale.unique())
        if not central:
            assert len(skin_types) == num_clients
        if use_iid_partition:
            train_df = df.query("split == 'train'").sample(frac=1, random_state=seed)
            samples_per_client = round(len(train_df) / num_clients)
            train_partition = {
                str(i): train_df[i * samples_per_client : (i + 1) * samples_per_client]
                for i in range(num_clients)
            }
        else:
            if central:
                train_partition = {"central": df.query("split == 'train'")}
            else:
                train_partition = {
                    str(st): df.query(
                        "aggregated_fitzpatrick_scale == @st and split == 'train'"
                    )
                    for st in skin_types
                }
        val_df = df.query("split == 'val'")
        test_df = df.query("split == 'test'")
        samplers = {
            str(st): get_weighted_sampler(df) for st, df in train_partition.items()
        }

        train_datasets = {
            str(st): SkinDataset(
                image_dir=fitzpatrick_image_dir,
                label_mapping=dict(df[["md5hash", "target"]].values),
                transform=TRAIN_TRANSFORM if use_data_augmentation else TEST_TRANSFORM,
            )
            for st, df in train_partition.items()
        }
        val_map = dict(val_df[["md5hash", "target"]].values)
        val_dataset = SkinDataset(
            image_dir=fitzpatrick_image_dir,
            label_mapping=val_map,
            transform=TEST_TRANSFORM,
        )
        test_map = dict(test_df[["md5hash", "target"]].values)
        test_dataset = SkinDataset(
            image_dir=fitzpatrick_image_dir,
            label_mapping=test_map,
            transform=TEST_TRANSFORM,
        )

        train_loaders = [
            DataLoader(
                ds,
                batch_size=batch_size,
                sampler=samplers[st],
                num_workers=num_workers,
                pin_memory=True,
            )
            for st, ds in train_datasets.items()
        ]
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=False,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=False,
        )

        val_items = list(val_map.items())
        test_items = list(test_map.items())
        if val_dataset[0][1] != val_items[0][1]:
            raise ValueError(f"{val_dataset[0][1]=} {val_items[0]=}")
        if val_dataset[-1][1] != val_items[-1][1]:
            raise ValueError(f"{val_dataset[-1][1]=} != {val_items[-1]=}")
        if test_dataset[0][1] != test_items[0][1]:
            raise ValueError(f"{test_dataset[0][1]=} != {test_items[0]=}")
        if test_dataset[-1][1] != test_items[-1][1]:
            raise ValueError(f"{test_dataset[-1][1]=} != {test_items[-1]=}")
        val_df.to_csv(save_dir / "val_df.csv")
        test_df.to_csv(save_dir / "test_df.csv")

    else:
        _datasets = get_datasets(
            dataset_name, data_dir, use_data_augmentation=use_data_augmentation
        )
        client_train_datasets = partition_dataset(
            _datasets["train"],
            client_label_map,
            samples_per_client,
            use_iid_partition=use_iid_partition,
            seed=seed,
        )
        train_loaders = [
            DataLoader(
                train_subset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
            )
            for train_subset in client_train_datasets.values()
        ]

        test_dataset = _datasets["test"]

        if dataset_name in ("bloodmnist", "pathmnist", "tissuemnist"):
            val_dataset = _datasets["val"]
            val_index = torch.randperm(len(val_dataset))[:1000]
            test_index = torch.randperm(len(test_dataset))[:num_test_samples]
            val_subsample = Subset(val_dataset, val_index.tolist())
            test_subsample = Subset(test_dataset, test_index.tolist())
        else:
            val_split: float = 0.1
            num_val = round(val_split * len(test_dataset))
            rand_index = torch.randperm(len(test_dataset))
            val_index = rand_index[:num_val]
            test_index = rand_index[num_val:][:num_test_samples]
            val_subsample = Subset(test_dataset, val_index.tolist())
            test_subsample = Subset(test_dataset, test_index.tolist())

        val_loader = DataLoader(
            val_subsample,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_subsample,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    logger.debug(f"{len(val_loader)=}")
    logger.debug(f"{len(test_loader)=}")
    logger.debug(f"{len(val_loader.dataset)=}")
    logger.debug(f"{len(test_loader.dataset)=}")

    for i, loader in enumerate(train_loaders):
        logger.info(
            f"client {i} train samples".ljust(20, "-") + f" {len(loader.dataset)}"
        )
    logger.info("val samples".ljust(20, "-") + f" {len(val_loader.dataset)}")
    logger.info("test samples".ljust(20, "-") + f" {len(test_loader.dataset)}")

    if not start_from_stage2:
        logger.info("===================== Start Stage-1 =====================")

        # Instantiate models and optimizers
        global_model = torch.nn.DataParallel(
            make_model(architecture, in_channels, num_classes)
        ).cuda()
        client_models = [
            torch.nn.DataParallel(
                make_model(architecture, in_channels, num_classes)
            ).cuda()
            for _ in range(num_clients)
        ]

        if start_from_stage1:
            checkpoint = max(
                checkpoint_dir.glob(f"{save_name}_stage1_model_*.pth"),
                key=lambda x: int(x.stem.split("_")[-1]),
            )
            global_model.module.load_state_dict(torch.load(checkpoint))
            r = int(checkpoint.stem.split("_")[-1])
            logger.info(f"starting stage 1 with {checkpoint} at round {r}")
        else:
            r = 1

        for model in client_models:
            model.module.load_state_dict(global_model.module.state_dict())
        opt = [
            optim.SGD(
                model.parameters(),
                lr=lr_stage1,
                momentum=momentum,
            )
            for model in client_models
        ]

        logger.info(f"{len(client_models)=}")
        logger.info(f"{len(train_loaders)=}")
        logger.info(f"{opt[0]=}")

        stage1_loss = collections.defaultdict(list)
        stage1_accuracy = collections.defaultdict(list)
        stage1_max_score = collections.defaultdict(list)

        # Run TCT-Stage1 (i.e., FedAvg)
        for r in range(r, num_rounds_stage1 + 1):

            # load global weights
            for model in client_models:
                model.module.load_state_dict(global_model.module.state_dict())

            # client update
            loss = 0
            for i in range(num_clients):
                print(".", end="")
                loss += client_update(
                    client_models[i],
                    opt[i],
                    train_loaders[i],
                    epoch=local_epochs_stage1,
                )
            loss /= num_clients

            # average params across neighbors
            average_models(global_model, client_models)

            # save global model
            torch.save(
                global_model.module.state_dict(),
                checkpoint_dir / f"{save_name}_stage1_model_{r}.pth",
            )

            # evaluate clients
            clients_train_loss, clients_train_acc = 0, 0
            for i in range(num_clients):
                train_loss, train_acc = evaluate_model(
                    client_models[i],
                    train_loaders[i],
                    num_batches=8,
                )
                clients_train_loss += train_loss
                clients_train_acc += train_acc
            clients_train_loss /= num_clients
            clients_train_acc /= num_clients

            stage1_loss["clients_train"].append(clients_train_loss)
            stage1_accuracy["clients_train"].append(clients_train_acc)

            # save mean and variance of client accuracy and max score

            clients_test_loss = 0
            clients_test_acc = []
            clients_test_max_score = []
            for i, model in enumerate(client_models):
                test_loss, test_acc, logits, _ = evaluate_model(
                    model,
                    test_loader,
                    num_batches=12,
                    return_logits=True,
                )
                clients_test_loss += test_loss
                clients_test_acc.append(test_acc)
                clients_test_max_score.append(torch.softmax(logits, 1).argmax(1))

            clients_test_loss /= num_clients
            clients_test_max_score = torch.cat(clients_test_max_score).cpu().tolist()

            stage1_loss["clients_test"].append(clients_test_loss)
            stage1_accuracy["clients_test_mean"].append(
                float(np.mean(clients_test_acc))
            )
            stage1_accuracy["clients_test_std"].append(float(np.std(clients_test_acc)))

            # TODO debug max score (larger than 1)
            stage1_max_score["clients_test_mean"].append(
                float(np.mean(clients_test_max_score))
            )
            stage1_max_score["clients_test_std"].append(
                float(np.std(clients_test_max_score))
            )

            # evaluate global model
            global_train_loss, global_train_acc = 0, 0
            for i in range(num_clients):
                train_loss, train_acc = evaluate_model(
                    global_model, train_loaders[i], num_batches=8
                )
                global_train_loss += train_loss
                global_train_acc += train_acc
            global_train_loss /= num_clients
            global_train_acc /= num_clients

            global_test_loss, global_test_acc = evaluate_model(
                global_model,
                test_loader,
                num_batches=8,
            )

            stage1_loss["global_train"].append(global_train_loss)
            stage1_loss["global_test"].append(global_test_loss)
            stage1_accuracy["global_train"].append(global_train_acc)
            stage1_accuracy["global_test"].append(global_test_acc)

            logger.info(
                f"{str(r).zfill(3)} == clients =="
                f" {loss:.3f} train loss {clients_train_loss:.3f} "
                f"| test loss {clients_test_loss:.3f} "
                f"| train acc {clients_train_acc:.3f} "
                f"| test acc {mean(clients_test_acc):.3f} "
                f"   == global model =="
                f"  train loss {global_train_loss:.3f} "
                f"| test loss {global_test_loss:.3f} "
                f"| train acc {global_train_acc:.3f} "
                f"| test acc {global_test_acc:.3f} "
            )

            if r % 50 == 0:
                val_loss, val_acc, val_scores, val_targets = evaluate_model(
                    global_model,
                    val_loader,
                    num_batches=len(val_loader),
                    return_logits=True,
                )
                test_loss, test_acc, test_scores, test_targets = evaluate_model(
                    global_model,
                    test_loader,
                    num_batches=len(test_loader),
                    return_logits=True,
                )
                logger.info(
                    f"global model -- {val_loss=:.3f} -- {val_acc=:.3f} -- {test_loss=:.3f} -- {test_acc=:.3f}"
                )

                torch.save(val_scores, score_dir / f"{save_name}_stage1_val_scores.pth")
                torch.save(
                    val_targets, score_dir / f"{save_name}_stage1_val_targets.pth"
                )
                torch.save(
                    test_scores, score_dir / f"{save_name}_stage1_test_scores.pth"
                )
                torch.save(
                    test_targets, score_dir / f"{save_name}_stage1_test_targets.pth"
                )

        with open(save_dir / "history.json", "w") as f:
            json.dump(
                dict(
                    loss=stage1_loss,
                    accuracy=stage1_accuracy,
                    score=stage1_max_score,
                ),
                f,
                indent=4,
                default=float,
            )

        fig, ax = plt.subplots(ncols=2, figsize=(16, 5))
        fontsize = 24
        ax[0].plot(stage1_loss["clients_train"], ".:", label="clients_train")
        ax[0].plot(stage1_loss["clients_test"], ".:", label="clients_test")
        ax[0].plot(stage1_loss["global_train"], "o--", label="global_train")
        ax[0].plot(stage1_loss["global_test"], "o--", label="global_test")
        ax[0].set_xlabel("round", fontsize=fontsize)
        ax[0].set_ylabel("loss", fontsize=fontsize)
        ax[0].legend(fontsize=fontsize)
        ax[1].plot(stage1_accuracy["clients_train"], ".:", label="clients_train")
        ax[1].plot(stage1_accuracy["clients_test_mean"], ".:", label="clients_test")
        ax[1].plot(stage1_accuracy["global_train"], "o--", label="global_train")
        ax[1].plot(stage1_accuracy["global_test"], "o--", label="global_test")
        ax[1].set_xlabel("round", fontsize=fontsize)
        ax[1].set_ylabel("accuracy", fontsize=fontsize)
        ax[1].legend(fontsize=fontsize)
        plt.savefig(figure_dir / f"{save_name}_stage1_loss_curve.png")

    if num_rounds_stage2 != 0 and experiment == "tct":

        # turn off data augmentation in training datasets
        if dataset_name == "fitzpatrick":
            if use_iid_partition:
                train_df = df.query("split == 'train'").sample(
                    frac=1, random_state=seed
                )
                samples_per_client = round(len(train_df) / num_clients)
                train_partition = {
                    str(i): train_df[
                        i * samples_per_client : (i + 1) * samples_per_client
                    ]
                    for i in range(num_clients)
                }
            else:
                train_partition = {
                    str(st): df.query(
                        "aggregated_fitzpatrick_scale == @st and split == 'train'"
                    )
                    for st in skin_types
                }
            train_datasets = {
                str(st): SkinDataset(
                    image_dir=fitzpatrick_image_dir,
                    label_mapping=dict(df[["md5hash", "target"]].values),
                    transform=TEST_TRANSFORM,
                )
                for st, df in train_partition.items()
            }
            train_loaders = [
                DataLoader(
                    ds,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                )
                for st, ds in train_datasets.items()
            ]
        else:
            _datasets = get_datasets(
                dataset_name, data_dir, use_data_augmentation=False
            )
            client_train_datasets = partition_dataset(
                _datasets["train"],
                client_label_map,
                samples_per_client,
                use_iid_partition=use_iid_partition,
                seed=seed,
            )
            train_loaders = [
                DataLoader(
                    train_subset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                )
                for train_subset in client_train_datasets.values()
            ]

        logger.info(" Start Stage-2 ".center(20, "="))
        checkpoint = checkpoint_dir / f"{save_name}_stage1_model.pth"
        logger.info(f"{checkpoint=}")

        # Init and load model ckpt
        global_model = make_model(architecture, in_channels, num_classes).cuda()
        global_model.load_state_dict(torch.load(checkpoint))
        global_model = global_model.cuda()

        # TODO find out why train accuracy is so low here
        train_loss, train_acc = 0, 0
        for i in range(num_clients):
            train_loss, train_acc = evaluate_model(
                global_model,
                train_loaders[i],
                num_batches=len(train_loaders[i]),
            )
            train_loss += train_loss
            train_acc += train_acc
        train_loss /= num_clients
        train_acc /= num_clients

        test_loss, test_acc, test_scores, test_targets = evaluate_model(
            global_model, test_loader, return_logits=True, num_batches=len(test_loader)
        )
        logger.info(
            f"stage2 model -- {train_loss=:.3f} -- {train_acc=:.3f} -- {test_loss=:.3f} -- {test_acc=:.3f}"
        )

        global_model = replace_last_layer(global_model, architecture, num_classes=1)
        global_model = global_model.cuda()
        logger.info("loaded eNTK model")

        logger.info("Compute eNTK representations")
        # Train
        grad_all = []
        target_all = []
        for i in range(num_clients):
            grad_i, target_i = compute_eNTK(
                global_model,
                train_loaders[i],
                subsample_size=num_random_grad,
                seed=seed,
                num_classes=num_classes,
            )
            grad_all.append(copy.deepcopy(grad_i).cpu())
            target_all.append(copy.deepcopy(target_i).cpu())
            del grad_i
            del target_i
            torch.cuda.empty_cache()

        assert len(grad_all) == num_clients

        # Test
        grad_test, target_test = compute_eNTK(
            global_model,
            test_loader,
            subsample_size=num_random_grad,
            seed=seed,
            num_classes=num_classes,
        )

        # For calibration
        grad_val, target_val = compute_eNTK(
            global_model,
            val_loader,
            subsample_size=num_random_grad,
            seed=seed,
            num_classes=num_classes,
        )

        # normalization
        logger.info("normalization")
        scaler = StandardScaler()
        # TODO log normalization scale
        # logger.info(f"{len(grad_all[0])=}")
        # scaler.fit(torch.cat([grad[torch.randperm(grad.shape[0])[:20000] for grad in grad_all]]).cpu().numpy())
        scaler.fit(torch.cat(grad_all).cpu().numpy())
        for idx in range(len(grad_all)):
            print(".", end="")
            grad_all[idx] = torch.from_numpy(
                scaler.transform(grad_all[idx].cpu().numpy())
            )
        grad_test = torch.from_numpy(scaler.transform(grad_test.cpu().numpy()))
        grad_val = torch.from_numpy(scaler.transform(grad_val.cpu().numpy()))

        # Init linear models
        theta_global = torch.zeros(num_random_grad, num_classes)
        theta_global = torch.tensor(theta_global, requires_grad=False)
        client_thetas = [torch.zeros_like(theta_global) for _ in range(num_clients)]
        client_hi_s = [torch.zeros_like(theta_global) for _ in range(num_clients)]

        # Run TCT-Stage2
        for round_idx in range(num_rounds_stage2):
            theta_list = []
            for i in range(num_clients):
                print(".", end="")
                theta_hat_update, h_i_client_update = scaffold_update(
                    grad_all[i],
                    target_all[i],
                    client_thetas[i],
                    client_hi_s[i],
                    theta_global,
                    M=num_local_steps,
                    lr_local=lr_stage2,
                    num_classes=num_classes,
                )
                client_hi_s[i] = h_i_client_update * 1.0
                client_thetas[i] = theta_hat_update * 1.0
                theta_list.append(theta_hat_update)

            # averaging
            theta_global = torch.zeros_like(theta_list[0])
            for theta_idx in range(num_clients):
                theta_global += (1.0 / num_clients) * theta_list[theta_idx]

            # eval on train
            logits_class_train = torch.cat(grad_all).cpu() @ theta_global.cpu()
            target_train = torch.cat(target_all).cpu()
            train_acc = (
                (logits_class_train.argmax(1) == target_train.cpu()).sum()
                / logits_class_train.shape[0]
            ).item()
            train_max_score = logits_class_train.max(1).values.mean().item()

            # eval on val
            logits_class_val = grad_val.cpu() @ theta_global.cpu()

            # eval on test
            logits_class_test = grad_test.cpu() @ theta_global.cpu()
            test_acc = (
                (logits_class_test.argmax(1) == target_test.cpu()).sum()
                / logits_class_test.shape[0]
            ).item()
            test_max_score = logits_class_test.max(1).values.mean().item()

            logger.info(f"{round_idx=}: {train_acc=:.3f} {test_acc=:.3f}")

            # logger.debug(f"{torch.cat(grad_all).shape=}")
            # logger.debug(f"{grad_test.shape=}")
            # logger.debug(f"{theta_global.shape=}")
            # logger.debug(f"{train_max_score=:.3f} {test_max_score=:.3f}")

        if not debug:
            torch.save(
                logits_class_train, score_dir / f"{save_name}_stage2_train_scores.pth"
            )
            torch.save(
                target_train, score_dir / f"{save_name}_stage2_train_targets.pth"
            )
            torch.save(
                logits_class_val, score_dir / f"{save_name}_stage2_val_scores.pth"
            )
            torch.save(target_val, score_dir / f"{save_name}_stage2_val_targets.pth")
            torch.save(
                logits_class_test, score_dir / f"{save_name}_stage2_test_scores.pth"
            )
            torch.save(target_test, score_dir / f"{save_name}_stage2_test_targets.pth")

        logger.info(" Finished TCT Stage-2 ".center(20, "="))

    end_time = time.perf_counter()
    total_runtime = end_time - start_time
    logger.info(f"total runtime {total_runtime:.0f}")

    if not debug:
        with open(save_dir / "finished.txt", "w") as f:
            f.write(f"total runtime: {total_runtime:.0f}")


if __name__ == "__main__":
    main()
