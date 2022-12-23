import argparse
import collections
import copy
import json
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import matplotlib.pyplot as plt
import medmnist
import numpy as np
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

from utils import (Net, Net_eNTK, average_models, client_compute_eNTK,
                   client_update, compute_eNTK, evaluate_many_models,
                   evaluate_model, get_datasets, make_model, partition_dataset,
                   replace_last_layer, scaffold_update)

plt.style.use("seaborn")


if __name__ == "__main__":
    """Configuration"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_client", default=5, type=int)
    parser.add_argument("--seed", default=123, type=int)
    parser.add_argument("--samples_per_client", default=1000, type=int)
    parser.add_argument("--rounds_stage1", default=50, type=int)
    parser.add_argument("--local_epochs_stage1", default=5, type=int)
    parser.add_argument("--mini_batchsize_stage1", default=128, type=int)
    parser.add_argument("--local_lr_stage1", default=0.1, type=float)
    parser.add_argument("--rounds_stage2", default=100, type=int)
    parser.add_argument("--local_steps_stage2", default=200, type=int)
    parser.add_argument("--local_lr_stage2", default=0.00001, type=float)
    parser.add_argument("--data_dir", default="data", type=str)
    parser.add_argument("--save_dir", default="experiments", type=str)
    parser.add_argument("--dataset", default="fashion", type=str)
    parser.add_argument("--architecture", default="cnn", type=str)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--start_from_stage2", action="store_true")
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--num_test_samples", default=5000, type=int)
    parser.add_argument("--use_iid_partition", action="store_true")
    parser.add_argument("--central", action="store_true")
    args = vars(parser.parse_args())

    seed = args["seed"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Hyperparameters
    num_clients = args["num_client"]
    num_rounds_stage1 = args["rounds_stage1"]
    num_rounds_stage2 = args["rounds_stage2"]
    local_epochs_stage1 = args["local_epochs_stage1"]
    batch_size_stage1 = args["mini_batchsize_stage1"]
    lr_stage1 = args["local_lr_stage1"]
    samples_per_client = args["samples_per_client"]
    batch_size = args["mini_batchsize_stage1"]
    num_local_steps = args["local_steps_stage2"]
    lr_stage2 = args["local_lr_stage2"]
    architecture = args["architecture"]
    start_from_stage2 = args["start_from_stage2"]
    num_workers = args["num_workers"]
    num_test_samples = args["num_test_samples"]
    use_iid_partition = args["use_iid_partition"]
    central = args["central"]

    debug = args["debug"]  # debugging mode
    dataset_name = args["dataset"]

    if dataset_name == "fashion":
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

    if debug:
        save_name = "debug_" + save_name
        num_rounds_stage1 = 5
        num_rounds_stage2 = 5
        samples_per_client = 100
        batch_size = 8
        num_test_samples = 100

    print(f" {save_name} ".center(40, "="))

    save_dir = Path(args["save_dir"]) / save_name
    data_dir = Path(args["data_dir"])

    save_dir.mkdir(exist_ok=True, parents=True)
    data_dir.mkdir(exist_ok=True, parents=True)

    checkpoint_dir = save_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    score_dir = save_dir / "scores"
    score_dir.mkdir(exist_ok=True, parents=True)

    figure_dir = save_dir / "figures"
    figure_dir.mkdir(exist_ok=True, parents=True)

    with open(save_dir / "commands.txt", "w") as f:
        json.dump(args, f, indent=4)

    _datasets = get_datasets(dataset_name, data_dir)
    client_train_datasets = partition_dataset(
        _datasets["train"],
        client_label_map,
        samples_per_client,
        use_iid_partition=use_iid_partition,
    )
    train_loaders = [
        DataLoader(
            train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers
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
        val_index = rand_index[:num_val][:num_test_samples]
        test_index = rand_index[num_val:][:num_test_samples]
        val_subsample = Subset(test_dataset, val_index.tolist())
        test_subsample = Subset(test_dataset, test_index.tolist())

    val_loader = DataLoader(
        val_subsample, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_subsample, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    if debug:
        print(f"{val_index.shape=}")
        print(f"{test_index.shape=}")
        print(f"{len(val_loader)=}")
        print(f"{len(test_loader)=}")
        print(f"{len(val_loader.dataset)=}")
        print(f"{len(test_loader.dataset)=}")

    for i, loader in enumerate(train_loaders):
        print(f"client {i} train samples".ljust(20, "-"), len(loader.dataset))
    print("val samples".ljust(20, "-"), len(val_loader.dataset))
    print("test samples".ljust(20, "-"), len(test_loader.dataset))

    if not start_from_stage2:
        print("===================== Start TCT Stage-1 =====================")

        # Instantiate models and optimizers
        global_model = make_model(architecture, in_channels, num_classes).cuda()
        client_models = [
            make_model(architecture, in_channels, num_classes).cuda()
            for _ in range(num_clients)
        ]
        for model in client_models:
            model.load_state_dict(global_model.state_dict())
        opt = [optim.SGD(model.parameters(), lr=lr_stage1) for model in client_models]

        stage1_loss = collections.defaultdict(list)
        stage1_accuracy = collections.defaultdict(list)

        # Run TCT-Stage1 (i.e., FedAvg)
        for r in range(num_rounds_stage1):

            # load global weights
            for model in client_models:
                model.load_state_dict(global_model.state_dict())

            # client update
            loss = 0
            for i in range(num_clients):
                loss += client_update(
                    client_models[i],
                    opt[i],
                    train_loaders[i],
                    epoch=local_epochs_stage1,
                )

            # average params across neighbors
            average_models(global_model, client_models)

            # save global model
            torch.save(
                global_model.state_dict(),
                checkpoint_dir / f"{save_name}_stage1_model.pth",
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

            clients_test_loss, clients_test_acc = evaluate_many_models(
                client_models,
                test_loader,
                num_batches=8,
            )

            stage1_loss["clients_train"].append(clients_train_loss)
            stage1_loss["clients_test"].append(clients_test_loss)
            stage1_accuracy["clients_train"].append(clients_train_acc)
            stage1_accuracy["clients_test"].append(clients_test_acc)

            print(
                f"{str(r).zfill(3)} == clients =="
                f"  train loss {clients_train_loss:.3f} "
                f"| test loss {clients_test_loss:.3f} "
                f"| train acc {clients_train_acc:.3f} "
                f"| test acc {clients_test_acc:.3f} ",
                end="\t",
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
                global_model, test_loader, num_batches=8
            )

            stage1_loss["global_train"].append(global_train_loss)
            stage1_loss["global_test"].append(global_test_loss)
            stage1_accuracy["global_train"].append(global_train_acc)
            stage1_accuracy["global_test"].append(global_test_acc)

            print(
                f"\t== global model =="
                f"  train loss {global_train_loss:.3f} "
                f"| test loss {global_test_loss:.3f} "
                f"| train acc {global_train_acc:.3f} "
                f"| test acc {global_test_acc:.3f} "
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
        ax[1].plot(stage1_accuracy["clients_test"], ".:", label="clients_test")
        ax[1].plot(stage1_accuracy["global_train"], "o--", label="global_train")
        ax[1].plot(stage1_accuracy["global_test"], "o--", label="global_test")
        ax[1].set_xlabel("round", fontsize=fontsize)
        ax[1].set_ylabel("accuracy", fontsize=fontsize)
        ax[1].legend(fontsize=fontsize)
        plt.savefig(figure_dir / f"{save_name}_stage1_loss_curve.png")

        val_loss, val_acc, val_scores, val_targets = evaluate_model(
            global_model, val_loader, return_logits=True, num_batches=len(val_loader)
        )
        test_loss, test_acc, test_scores, test_targets = evaluate_model(
            global_model, test_loader, return_logits=True, num_batches=len(test_loader)
        )

        if not debug:
            torch.save(val_scores, score_dir / f"{save_name}_stage1_val_scores.pth")
            torch.save(val_targets, score_dir / f"{save_name}_stage1_val_targets.pth")
            torch.save(test_scores, score_dir / f"{save_name}_stage1_test_scores.pth")
            torch.save(test_targets, score_dir / f"{save_name}_stage1_test_targets.pth")

        print(
            f"global model -- {val_loss=:.3f} -- {val_acc=:.3f} -- {test_loss=:.3f} -- {test_acc=:.3f}"
        )

    if num_rounds_stage2 != 0 and experiment == "tct":
        print(" Start TCT Stage-2 ".center(20, "="))
        checkpoint = checkpoint_dir / f"{save_name}_stage1_model.pth"

        # Init and load model ckpt
        global_model = make_model(architecture, in_channels, num_classes).cuda()
        global_model.load_state_dict(torch.load(checkpoint))

        global_model = replace_last_layer(global_model, architecture, num_classes=1)
        global_model = global_model.cuda()

        print("loaded eNTK model")

        # Compute eNTK representations

        # Train
        grad_all = []
        target_all = []
        for i in range(num_clients):
            grad_i, target_onehot_i, target_i = client_compute_eNTK(
                global_model,
                train_loaders[i],
                num_batches=len(train_loaders[i]),
                seed=seed,
                num_classes=num_classes,
            )
            grad_all.append(copy.deepcopy(grad_i).cpu())
            target_all.append(copy.deepcopy(target_i).cpu())
            del grad_i
            del target_onehot_i
            del target_i
            torch.cuda.empty_cache()

        # Test
        grad_test, target_test_onehot, target_test = client_compute_eNTK(
            global_model, test_loader, num_batches=len(test_loader), seed=seed
        )

        # For calibration
        grad_val, target_val_onehot, target_val = client_compute_eNTK(
            global_model, val_loader, num_batches=len(val_loader), seed=seed
        )

        # normalization
        scaler = StandardScaler()
        scaler.fit(torch.cat(grad_all).cpu().numpy())
        for idx in range(len(grad_all)):
            grad_all[idx] = torch.from_numpy(
                scaler.transform(grad_all[idx].cpu().numpy())
            )
        grad_test = torch.from_numpy(scaler.transform(grad_test.cpu().numpy())).cuda()
        grad_val = torch.from_numpy(scaler.transform(grad_val.cpu().numpy())).cuda()

        # Init linear models
        theta_global = torch.zeros(100000, num_classes).cuda()
        theta_global = torch.tensor(theta_global, requires_grad=False)
        client_thetas = [
            torch.zeros_like(theta_global).cuda() for _ in range(num_clients)
        ]
        client_hi_s = [
            torch.zeros_like(theta_global).cuda() for _ in range(num_clients)
        ]

        # Run TCT-Stage2
        for round_idx in range(num_rounds_stage2):
            theta_list = []
            for i in range(num_clients):
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
            theta_global = torch.zeros_like(theta_list[0]).cuda()
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

            print(f"{round_idx=}: {train_acc=:.3f} {test_acc=:.3f}")

            if debug:
                print(f"{torch.cat(grad_all).shape=}", torch.cat(grad_all).max())
                print(f"{grad_test.shape=}", grad_test.max())
                print(f"{theta_global.shape=}", theta_global.max())
                print(f"{train_max_score=:.3f} {test_max_score=:.3f}")

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

        print(" Finished TCT Stage-2 ".center(20, "="))
