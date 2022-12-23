import argparse
import copy

import medmnist
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, models, transforms
from tqdm import tqdm

from resnet import (small_resnet20, small_resnet32, small_resnet44,
                    small_resnet56, small_resnet110)


def get_datasets(dataset_name: str, data_dir: str) -> dict[str, Dataset]:  # noqa: E501
    if dataset_name == "mnist":
        construct_dataset = datasets.MNIST
        train_transform = [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
        test_transform = [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    elif dataset_name == "fashion":
        construct_dataset = datasets.FashionMNIST
        train_transform = [
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ]
        test_transform = [
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ]
    elif dataset_name == "cifar10":
        construct_dataset = datasets.CIFAR10
        train_transform = [
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4913, 0.4821, 0.4465), (0.2470, 0.2434, 0.2615)),
        ]
        test_transform = [
            transforms.ToTensor(),
            transforms.Normalize((0.4913, 0.4821, 0.4465), (0.2470, 0.2434, 0.2615)),
        ]
    elif dataset_name == "cifar100":
        construct_dataset = datasets.CIFAR100
        train_transform = [
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        ]
        test_transform = [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        ]
    elif dataset_name == "bloodmnist":
        info = medmnist.INFO["bloodmnist"]
        construct_dataset = getattr(medmnist, info["python_class"])
        train_transform = [
            transforms.ToTensor(),
            transforms.Normalize((0.7943, 0.6597, 0.6962), (0.2156, 0.2416, 0.1179)),
        ]
        test_transform = [
            transforms.ToTensor(),
            transforms.Normalize((0.7943, 0.6597, 0.6962), (0.2156, 0.2416, 0.1179)),
        ]
        target_transform = transforms.Lambda(lambda x: x[0])
    elif dataset_name == "pathmnist":
        info = medmnist.INFO["pathmnist"]
        construct_dataset = getattr(medmnist, info["python_class"])
        train_transform = [
            transforms.ToTensor(),
            transforms.Normalize((0.7405, 0.533, 0.7058), (0.1237, 0.1768, 0.1244)),
        ]
        test_transform = [
            transforms.ToTensor(),
            transforms.Normalize((0.7405, 0.533, 0.7058), (0.1237, 0.1768, 0.1244)),
        ]
        target_transform = transforms.Lambda(lambda x: x[0])
    elif dataset_name == "tissuemnist":
        info = medmnist.INFO["tissuemnist"]
        construct_dataset = getattr(medmnist, info["python_class"])
        train_transform = [
            transforms.ToTensor(),
            transforms.Normalize((0.1020,), (0.1000,)),
        ]
        test_transform = [
            transforms.ToTensor(),
            transforms.Normalize((0.1020,), (0.1000,)),
        ]
        target_transform = transforms.Lambda(lambda x: x[0])
    else:
        raise ValueError(f"Dataset name: {dataset_name} not valid.".center(20, "="))

    if dataset_name in ("bloodmnist", "pathmnist", "tissuemnist"):
        train_data = construct_dataset(
            split="train",
            root=data_dir,
            download=True,
            transform=transforms.Compose(train_transform),
            target_transform=target_transform,
        )
        val_data = construct_dataset(
            split="val",
            root=data_dir,
            download=True,
            transform=transforms.Compose(test_transform),
            target_transform=target_transform,
        )
        test_data = construct_dataset(
            split="test",
            root=data_dir,
            download=True,
            transform=transforms.Compose(test_transform),
            target_transform=target_transform,
        )
        return {"train": train_data, "val": val_data, "test": test_data}
    else:
        train_data = construct_dataset(
            data_dir,
            train=True,
            download=True,
            transform=transforms.Compose(train_transform),
        )
        test_data = construct_dataset(
            data_dir,
            train=False,
            download=True,
            transform=transforms.Compose(test_transform),
        )

        return {"train": train_data, "test": test_data}


def partition_dataset(
    dataset: Dataset,
    client_label_map: dict[str, list[int]],
    samples_per_client: int,
    use_iid_partition: bool = False,
) -> dict[str, Dataset]:
    client_datasets: dict[str, Dataset] = {}

    if use_iid_partition:
        perm = torch.randperm(len(dataset))
        for i, client in enumerate(client_label_map.keys()):
            client_datasets[client] = Subset(
                dataset, perm[i * samples_per_client : (i + 1) * samples_per_client]
            )
    else:
        if hasattr(dataset, "targets"):
            targets = torch.tensor(dataset.targets)
        elif hasattr(dataset, "labels"):
            targets = torch.tensor(dataset.labels)
        else:
            raise ValueError("Cannot find labels")
        labels = targets.unique()
        bool_targets = torch.stack([targets == y for y in labels])

        for client, labels in client_label_map.items():
            index = torch.where(bool_targets[labels].sum(0))[0]
            perm = torch.randperm(index.size(0))
            subsample = index[perm][:samples_per_client]
            assert (
                subsample.size(0) != 0
            ), f"Dataset error for client {client} with label {labels}"
            client_datasets[client] = Subset(dataset, subsample)

    return client_datasets


def init_params(net):
    """Init layer parameters."""
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal(m.weight, mode="fan_out")
            if m.bias:
                torch.nn.init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            torch.nn.init.constant(m.weight, 1)
            torch.nn.init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            torch.nn.init.normal(m.weight, std=1e-3)
            if m.bias:
                torch.nn.init.constant(m.bias, 0)


def make_model(architecture, in_channels=1, num_classes=10):
    if architecture == "cnn":
        model = Net_eNTK(in_channels, num_classes)
    elif architecture == "small_resnet20":
        model = small_resnet20(in_channels=in_channels, num_classes=num_classes)
    elif architecture == "small_resnet32":
        model = small_resnet32(in_channels=in_channels, num_classes=num_classes)
    elif architecture == "small_resnet44":
        model = small_resnet44(in_channels=in_channels, num_classes=num_classes)
    elif architecture == "small_resnet56":
        model = small_resnet56(in_channels=in_channels, num_classes=num_classes)
    elif architecture == "resnet34":
        model = models.resnet34()
        model.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        model.bn1 = nn.BatchNorm2d(in_channels)
        model.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)
    elif architecture == "efficientnet-b0":
        model = models.efficientnet_b0()
        model.features[0][0] = nn.Conv2d(
            in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False
        )
        model.classifier[1] = nn.Linear(
            in_features=1280, out_features=num_classes, bias=True
        )
    elif architecture == "efficientnet-b1":
        model = models.efficientnet_b1()
        model.features[0][0] = nn.Conv2d(
            in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False
        )
        model.classifier[1] = nn.Linear(
            in_features=1280, out_features=num_classes, bias=True
        )
    elif architecture == "efficientnet-b2":
        model = models.efficientnet_b2()
        model.features[0][0] = nn.Conv2d(
            in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False
        )
        model.classifier[1] = nn.Linear(
            in_features=1280, out_features=num_classes, bias=True
        )
    elif architecture == "efficientnet-b3":
        model = models.efficientnet_b3()
        model.features[0][0] = nn.Conv2d(
            in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False
        )
        model.classifier[1] = nn.Linear(
            in_features=1280, out_features=num_classes, bias=True
        )
    elif architecture == "efficientnet-b4":
        model = models.efficientnet_b4()
        model.features[0][0] = nn.Conv2d(
            in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False
        )
        model.classifier[1] = nn.Linear(
            in_features=1280, out_features=num_classes, bias=True
        )
    else:
        raise ValueError(f'Architecture "{architecture}" not supported.')
    return model


def replace_last_layer(model, architecture, num_classes=1):
    if architecture == "cnn":
        model.fc2 = nn.Linear(128, num_classes).cuda()
    elif architecture == "resnet":
        model.fc = nn.Linear(
            in_features=512, out_features=num_classes, bias=True
        ).cuda()
    elif architecture == "efficientnet":
        model.classifier[1] = nn.Linear(
            in_features=1280, out_features=num_classes, bias=True
        )
    else:
        raise ValueError(f'Architecture "{architecture}" not supported.')

    return model


class Net(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 10):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(2048, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        # output = F.log_softmax(x, dim=1)
        # return output
        return x


def client_update(client_model, optimizer, train_loader, epoch=5):
    """Train a client_model on the train_loder data."""
    client_model.train()
    for e in range(epoch):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = client_model(data)
            score = F.log_softmax(output, dim=1)
            loss = F.nll_loss(score, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    return total_loss / len(train_loader)


def average_models(global_model, client_models):
    """Average models across all clients."""
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = (
            torch.stack(
                [client_models[i].state_dict()[k] for i in range(len(client_models))], 0
            )
            .float()
            .mean(0)
        )
    global_model.load_state_dict(global_dict)


def evaluate_model(model, data_loader, return_logits=False, num_batches=1):
    """Compute loss and accuracy of a single model on a data_loader."""
    model.eval()
    loss = 0
    correct = 0
    total = 0
    logits, targets = [], []
    with torch.no_grad():
        for i, (data, target) in enumerate(data_loader):
            if i > num_batches:
                break
            data, target = data.cuda(), target.cuda()
            output = model(data)
            score = F.log_softmax(output, dim=1)
            loss += F.nll_loss(
                score, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = score.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += data.shape[0]

            if return_logits:
                logits.append(output.detach().cpu())
                targets.append(target.detach().cpu())

    loss /= (i + 1) * data_loader.batch_size
    acc = correct / total

    if return_logits:
        return loss, acc, torch.cat(logits), torch.cat(targets)
    else:
        return loss, acc


def evaluate_many_models(models, data_loader, num_batches=1):
    """Compute average loss and accuracy of multiple models on a data_loader."""
    num_models = len(models)
    total_loss = 0
    total_accuracy = 0
    for i in range(num_models):
        loss, accuracy = evaluate_model(models[i], data_loader, num_batches=num_batches)
        total_loss += loss
        total_accuracy += accuracy
    return total_loss / num_models, total_accuracy / num_models


class Net_eNTK(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 10):
        super(Net_eNTK, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(2048, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


def compute_eNTK(model, X, subsample_size=100000, seed=123):
    """ "compute eNTK"""
    model.eval()
    params = list(model.parameters())
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("num_params:", num_params)
    random_index = torch.randperm(num_params)[:subsample_size]
    grads = None
    for i in tqdm(range(X.size()[0])):
        model.zero_grad()
        model.forward(X[i : i + 1])[0].backward()

        grad = []
        for param in params:
            if param.requires_grad:
                grad.append(param.grad.flatten())
        grad = torch.cat(grad)
        grad = grad[random_index]

        if grads is None:
            grads = torch.zeros((X.size()[0], grad.size()[0]), dtype=torch.half)
        grads[i, :] = grad

    return grads


def scaffold_update(
    grads_data,
    targets,
    theta_client,
    h_i_client_pre,
    theta_global,
    M=200,
    lr_local=0.00001,
    num_classes=10,
):
    # set up data / eNTK
    grads_data = grads_data.float().cuda()
    targets = targets.cuda()

    # compute transformed onehot label
    targets_onehot = F.one_hot(targets, num_classes=num_classes).cuda() - (
        1.0 / num_classes
    )
    num_samples = targets_onehot.shape[0]

    # compute updates
    h_i_client_update = h_i_client_pre + (1 / (M * lr_local)) * (
        theta_global - theta_client
    )
    theta_hat_local = (theta_global) * 1.0

    # local gd
    for local_iter in range(M):
        theta_hat_local -= lr_local * (
            (1.0 / num_samples)
            * grads_data.t()
            @ (grads_data @ theta_hat_local - targets_onehot)
            - h_i_client_update
        )

    del targets
    del grads_data
    torch.cuda.empty_cache()
    return theta_hat_local, h_i_client_update


def client_compute_eNTK(client_model, loader, num_batches=1, seed=123, num_classes=10):
    """Train a client_model on the train_loder data."""
    it = iter(loader)
    data = []
    targets = []
    for _ in range(num_batches):
        _data, _targets = next(it)
        data.append(_data)
        targets.append(_targets)

    data = torch.cat(data)
    targets = torch.cat(targets)

    grads_data = compute_eNTK(client_model, data.cuda(), seed=seed)
    grads_data = grads_data.float().cuda()
    targets = targets.cuda()

    # gradient
    targets_onehot = F.one_hot(targets, num_classes=num_classes).cuda() - (
        1.0 / num_classes
    )
    del data
    torch.cuda.empty_cache()
    return grads_data, targets_onehot, targets
