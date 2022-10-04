import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm
import copy
import argparse


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(2048, 128)
        self.fc2 = nn.Linear(128, 10)

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
        output = F.log_softmax(x, dim=1)
        return output


def client_update(client_model, optimizer, train_loader, epoch=5):
    """Train a client_model on the train_loder data."""
    client_model.train()
    for e in range(epoch):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = client_model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
    return loss.item()


def average_models(global_model, client_models):
    """Average models across all clients."""
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack([client_models[i].state_dict()[k] for i in range(len(client_models))], 0).mean(0)
    global_model.load_state_dict(global_dict)


def evaluate_model(model, data_loader):
    """Compute loss and accuracy of a single model on a data_loader."""
    model.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    loss /= len(data_loader.dataset)
    acc = correct / len(data_loader.dataset)

    return loss, acc


def evaluate_many_models(models, data_loader):
    """Compute average loss and accuracy of multiple models on a data_loader."""
    num_nodes = len(models)
    losses = np.zeros(num_nodes)
    accuracies = np.zeros(num_nodes)
    for i in range(num_nodes):
        losses[i], accuracies[i] = evaluate_model(models[i], data_loader)
    return losses, accuracies


class Net_eNTK(nn.Module):
    def __init__(self):
        super(Net_eNTK, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(2048, 128)
        self.fc2 = nn.Linear(128, 10)

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
    """"compute eNTK"""
    model.eval()
    params = list(model.parameters())
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random_index = torch.randperm(355073)[:subsample_size]
    grads = None
    for i in tqdm(range(X.size()[0])):
        model.zero_grad()
        model.forward(X[i : i+1])[0].backward()

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


def scaffold_update(grads_data, targets, theta_client, h_i_client_pre, theta_global,
                    M=200, lr_local=0.00001):
    # set up data / eNTK
    grads_data = grads_data.float().cuda()
    targets = targets.cuda()

    # compute transformed onehot label
    targets_onehot = F.one_hot(targets, num_classes=10).cuda() - (1.0 / 10.0)
    num_samples = targets_onehot.shape[0]

    # compute updates
    h_i_client_update = h_i_client_pre + (1 / (M * lr_local)) * (theta_global - theta_client)
    theta_hat_local = (theta_global) * 1.0

    # local gd
    for local_iter in range(M):
        theta_hat_local -= lr_local * ((1.0 / num_samples) * grads_data.t() @ (grads_data @ theta_hat_local - targets_onehot) - h_i_client_update)

    del targets
    del grads_data
    torch.cuda.empty_cache()
    return theta_hat_local, h_i_client_update


def client_compute_eNTK(client_model, train_loader):
    """Train a client_model on the train_loder data."""
    client_model.train()

    data, targets = next(iter(train_loader))
    grads_data = compute_eNTK(client_model, data.cuda())
    grads_data = grads_data.float().cuda()
    targets = targets.cuda()
    # gradient
    targets_onehot = F.one_hot(targets, num_classes=10).cuda() - (1.0 / 10.0)
    del data
    torch.cuda.empty_cache()
    return grads_data, targets_onehot, targets

