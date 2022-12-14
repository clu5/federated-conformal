import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt; plt.style.use('seaborn')
from tqdm import tqdm
import copy
import argparse
import os
import collections
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from pathlib import Path
from utils import (
    Net, Net_eNTK, client_update, average_models, evaluate_model,
    evaluate_many_models, compute_eNTK, scaffold_update,
    client_compute_eNTK,
)


def get_datasets(dataset_name: str, data_dir: str) -> dict[str, Dataset]:  # noqa: E501
    if dataset_name == 'mnist':
        _dataset = datasets.MNIST
        _transform = [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    elif dataset_name == 'fashion':
        _dataset = datasets.FashionMNIST
        _transform = [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    elif dataset_name == 'cifar':
        _dataset = datasets.CIFAR10
        _transform = [
            transforms.ToTensor(),
            transforms.Normalize((0.4913, 0.4821, 0.4465), (0.2470, 0.2434, 0.2615))  # noqa: E501
        ]
    else:
        raise ValueError(f'Dataset name: {dataset_name} not valid.'.center(20, '='))  # noqa: E501

    train_data = _dataset(data_dir, train=True, download=True, transform=transforms.Compose(_transform))  # noqa: E501
    test_data = _dataset(data_dir, train=False, download=True, transform=transforms.Compose(_transform))  # noqa: E501

    return {'train': train_data, 'test': test_data}


def partition_dataset(dataset: Dataset, client_label_map: dict[str, list[int]], samples_per_client: int) -> dict[str, Dataset]:  # noqa: E501
    targets = torch.tensor(dataset.targets)
    labels = targets.unique()
    bool_targets = torch.stack([targets == y for y in labels])

    client_datasets: dict[str, Dataset] = {}
    for client, labels in client_label_map.items():
        index = torch.where(bool_targets[labels].sum(0))[0]
        print(f'partition: {labels}\t {index.shape}')
        perm = torch.randperm(index.size(0))
        subsample = index[perm][:samples_per_client]
        assert subsample.size(0) != 0, f'Dataset error for client {client} with label {labels}'  # noqa: E501
        client_datasets[client] = Subset(dataset, subsample)

    return client_datasets


if __name__ == '__main__':
    """# Configuration"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_client', default=5, type=int)
    parser.add_argument('--seed', default=123, type=int)
    parser.add_argument('--num_samples_per_client', default=500, type=int)
    parser.add_argument('--rounds_stage1', default=100, type=int)
    parser.add_argument('--local_epochs_stage1', default=5, type=int)
    parser.add_argument('--mini_batchsize_stage1', default=64, type=int)
    parser.add_argument('--local_lr_stage1', default=0.1, type=float)
    parser.add_argument('--rounds_stage2', default=100, type=int)
    parser.add_argument('--local_steps_stage2', default=100, type=int)
    parser.add_argument('--local_lr_stage2', default=0.001, type=float)
    parser.add_argument('--data_dir', default='data', type=str)
    parser.add_argument('--checkpoint_dir', default='checkpoints', type=str)
    parser.add_argument('--dataset', default='mnist', type=str)
    parser.add_argument('--score_dir', default='scores', type=str)
    parser.add_argument('--figure_dir', default='plots', type=str)
    parser.add_argument('--resnet', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = vars(parser.parse_args())

    torch.manual_seed(args["seed"])
    torch.cuda.manual_seed(args["seed"])

    checkpoint_dir = Path(args['checkpoint_dir'])
    data_dir = Path(args['data_dir'])
    score_dir = Path(args['score_dir'])
    figure_dir = Path(args['figure_dir'])
    dataset_name = args['dataset']

    data_dir.mkdir(exist_ok=True, parents=True)

    checkpoint_dir = checkpoint_dir / dataset_name
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    score_dir = score_dir / dataset_name
    score_dir.mkdir(exist_ok=True, parents=True)

    figure_dir = figure_dir / dataset_name
    figure_dir.mkdir(exist_ok=True, parents=True)

    # Hyperparameters
    num_clients = args["num_client"]
    num_rounds_stage1 = args["rounds_stage1"]
    num_rounds_stage2 = args["rounds_stage2"]
    epochs_stage1 = args["local_epochs_stage1"]
    batch_size_stage1 = args["mini_batchsize_stage1"]
    lr_stage1 = args["local_lr_stage1"]
    samples_per_client = args['num_samples_per_client']
    batch_size = args['mini_batchsize_stage1']
    num_local_steps = args['local_steps_stage2']
    lr_stage2 = args['local_lr_stage2']
    use_resnet = args['resnet']

    debug = args['debug']  # debugging mode
    if debug:
        num_rounds_stage1 = 5
        num_rounds_stage2 = 5
        samples_per_client = 100
        batch_size = 8

    if num_clients == 1:
        save_name = 'central'
    elif num_rounds_stage2 == 0:
        save_name = 'fedavg'
    else:
        save_name = 'tct'

    in_channels = 3 if dataset_name == 'cifar' else 1
    out_channels = 10

    if save_name == 'central':
        client_label_map = {'central_server': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}
        samples_per_client *= num_clients
    else:
        client_label_map = {
            'client_0': [0, 1],
            'client_1': [2, 3],
            'client_2': [4, 5],
            'client_3': [6, 7],
            'client_4': [8, 9],
        }

    if use_resnet:
        save_name = f'resnet_{save_name}'
    else:
        save_name = f'cnn_{save_name}'


    _datasets = get_datasets(dataset_name, data_dir)
    client_train_datasets = partition_dataset(_datasets['train'], client_label_map, samples_per_client)  # noqa: E501
    train_loader = [DataLoader(train_subset, batch_size=batch_size, shuffle=True) for train_subset in client_train_datasets.values()]  # noqa: E501

    test_dataset = _datasets['test']
    val_split: float = 0.1
    num_val = round(val_split * len(test_dataset))
    rand_index = torch.randperm(len(test_dataset))
    val_index = rand_index[:num_val][:1000]
    test_index = rand_index[num_val:][:1000]
    val_subsample = Subset(test_dataset, val_index.tolist())
    test_subsample = Subset(test_dataset, test_index.tolist())

    val_loader = DataLoader(val_subsample, batch_size=batch_size, shuffle=False)  # noqa: E501
    test_loader = DataLoader(test_subsample, batch_size=batch_size, shuffle=False)  # noqa: E501

    print('train (one client) samples'.ljust(20, '-'), batch_size * len(train_loader[0]))
    print('val samples'.ljust(20, '-'), batch_size * len(val_loader))
    print('test samples'.ljust(20, '-'), batch_size * len(test_loader))

    def make_model(use_resnet, in_channels=1, out_channels=10):
        if use_resnet:
            model = models.resnet18()
            model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            model.fc = nn.Linear(in_features=512, out_features=out_channels, bias=True)
        else:
            model = Net_eNTK(in_channels, out_channels)
        return model

    print('===================== Start TCT Stage-1 =====================')

    # Instantiate models and optimizers
    global_model = make_model(use_resnet, in_channels).cuda()
    client_models = [make_model(use_resnet, in_channels).cuda() for _ in range(num_clients)]
    for model in client_models:
        model.load_state_dict(global_model.state_dict())
    opt = [optim.SGD(model.parameters(), lr=lr_stage1) for model in client_models]  # noqa: E501

    stage1_losses = collections.defaultdict(list)
    stage1_accuracies = []

    # Run TCT-Stage1 (i.e., FedAvg)
    for r in range(num_rounds_stage1):
        # load global weights
        for model in client_models:
            model.load_state_dict(global_model.state_dict())

        # client update
        loss = 0
        for i in range(num_clients):
            loss += client_update(client_models[i], opt[i], train_loader[i], epoch=epochs_stage1)  # noqa: E501

        # average params across neighbors
        average_models(global_model, client_models)

        # save global model
        torch.save(global_model.state_dict(), checkpoint_dir / f'{save_name}_stage1_model.pth')  # noqa: E501

        # evaluate
        test_losses, accuracies = evaluate_many_models(client_models, test_loader)  # noqa: E501

        stage1_losses['train'].append(loss / num_clients)
        stage1_losses['test'].append(test_losses.mean())
        stage1_accuracies.append(accuracies.mean())

        print(f'{r}d-th round: average train loss %0.3g | average test loss %0.3g | average test acc: %0.3f' % (
                loss / num_clients, test_losses.mean(), accuracies.mean()))

    fig, ax = plt.subplots(ncols=2, figsize=(16, 5))
    train_acc = np.arange(0, 1, 0.1) + 0.1
    test_acc = np.arange(0, 1, 0.1)
    fontsize = 24
    ax[0].plot(stage1_losses['train'], label='train')
    ax[0].plot(stage1_losses['test'], label='test')
    ax[0].set_xlabel('round', fontsize=fontsize)
    ax[0].set_ylabel('loss', fontsize=fontsize)
    ax[0].legend(fontsize=fontsize)
    ax[1].plot(stage1_accuracies, label='client test')
    ax[1].set_xlabel('round', fontsize=fontsize)
    ax[1].set_ylabel('accuracy', fontsize=fontsize)
    ax[1].legend(fontsize=fontsize)
    plt.savefig(figure_dir / f'{save_name}_stage1_loss_curve.png')

    val_loss, val_acc, val_scores, val_targets = evaluate_model(global_model, val_loader, return_logits=True)  # noqa: E501
    test_loss, test_acc, test_scores, test_targets = evaluate_model(global_model, test_loader, return_logits=True)  # noqa: E501

    if not debug:
        torch.save(val_scores, score_dir / f'{save_name}_stage1_val_scores.pth')
        torch.save(val_targets, score_dir / f'{save_name}_stage1_val_targets.pth')
        torch.save(test_scores, score_dir / f'{save_name}_stage1_test_scores.pth')
        torch.save(test_targets, score_dir / f'{save_name}_stage1_test_targets.pth')

    print(f'global model -- val loss: {val_loss} -- val acc: {val_acc} -- test_loss: {test_loss} -- test acc: {test_acc}')

    if num_rounds_stage2 != 0:
        print(' Start TCT Stage-2 '.center(20, '='))
        checkpoint  = checkpoint_dir / f'{save_name}_stage1_model.pth'

        # Init and load model ckpt

        global_model = make_model(use_resnet, in_channels, out_channels).cuda()
        global_model.load_state_dict(torch.load(checkpoint))  # noqa: E501

        # Replace last layer
        if use_resnet:
            global_model.fc = nn.Linear(in_features=512, out_features=1, bias=True).cuda()
        else:
            global_model.fc2 = nn.Linear(128, 1).cuda()
        print('loaded eNTK model')


        # Compute eNTK representations

        # Train
        grad_all = []
        target_all = []
        target_onehot_all = []
        for i in range(num_clients):
            grad_i, target_onehot_i, target_i = client_compute_eNTK(global_model, train_loader[i])  # noqa: E501
            grad_all.append(copy.deepcopy(grad_i).cpu())
            target_all.append(copy.deepcopy(target_i).cpu())
            target_onehot_all.append(copy.deepcopy(target_onehot_i).cpu())
            del grad_i
            del target_onehot_i
            del target_i
            torch.cuda.empty_cache()

        # Test
        grad_eval, target_eval_onehot, target_eval = client_compute_eNTK(global_model, test_loader, num_batches=len(test_loader))  # noqa: E501

        # For calibration
        grad_val, target_val_onehot, target_val = client_compute_eNTK(global_model, val_loader, num_batches=len(val_loader))  # noqa: E501

        # Init linear models
        theta_global = torch.zeros(100000, 10).cuda()
        theta_global = torch.tensor(theta_global, requires_grad=False)
        client_thetas = [torch.zeros_like(theta_global).cuda() for _ in range(num_clients)]  # noqa: E501
        client_hi_s = [torch.zeros_like(theta_global).cuda() for _ in range(num_clients)]  # noqa: E501

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
                )
                client_hi_s[i] = h_i_client_update * 1.0
                client_thetas[i] = theta_hat_update * 1.0
                theta_list.append(theta_hat_update)

            # averaging
            theta_global = torch.zeros_like(theta_list[0]).cuda()
            for theta_idx in range(num_clients):
                theta_global += (1.0 / num_clients) * theta_list[theta_idx]

            # eval on train
            logits_class_train = torch.cat(grad_all).cuda() @ theta_global
            _, targets_pred_train = logits_class_train.max(1)
            train_acc = targets_pred_train.eq(torch.cat(target_all).cuda()).sum() / (1.0 * logits_class_train.shape[0])  # noqa: E501

            # eval on test
            logits_class_test = grad_eval @ theta_global
            _, targets_pred_test = logits_class_test.max(1)
            test_acc = targets_pred_test.eq(target_eval.cuda()).sum() / (1.0 * logits_class_test.shape[0])  # noqa: E501
            print('Round %d: train accuracy=%0.5g test accuracy=%0.5g' % (round_idx, train_acc.item(), test_acc.item()))  # noqa: E501

            # eval on val
            logits_class_val = grad_val @ theta_global
            _, targets_pred_val = logits_class_val.max(1)

        if not debug:
            torch.save(logits_class_val, score_dir / f'{save_name}_stage2_val_scores.pth')
            torch.save(target_val, score_dir / f'{save_name}_stage2_val_targets.pth')
            torch.save(logits_class_test, score_dir / f'{save_name}_stage2_test_scores.pth')
            torch.save(target_eval, score_dir / f'{save_name}_stage2_test_targets.pth')

        print(' Finished TCT Stage-2 '.center(20, '='))
