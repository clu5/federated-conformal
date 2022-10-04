import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm
import copy
import argparse
import os
from utils import *

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
args = vars(parser.parse_args())

isExist = os.path.exists('./ckpt_stage1')
if not isExist:
   os.makedirs('./ckpt_stage1')

print('===================== Start TCT Stage-1 =====================')

# Hyperparameters
num_clients = args["num_client"]
num_rounds_stage1 = args["rounds_stage1"]
epochs_stage1 = args["local_epochs_stage1"]
batch_size_stage1 = args["mini_batchsize_stage1"]
lr_stage1 = args["local_lr_stage1"]

# Creating decentralized datasets
# NON-IID case:
# every client has images of two categories chosen from [0, 1], [2, 3], [4, 5], [6, 7], or [8, 9].
traindata = datasets.MNIST('./data_mnist', train=True, download=True,
                           transform=transforms.Compose([transforms.ToTensor(),
                                                         transforms.Normalize((0.1307,), (0.3081,))]))
target_labels = torch.stack([traindata.targets == i for i in range(10)])
target_labels_split = []
torch.manual_seed(args["seed"])
torch.cuda.manual_seed(args["seed"])
for i in range(num_clients):
    index_split = torch.where(target_labels[(2 * i):(2 * (i + 1))].sum(0))[0]
    perm_split = torch.randperm(index_split.size(0))
    index_split_subsample = index_split[perm_split[:args["num_samples_per_client"]]]
    target_labels_split += [index_split_subsample]

# Training datasets (subsampled)
traindata_split = [torch.utils.data.Subset(traindata, tl) for tl in target_labels_split]
train_loader = [torch.utils.data.DataLoader(train_subset, batch_size=batch_size_stage1, shuffle=True)
                for train_subset in traindata_split]
# Test dataset (subsampled)
testdata = datasets.MNIST('./data_mnist', train=False,
                          transform=transforms.Compose([transforms.ToTensor(),
                                                        transforms.Normalize((0.1307,), (0.3081,))]))

torch.manual_seed(args["seed"])
torch.cuda.manual_seed(args["seed"])
perm_split_test = torch.randperm(testdata.targets.shape[0])
testdata_subset = torch.utils.data.Subset(testdata, perm_split_test[:1000])
test_loader = torch.utils.data.DataLoader(testdata_subset, batch_size=batch_size_stage1, shuffle=False)

# Instantiate models and optimizers
global_model = Net().cuda()
client_models = [Net().cuda() for _ in range(num_clients)]
for model in client_models:
    model.load_state_dict(global_model.state_dict())
opt = [optim.SGD(model.parameters(), lr=lr_stage1) for model in client_models]

# Run TCT-Stage1 (i.e., FedAvg)
for r in range(num_rounds_stage1):
    # load global weights
    for model in client_models:
        model.load_state_dict(global_model.state_dict())

    # client update
    loss = 0
    for i in range(num_clients):
        loss += client_update(client_models[i], opt[i], train_loader[i], epoch=epochs_stage1)

    # average params across neighbors
    average_models(global_model, client_models)

    # evaluate
    test_losses, accuracies = evaluate_many_models(client_models, test_loader)
    torch.save(client_models[0].state_dict(), './ckpt_stage1/model_tct_stage1.pth')

    print('%d-th round: average train loss %0.3g | average test loss %0.3g | average test acc: %0.3f' % (
    r, loss / num_clients, test_losses.mean(), accuracies.mean()))


print('===================== Start TCT Stage-2 =====================')
# Hyperparameters
num_rounds_stage2 = args["rounds_stage2"]
batch_size = args["num_samples_per_client"]

# Init and load model ckpt
global_model = Net_eNTK().cuda()
global_model.load_state_dict(torch.load('./ckpt_stage1/model_tct_stage1.pth'))
global_model.fc2 = nn.Linear(128, 1).cuda()
print('load model')

# Compute eNTK representations
# Train
grad_all = []
target_all = []
target_onehot_all = []
for i in range(num_clients):
    grad_i, target_onehot_i, target_i = client_compute_eNTK(global_model, train_loader[i])
    grad_all.append(copy.deepcopy(grad_i).cpu())
    target_all.append(copy.deepcopy(target_i).cpu())
    target_onehot_all.append(copy.deepcopy(target_onehot_i).cpu())
    del grad_i
    del target_onehot_i
    del target_i
    torch.cuda.empty_cache()
# Test
grad_eval, target_eval_onehot, target_eval  = client_compute_eNTK(global_model, test_loader)

# Init linear models
theta_global = torch.zeros(100000, 10).cuda()
theta_global = torch.tensor(theta_global, requires_grad=False)
client_thetas = [torch.zeros_like(theta_global).cuda() for _ in range(num_clients)]
client_hi_s = [torch.zeros_like(theta_global).cuda() for _ in range(num_clients)]

# Run TCT-Stage2
for round_idx in range(num_rounds_stage2):
    theta_list = []
    for i in range(num_clients):
        theta_hat_update, h_i_client_update = scaffold_update(grad_all[i],
                                                              target_all[i],
                                                              client_thetas[i],
                                                              client_hi_s[i],
                                                              theta_global,
                                                              M=args["local_steps_stage2"],
                                                              lr_local=args["local_lr_stage2"])
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
    train_acc = targets_pred_train.eq(torch.cat(target_all).cuda()).sum() / (1.0 * logits_class_train.shape[0])
    # eval on test
    logits_class_test = grad_eval @ theta_global
    _, targets_pred_test = logits_class_test.max(1)
    test_acc = targets_pred_test.eq(target_eval.cuda()).sum() / (1.0 * logits_class_test.shape[0])
    print('Round %d: train accuracy=%0.5g test accuracy=%0.5g' % (round_idx, train_acc.item(), test_acc.item()))

print('===================== Finished TCT training =====================')

