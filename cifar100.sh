#! /usr/bin/bash

# TCT
CUDA_VISIBLE_DEVICES=2 python src/run_TCT.py --dataset cifar100 --architecture small_resnet14 --rounds_stage1 100 --rounds_stage2 100 --local_epochs_stage1 5 --local_lr_stage1 0.01 --samples_per_client 10000 --local_steps_stage2 500 --local_lr_stage2 0.00001 --momentum 0.9 --use_data_augmentation --save_dir new_experiments

# FedAvg
#CUDA_VISIBLE_DEVICES=3 python src/run_TCT.py --dataset cifar100 --architecture small_resnet14 --rounds_stage1 200 --rounds_stage2 0 --local_epochs_stage1 5 --local_lr_stage1 0.01 --samples_per_client 10000 --momentum 0.9 --use_data_augmentation --save_dir new_experiments

# Centrally hosted
#CUDA_VISIBLE_DEVICES=2 python src/run_TCT.py --dataset cifar100 --architecture small_resnet14 --central --rounds_stage1 200 --rounds_stage2 0 --local_epochs_stage1 1 --local_lr_stage1 0.01 --samples_per_client 50000 --momentum 0.9 --use_data_augmentation --save_dir new_experiments

# TCT
#CUDA_VISIBLE_DEVICES=7 python src/run_TCT.py --dataset cifar100 --architecture small_resnet14 --rounds_stage1 100 --rounds_stage2 100 --local_epochs_stage1 5 --local_lr_stage1 0.01 --samples_per_client 10000 --local_steps_stage2 500 --local_lr_stage2 0.00001 --momentum 0.9 --use_data_augmentation --use_iid_partition --save_dir new_experiments

# FedAvg
#CUDA_VISIBLE_DEVICES=7 python src/run_TCT.py --dataset cifar100 --architecture small_resnet14 --rounds_stage1 200 --rounds_stage2 0 --local_epochs_stage1 5 --local_lr_stage1 0.01 --samples_per_client 10000 --momentum 0.9 --use_data_augmentation --use_iid_partition --save_dir new_experiments
