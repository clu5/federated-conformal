#! /usr/bin/bash

# TCT
CUDA_VISIBLE_DEVICES=0 python src/run_TCT.py --dataset fitzpatrick --architecture resnet18 --rounds_stage1 100 --rounds_stage2 100 --local_epochs_stage1 5 --local_lr_stage1 0.01 --local_steps_stage2 500 --local_lr_stage2 0.00001 --momentum 0.9 --batch_size 64 --use_data_augmentation --save_dir new_experiments

# FedAvg
CUDA_VISIBLE_DEVICES=0 python src/run_TCT.py --dataset fitzpatrick --architecture resnet18 --rounds_stage1 200 --rounds_stage2 0 --local_epochs_stage1 5 --local_lr_stage1 0.01 --momentum 0.9 --batch_size 64 --use_data_augmentation --save_dir new_experiments

# Centrally hosted
CUDA_VISIBLE_DEVICES=0 python src/run_TCT.py --dataset fitzpatrick --architecture resnet18 --central --rounds_stage1 200 --rounds_stage2 0 --local_epochs_stage1 1 --local_lr_stage1 0.01 --momentum 0.9 --batch_size 64 --use_data_augmentation --save_dir new_experiments

# TCT (IID)
CUDA_VISIBLE_DEVICES=0 python src/run_TCT.py --dataset fitzpatrick --architecture resnet18 --rounds_stage1 100 --rounds_stage2 100 --local_epochs_stage1 5 --local_lr_stage1 0.01 --local_steps_stage2 500 --local_lr_stage2 0.00001 --momentum 0.9 --batch_size 64 --use_data_augmentation --use_iid_partition --save_dir new_experiments

# FedAvg (IID)
CUDA_VISIBLE_DEVICES=0 python src/run_TCT.py --dataset fitzpatrick --architecture resnet18 --rounds_stage1 200 --rounds_stage2 0 --local_epochs_stage1 5 --local_lr_stage1 0.01 --momentum 0.9 --batch_size 64 --use_data_augmentation --use_iid_partition --save_dir new_experiments

# TCT
CUDA_VISIBLE_DEVICES=0 python src/run_TCT.py --dataset fitzpatrick --architecture efficientnet-b0 --rounds_stage1 100 --rounds_stage2 100 --local_epochs_stage1 5 --local_lr_stage1 0.01 --local_steps_stage2 500 --local_lr_stage2 0.00001 --momentum 0.9 --batch_size 64 --use_data_augmentation --save_dir new_experiments

# FedAvg
CUDA_VISIBLE_DEVICES=0 python src/run_TCT.py --dataset fitzpatrick --architecture efficientnet-b0 --rounds_stage1 200 --rounds_stage2 0 --local_epochs_stage1 5 --local_lr_stage1 0.01 --momentum 0.9 --batch_size 64 --use_data_augmentation --save_dir new_experiments

# Centrally hosted
CUDA_VISIBLE_DEVICES=0 python src/run_TCT.py --dataset fitzpatrick --architecture efficientnet-b0 --central --rounds_stage1 200 --rounds_stage2 0 --local_epochs_stage1 1 --local_lr_stage1 0.01 --momentum 0.9 --batch_size 64 --use_data_augmentation --save_dir new_experiments

# TCT (IID)
CUDA_VISIBLE_DEVICES=0 python src/run_TCT.py --dataset fitzpatrick --architecture efficientnet-b0 --rounds_stage1 100 --rounds_stage2 100 --local_epochs_stage1 5 --local_lr_stage1 0.01 --local_steps_stage2 500 --local_lr_stage2 0.00001 --momentum 0.9 --batch_size 64 --use_data_augmentation --use_iid_partition --save_dir new_experiments

# FedAvg (IID)
CUDA_VISIBLE_DEVICES=0 python src/run_TCT.py --dataset fitzpatrick --architecture efficientnet-b0 --rounds_stage1 200 --rounds_stage2 0 --local_epochs_stage1 5 --local_lr_stage1 0.01 --momentum 0.9 --batch_size 64 --use_data_augmentation --use_iid_partition --save_dir new_experiments
