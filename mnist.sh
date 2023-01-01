#! /usr/bin/bash

# TCT
CUDA_VISIBLE_DEVICES=0 python src/run_TCT.py --dataset mnist --architecture efficientnet-b0 --rounds_stage1 100 --rounds_stage2 100 --local_epochs_stage1 5 --local_lr_stage1 0.01 --samples_per_client 12000 --local_steps_stage2 500 --local_lr_stage2 0.00001 --momentum 0.9 --use_data_augmentation --save_dir new_experiments --num_test_samples 10000 --start_from_stage2

# FedAvg
CUDA_VISIBLE_DEVICES=0 python src/run_TCT.py --dataset mnist --architecture efficientnet-b0 --rounds_stage1 200 --rounds_stage2 0 --local_lr_stage1 0.01 --samples_per_client 12000 --momentum 0.9 --use_data_augmentation --save_dir new_experiments --num_test_samples 10000

# Centrally hosted
CUDA_VISIBLE_DEVICES=0 python src/run_TCT.py --dataset mnist --architecture efficientnet-b0 --rounds_stage1 200 --rounds_stage2 0 --local_lr_stage1 0.01 --samples_per_client 12000 --momentum 0.9 --use_data_augmentation --save_dir new_experiments --num_test_samples 10000 --central

# TCT (IID)
CUDA_VISIBLE_DEVICES=0 python src/run_TCT.py --dataset mnist --architecture efficientnet-b0 --rounds_stage1 100 --rounds_stage2 100 --local_epochs_stage1 5 --local_lr_stage1 0.01 --samples_per_client 12000 --local_steps_stage2 500 --local_lr_stage2 0.00001 --momentum 0.9 --use_data_augmentation --save_dir new_experiments --num_test_samples 10000 --use_iid_partition

# FedAvg (IID)
CUDA_VISIBLE_DEVICES=0 python src/run_TCT.py --dataset mnist --architecture efficientnet-b0 --rounds_stage1 200 --rounds_stage2 0 --local_lr_stage1 0.01 --samples_per_client 12000 --momentum 0.9 --use_data_augmentation --save_dir new_experiments --num_test_samples 10000 --use_iid_partition

