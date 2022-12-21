#! /usr/bin/bash

# TCT
#python src/run_TCT.py --dataset fashion --rounds_stage1 100 --rounds_stage2 100 --local_epochs_stage1 5  --local_lr_stage1 0.05 --num_samples_per_client 5000 --local_steps_stage2 500 --local_lr_stage2 0.00001

# FedAvg
python src/run_TCT.py --dataset fashion --rounds_stage1 200 --rounds_stage2 0 --local_lr_stage1 0.05 --num_samples_per_client 5000

# Centrally hosted
python src/run_TCT.py --dataset fashion --rounds_stage1 200 --rounds_stage2 0 --num_client 1 --local_lr_stage1 0.05 --num_samples_per_client 5000


python src/run_TCT.py --resnet --dataset cifar --rounds_stage1 100 --rounds_stage2 100 --local_epochs_stage1 5  --local_lr_stage1 0.05 --num_samples_per_client 5000 --local_steps_stage2 500 --local_lr_stage2 0.00001
python src/run_TCT.py --resnet --dataset cifar --rounds_stage1 200 --rounds_stage2 0 --local_lr_stage1 0.05 --num_samples_per_client 5000
python src/run_TCT.py --resnet --dataset cifar --rounds_stage1 200 --rounds_stage2 0 --num_client 1 --local_lr_stage1 0.05 --num_samples_per_client 5000
