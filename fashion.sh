#! /usr/bin/bash

# TCT
python src/run_TCT.py --dataset fashion --architecture cnn --rounds_stage1 50 --rounds_stage2 50 --local_epochs_stage1 5  --local_lr_stage1 0.01 --samples_per_client 10000 --local_steps_stage2 300 --local_lr_stage2 0.00001

# FedAvg
python src/run_TCT.py --dataset fashion --architecture cnn --rounds_stage1 100 --rounds_stage2 0 --local_epochs_stage1 5  --local_lr_stage1 0.01 --samples_per_client 10000

# Centrally hosted
python src/run_TCT.py --dataset fashion --architecture cnn --central --rounds_stage1 100 --rounds_stage2 0 --local_epochs_stage1 1  --local_lr_stage1 0.01 --samples_per_client 50000
