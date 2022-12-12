#! /usr/bin/bash

#  TCT
#python src/run_TCT.py --dataset mnist
#python src/run_TCT.py --dataset fashion
#python src/run_TCT.py --dataset cifar --local_steps_stage2 100 --num_samples_per_client 1000 --resnet --local_lr_stage1 0.001

#  FedAvg
#python src/run_TCT.py --rounds_stage1 200 --rounds_stage2 0 --dataset mnist
#python src/run_TCT.py --rounds_stage1 200 --rounds_stage2 0 --dataset fashion
#python src/run_TCT.py --rounds_stage1 200 --rounds_stage2 0 --dataset cifar #--resnet --num_samples_per_client 1000 --local_lr_stage1 0.001

#  Centrally hosted
#python src/run_TCT.py --rounds_stage1 100 --rounds_stage2 0 --num_client 1 --dataset mnist
#python src/run_TCT.py --rounds_stage1 100 --rounds_stage2 0 --num_client 1 --dataset fashion
#python src/run_TCT.py --rounds_stage1 100 --rounds_stage2 0 --num_client 1 --dataset cifar #--resnet --num_samples_per_client 1000 --local_lr_stage1 0.001

python src/run_TCT.py --dataset fashion --rounds_stage1 100 --rounds_stage2 100 --num_samples_per_client 5000 --local_steps_stage2 500 --local_lr_stage2 0.0001
python src/run_TCT.py --dataset fashion --rounds_stage1 200 --rounds_stage2 0 --num_samples_per_client 5000
