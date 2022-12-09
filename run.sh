#! /usr/bin/bash

#  TCT
python src/run_TCT.py --dataset mnist
python src/run_TCT.py --dataset fashion
python src/run_TCT.py --dataset cifar

#  FedAvg
python src/run_TCT.py --rounds_stage1 200 --rounds_stage2 0 --dataset mnist
python src/run_TCT.py --rounds_stage1 200 --rounds_stage2 0 --dataset fashion
python src/run_TCT.py --rounds_stage1 200 --rounds_stage2 0 --dataset cifar

#  Centrally hosted
python src/run_TCT.py --rounds_stage1 200 --rounds_stage2 0 --num_client 1 --dataset mnist
python src/run_TCT.py --rounds_stage1 200 --rounds_stage2 0 --num_client 1 --dataset fashion
python src/run_TCT.py --rounds_stage1 200 --rounds_stage2 0 --num_client 1 --dataset cifar
