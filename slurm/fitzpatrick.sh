#! /bin/bash

# Slurm sbatch options
#SBATCH -o ../outputs/fitzpatrick.%j
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:volta:1

# Loading the required module
source /etc/profile
module load anaconda cuda

# TCT
python src/run_TCT.py --dataset fitzpatrick --architecture resnet34 --rounds_stage1 100 --rounds_stage2 100 --local_epochs_stage1 5  --local_lr_stage1 0.01 --local_steps_stage2 500 --local_lr_stage2 0.00001 --momentum 0.9 --use_data_augmentation --save_dir experiments --data_dir ../data

# FedAvg
python src/run_TCT.py --dataset fitzpatrick --architecture resnet34 --rounds_stage1 200 --rounds_stage2 0 --local_epochs_stage1 5  --local_lr_stage1 0.01 --momentum 0.9 --use_data_augmentation --save_dir experiments --data_dir ../data

# Centrally hosted
python src/run_TCT.py --dataset fitzpatrick --architecture resnet34 --rounds_stage1 200 --momentum 0.9 --use_data_augmentation --save_dir experiments --data_dir ../data --central

# TCT (IID)
python src/run_TCT.py --dataset fitzpatrick --architecture resnet34 --rounds_stage1 100 --rounds_stage2 100 --local_epochs_stage1 5  --local_lr_stage1 0.01 --local_steps_stage2 500 --local_lr_stage2 0.00001 --momentum 0.9 --use_data_augmentation --save_dir experiments --data_dir ../data --use_iid_partition

# FedAvg (IID)
python src/run_TCT.py --dataset fitzpatrick --architecture resnet34 --rounds_stage1 200 --rounds_stage2 0 --local_epochs_stage1 5  --local_lr_stage1 0.01 --momentum 0.9 --use_data_augmentation --save_dir experiments --data_dir ../data --use_iid_partition
