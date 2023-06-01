#! /bin/bash

# Slurm sbatch options
#SBATCH -o new_outputs/cifar10.%j
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:volta:1

# Loading the required module
source /etc/profile
module load anaconda cuda


python src/run_TCT.py --dataset cifar10-2 --architecture small_resnet14 --rounds_stage1 200 --rounds_stage2 0 --local_epochs_stage1 5 --local_lr_stage1 0.01 --samples_per_client 10000 --momentum 0.9 --use_data_augmentation  --save_dir new_experiments --data_dir ../data

python src/run_TCT.py --dataset cifar10-2 --architecture small_resnet14 --rounds_stage1 100 --rounds_stage2 100 --local_epochs_stage1 5 --local_lr_stage1 0.01 --samples_per_client 10000 --local_steps_stage2 500 --local_lr_stage2 0.0001 --momentum 0.9 --use_data_augmentation --save_dir new_experiments --data_dir ../data --override

python src/run_TCT.py --dataset cifar10-3 --architecture small_resnet14 --rounds_stage1 100 --rounds_stage2 100 --local_epochs_stage1 5 --local_lr_stage1 0.01 --samples_per_client 10000 --local_steps_stage2 500 --local_lr_stage2 0.0001 --momentum 0.9 --use_data_augmentation --save_dir new_experiments --data_dir ../data --override

python src/run_TCT.py --dataset cifar10 --architecture small_resnet14 --rounds_stage1 100 --rounds_stage2 100 --local_epochs_stage1 5 --local_lr_stage1 0.01 --samples_per_client 10000 --local_steps_stage2 500 --local_lr_stage2 0.0001 --momentum 0.9 --use_data_augmentation --use_iid_partition --save_dir new_experiments --data_dir ../data --override

