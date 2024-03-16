# Federated Conformal Predictors


## Abstract
Conformal prediction is emerging as a popular paradigm for providing rigorous uncertainty quantification in machine learning since it can be easily applied as a post-processing step to already trained models.
In this paper, we extend conformal prediction to the federated learning setting. The main challenge we face is data heterogeneity across the clients --- this violates the fundamental tenet of *exchangeability* required for conformal prediction. We instead propose a weaker notion of *partial exchangeability* which is better suited to the FL setting and use it to develop the Federated Conformal Prediction (FCP) framework. We show FCP enjoys rigorous theoretical guarantees as well as excellent empirical performance on several computer vision and medical imaging datasets.
Our results demonstrate a practical approach to incorporating meaningful uncertainty quantification in distributed and heterogeneous environments.

[Paper](https://arxiv.org/abs/2305.17564)

## Datasets
* [Fitzpatrick17K](https://github.com/mattgroh/fitzpatrick17k)
* [MedMNIST](https://medmnist.com/)

## Training FCP models
`python src/run_TCT.py --dataset cifar100 --architecture small_resnet14 --rounds_stage1 100 --rounds_stage2 100 --local_epochs_stage1 5 --local_lr_stage1 0.01 --samples_per_client 10000 --local_steps_stage2 500 --local_lr_stage2 0.00001 --momentum 0.9 --use_data_augmentation --save_dir new_experiments`

Optionally train FedAvg and centralized baselines
`python src/run_TCT.py --dataset cifar100 --architecture small_resnet14 --rounds_stage1 200 --rounds_stage2 0 --local_epochs_stage1 5 --local_lr_stage1 0.01 --samples_per_client 10000 --momentum 0.9 --use_data_augmentation --save_dir new_experiments`

`python src/run_TCT.py --dataset cifar100 --architecture small_resnet14 --central --rounds_stage1 200 --rounds_stage2 0 --local_epochs_stage1 1 --local_lr_stage1 0.01 --samples_per_client 50000 --momentum 0.9 --use_data_augmentation --save_dir new_experiments`



## Citation
```
@article{lu2023federated,
  title={Federated Conformal Predictors for Distributed Uncertainty Quantification},
  author={Lu, Charles and Yu, Yaodong and Karimireddy, Sai Praneeth and Jordan, Michael I and Raskar, Ramesh},
  journal={arXiv preprint arXiv:2305.17564},
  year={2023}
}
```
