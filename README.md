# Federated Conformal Predictors


## Abstract
Conformal prediction is emerging as a popular paradigm for providing rigorous uncertainty quantification in machine learning since it can be easily applied as a post-processing step to already trained models.
In this paper, we extend conformal prediction to the federated learning setting. 
The main challenge we face is data heterogeneity across the clients --- this violates the fundamental tenet of \emph{exchangeability} required for conformal prediction. 
We instead propose a weaker notion of \emph{partial exchangeability} which is better suited to the FL setting and use it to develop the Federated Conformal Prediction (FCP) framework. 
We show FCP enjoys rigorous theoretical guarantees as well as excellent empirical performance on several computer vision and medical imaging datasets.
Our results demonstrate a practical approach to incorporating meaningful uncertainty quantification in distributed and heterogeneous environments.



```
@article{lu2023federated,
  title={Federated Conformal Predictors for Distributed Uncertainty Quantification},
  author={Lu, Charles and Yu, Yaodong and Karimireddy, Sai Praneeth and Jordan, Michael I and Raskar, Ramesh},
  journal={arXiv preprint arXiv:2305.17564},
  year={2023}
}
```
