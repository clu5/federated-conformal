# Example on MNIST

### TCT training on MNIST (5 clients)
```python
python run_TCT.py
```
In this example, we consider every client has images of two categories chosen from [0, 1], [2, 3], [4, 5], [6, 7], or [8, 9].

#### Arguments:
* ```num_client```: number of clients
* ```seed```: random seed
* ```num_samples_per_client```: number of training samples on each client
* ```rounds_stage1```: (TCT-stage1) number of training rounds
* ```local_epochs_stage1```: (TCT-stage1) number of training epochs on each client per round
* ```mini_batchsize_stage1```: (TCT-stage1) mini-batch size
* ```local_lr_stage1```: (TCT-stage1) local learning rate
* ```rounds_stage2```: (TCT-stage2) number of training rounds
* ```local_steps_stage2```: (TCT-stage2) number of local steps
* ```local_lr_stage2```: (TCT-stage2) local learning rate
The trade-off regularization parameter ```beta``` can be set in ```[1, 10]```. Larger ```beta``` leads to more robust and less accurate models.
