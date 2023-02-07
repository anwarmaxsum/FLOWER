## FLOWER:
This directory contain our implementation of our paper entitled ”Few-Shot Continual Learning via Flat-to-Wide Approaches" (FLOWER)
This project is developed based on [F2M](https://github.com/moukamisama/f2m) Project with modification and addition of our methods.

## Code:
Pleasee see "code" directory to see the implementation of the algorithms.

## Logs:
Pleasee see "logs" directory to see the raw results of our experiments.

## System Requirements:
This project is run by using the following  environment:
1. NVIDIA NGC container [21.03-py3](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags):
   The container contains python 3.8, pytorch 1.9.0, torchvision, numpy, and the other libraries. 
   Please read the fllowing link for the complete documentation:
   https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel_21-03.html#rel_21-03
2. [wandb](https://docs.wandb.com/quickstart) (optional)
   You can install wandb by using command "pip install wandb" on top of the container 


## Datasets
We evaluate our system in the benchmark datasets, including ```CUB-200-2011, CIFAR100, miniImageNet```.
Please download [CUB-200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011/), [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html) and [miniImageNet](https://github.com/yaoyao-liu/mini-imagenet-tools).

## Example of Running Script:
Please see file example_running_script.txt in code directory

## Contribution (please see code directory)
The main contribution of our work is as follow:
1. Base task learning (please see *_model.py methods directory):
F2MJ (Flat minima + projection), F2MMAS (Flat minima + MAS weight importance)+ F2MMASJ (FLOWER base task learning), F2MMASJNOPL (FLOWER base task learning without prototype loss), F2MSI Flat minima + SI weight importance).
Those files are developed from F2MModel with necessary modifications.

2. Continual tasks learning (please see *_model.py methods directory):
flower1 (FLOWER continual tasks learning), flower_no_psi (FLOWER without projection), flower_no_fm (FLOWER without flat minima), flower_no_ball (FLOWER without ball augmentation)   flower_no_mas (FLOWER without MAS weight importance), flower_no_pl (FLOWER without prototype loss)

3. Incremental learning procedure without memory: incremental_procedure_nomem.py
The file is developed from F2MModel with necessary modifications.

4. Minor modification:
We modify incremental learning procedure script (incremental_procedure.py) for debugging purpose.

## LICENSE
This FLOWER project is released under the Apache 2.0 license. Please see code/LICENSE directory.
