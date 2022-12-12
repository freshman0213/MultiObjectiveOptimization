# How Does Parameter Sharing Affect Multi-task Learning?

This code repository includes the source code for the CS330 Final Project Fall 2022 by


Li-Heng Lin|
Tz-Wei Mo|
Annie Ho|

This repo includes modification to the Multi-LeNet network to include FiLM layers. Multi-VGG is also added to further investigate the effects of FiLM layers by adding task specific batch normalization. CIFAR-10/SVHN dataset is also combined for a new experiment setup for Multi-VGG. 

Code for adding FiLM layers in ResNet is also included.


# Requirements and References
The code uses the following Python packages and they are required: ``tensorboardX, pytorch, click, numpy, torchvision, tqdm, scipy, Pillow, imageio``


We adapt and use some code snippets from:
* https://github.com/isl-org/MultiObjectiveOptimization [Scalarization Multi-task Training]
* https://github.com/kkweon/mnist-competition/blob/master/vgg5.py?fbclid=IwAR1LeFSiJ7ziHQyzDkaHLKVJmDKJw_Z_G4xLJ6hAsaB3PkjqbH0NIqZ52Ao  [VGG]



# Usage
The code base uses `configs.json` for the global configurations like dataset directories, etc.. Experiment specific parameters are provided seperately as a json file. See the `sample.json` for an example.

To train a model, use the command: 
```bash
python multi_task/train_multi_task_scalarization.py --param_file=./sample.json
```


