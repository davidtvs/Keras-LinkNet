# Keras-LinkNet

Keras implementation of [*LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation*](https://arxiv.org/abs/1707.03718), ported from the lua-torch ([LinkNet](https://github.com/e-lab/LinkNet)) and PyTorch ([pytorch-linknet](https://github.com/e-lab/pytorch-linknet)) implementation, both created by the authors.


## Installation

1. Python 3 and pip.
2. Set up a virtual environment (optional, but recommended).
3. Install dependencies using pip: ``pip install -r requirements.txt``.


## Usage

Run [``main.py``](https://github.com/davidtvs/Keras-LinkNet/blob/master/main.py), the main script file used for training and/or testing the model. The following options are supported:

```
python main.py [-h] [--mode {train,test,full}] [--resume RESUME]
               [--initial_epoch INITIAL_EPOCH]
               [--pretrained_encoder PRETRAINED_ENCODER]
               [--weights_path WEIGHTS_PATH] [--batch_size BATCH_SIZE]
               [--epochs EPOCHS] [--learning_rate LEARNING_RATE]
               [--lr_decay LR_DECAY] [--lr_decay_epochs LR_DECAY_EPOCHS]
               [--dataset {camvid,cityscapes}] [--dataset_dir DATASET_DIR]
               [--weighing {enet,mfb,none}]
               [--ignore_unlabeled IGNORE_UNLABELLED] [--workers WORKERS]
               [--verbose {0,1,2}] [--name NAME]
               [--checkpoint_dir CHECKPOINT_DIR]
```

For help on the optional arguments run: ``python main.py -h``


### Examples: Training

```
python main.py -m train --checkpoint_dir save/folder/ --name model_name --dataset name --dataset_dir path/root_directory/
```


### Examples: Resuming training

```
python main.py -m train --resume True --initial_epoch 10 --checkpoint_dir save/folder/ --name model_name --dataset name --dataset_dir path/root_directory/
```


### Examples: Testing

```
python main.py -m test --checkpoint_dir save/folder/ --name model_name --dataset name --dataset_dir path/root_directory/
```


## Project structure

### Folders

- [``data``](https://github.com/davidtvs/Keras-LinkNet/tree/master/data): Contains code to load the supported datasets.
- [``metrics``](https://github.com/davidtvs/Keras-LinkNet/tree/master/metric): Evaluation-related metrics.
- [``models``](https://github.com/davidtvs/Keras-LinkNet/tree/master/models): LinkNet model definition.
- [``checkpoints``](https://github.com/davidtvs/Keras-LinkNet/tree/master/checkpoints): By default, ``main.py`` will save models in this folder. The pre-trained encoder (ResNet18) trained on ImageNet can be found here.

### Files

- [``args.py``](https://github.com/davidtvs/Keras-LinkNet/blob/master/arg.py): Contains all command-line options.
- [``main.py``](https://github.com/davidtvs/Keras-LinkNet/blob/master/main.py): Main script file used for training and/or testing the model.
- [``callbacks.py``](https://github.com/davidtvs/Keras-LinkNet/blob/master/callbacks.py): Custom callbacks are defined here.
