# Transformer on CIFAR-10
This is a experimental comparison among resnet, vision transformer and visual transformer on the cifar-10 dataset.

## Dataset
When excuting `run.py`， data will be download from [CIFAR_dataset](http://www.cs.toronto.edu/~kriz/cifar.html) automatically to the root directory. 

## Project Structure

### models

The folder `models` defines 3 kinds of different networks - a modified ResNet for cifar, a Vision Transformer (named as ViT) and a Visual Transformer (named as VisT or T-ViT).

### runs

The folder `runs` stores the logs for each training process (tensorboard events).

### weights

The folder `weights` stores the best weights for each model.

### run

For model training, you just need to run `run.py` by 
```
python run.py --model ResNet --epochs 100 --name ResNet_cifar
```

You can do this for more detailed helps:
```
python run.py -h
```

## Reference
