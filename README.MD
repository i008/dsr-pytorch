### Download to data directory

https://s3.eu-central-1.amazonaws.com/i008data/fashion-mnist_train.csv  
https://s3.eu-central-1.amazonaws.com/i008data/fashion-mnist_test.csv  
https://s3.eu-central-1.amazonaws.com/i008data/small_sneaker_not_sneaker.tar.gz


### Environment

```bash
conda env update
conda install nb_conda
source activate dl
python -m ipykernel install --user --name dl --display-name "dsr-dl"

If you dont have a GPU in your system remove the cudatoolkit from the environment.yml file

python test_env.py

```

### Commands

```bash
tensorboard --port=8888 --logdir runs
```

## Pytorch and nn-basics

[Pytorch basics](1-pytorch-basics.ipynb)  
[The simplest neural network - perceptron](2-perceptron.ipynb)  
[Dataset and Dataloader](pytorch_dataset_and_dataloader.ipynb)

## Pytorch frameworks and logging
[Perceptron trained with skorch](2.1-perceptron-skorch.ipynb)  
[Perceptron trained and logged with tensorboard](2.2-perceptron-tensorboard.ipynb) 
[Perceptron trained with Catalyst](2.3-perceptron-catalyst.ipynb)

## Python
[Python refresher](3-Python-refresher-jupyter-tips-and-tricks.ipynb)

## Fundamentals

[Convolutions basics](4-convolutions.ipynb)

## FashionMnist CNN training 

[Training FashionMnist](training-fashion-mnist.ipynb)  
[Training FashionMnist with neptune logging](fashion-mnist-neptune.ipynb)  

## Transfer learning
[Training FashionMnist with transfer-learning](6-pytorch-fmnist-transfer-learning.ipynb)

## Deep Learning advanced concepts
[Finding the right learning rate](5.3-learning-rate-finder.ipynb)  
[Batch normalization](batchnom.ipynb)
[Grad CAM](grad-cam.ipynb)


## Project 1 - Sneaker Not Sneaker
[Sneaker not Sneaker transfer learning](7-transfer-learning-sneaker-not-sneaker.ipynb)


## Project 2 - Segmentation
[Segmentation Dataset](8-ISBI-Dataset.ipynb)
[Segmentation Models](9-Segmentation-Networks.ipynb)


## Object Detection
[Detecting Shapes with Retinanet](10-Object-Detection-Shapes.ipynb)


## Representaional Learning
[Auto Encoder](AutoEncoders%20%20-%20CAE.ipynb)
[Variational Auto Encoder](AutoEncoders%20%20-%20VAE.ipynb)


## Tabular Deep Learning
[Tabular Embeddings](tabular-embeddings-1.ipynb)
[Adult Salaries with DL](tabular-embeddings-2)
