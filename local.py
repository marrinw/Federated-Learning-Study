import torch
from torchvision import datasets,transforms
train_set=datasets.MNIST("data",train=True,download=True)