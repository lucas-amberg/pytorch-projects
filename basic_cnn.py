import torch
import torch.nn as NN
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Convert MNIST into tensors (# images, height, width, color channel)
transform = transforms.ToTensor()


# Train Data

train_data = datasets.MNIST(root='/cnn_data', train=True, download=True, transform=transform) #Train data, download to files, and transform (not in order)

# Test Data
test_data = datasets.MNIST(root='/cnn_data', train=False, download=True, transform=transform) #Test data and transform (not in order)

print(train_data)

print(test_data)