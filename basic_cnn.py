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

import time

# Convert MNIST into tensors (# images, height, width, color channel)
transform = transforms.ToTensor()


# Train Data

train_data = datasets.MNIST(root='/cnn_data', train=True, download=True, transform=transform) #Train data, download to files, and transform (not in order)

# Test Data
test_data = datasets.MNIST(root='/cnn_data', train=False, download=True, transform=transform) #Test data and transform (not in order)

print(train_data)

print(test_data)

# Create a batch for images
train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
test_loader = DataLoader(test_data, batch_size=10, shuffle=False)

# Define CNN model
# 2 Convolutional Layers
conv1 = NN.Conv2d(1, 6, 3, 1) # 1 input, 6 outputs, 3x3 kernel, 1 at a time strides
conv2 = NN.Conv2d(6, 16, 3, 1) # 6 inputs (from the last layer), 16 outputs, 3x3 kernel, 1 at a time strides

# Grab 1 MNIST record
for i, (X_train, y_train) in enumerate(train_data):
  break

print(X_train.shape) # Shows [1, 28, 28] meaning 1 image that is 28x28 size

x = X_train.view(1,1,28,28)

# Perform first convolution

x = F.relu(conv1(x)) # Rectified linear unit for activation function

print(x.shape) # Shows [1,6,26,26] meaning 1 image, 6 filters, and 26x26 size.
# It is 26 x 26 size because it removes the empty data on the edges (extra 0's that are redundant to the info)

# Pass through pooling layer:
x = F.max_pool2d(x, 2, 2) # Kernel size of 2, stride size of 2

print(x.shape) # Shows [1,6,13,13] = 1 image, 6 filters, 13x13 size
# Is 13x13 because 26 / 2 kernel size is 13

# Second convolutional layer
x = F.relu(conv2(x))

print(x.shape) # Shows [1, 16, 11, 11] = 1 image, 16 outputs (as defined above), 11 x 11 size because we lose redundant data on edges

# Pooling layer 2
x = F.max_pool2d(x, 2, 2)

print(x.shape) # Shows [1, 6, 5, 5] because 1 image, 6 filters, 5 x 5 size because 11 // 2 is 5 and we can't round to 6 because false data cannot be manufactured.

class ConvolutionalNetwork(NN.Module):

  def __init__(self):
    super().__init__()
    self.conv1 = NN.Conv2d(1, 6, 3, 1) # Convolutional layer 1
    self.conv2 = NN.Conv2d(6, 16, 3, 1) # Convolutional layer 2
    
    # Fully connected layers (FC layers)
    self.fc1 = NN.Linear(5*5*16, 120) #120 Neurons, 5*5*16 because of output from pooling
    self.fc2 = NN.Linear(120, 84)
    self.fc3 = NN.Linear(84, 10)


  # Forward function
  def forward(self, X):
    X = F.relu(self.conv1(X))
    X = F.max_pool2d(X, 2, 2) # 2x2 Kernel, stride of 2

    # Second pass

    X = F.relu(self.conv2(X))
    X = F.max_pool2d(X, 2, 2)

    # Re-View data to flatten
    X = X.view(-1, 16*5*5) # negative one to make batch size variable

    # FC layers

    X = F.relu(self.fc1(X))
    X = F.relu(self.fc2(X))
    X = self.fc3(X) # No relu on last one because it is the output

    return F.log_softmax(X, dim=1)


# Create Instance of model
  
torch.manual_seed(12)

model = ConvolutionalNetwork()


# Create loss function and optimizer

criterion = NN.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # Smaller the lr, longer to train but better results

start_time = time.time()

# Create variables to track
epochs = 5
train_losses = []
test_losses = []
train_correct = []
test_correct = []

# For loop of epochs
for i in range(epochs):
  trn_correct = 0
  tst_correct = 0

  # Train
  for b, (X_train, y_train) in enumerate(train_loader):
    b += 1
    y_pred = model(X_train) # Get predicted values from training set
    loss = criterion(y_pred, y_train) # Check the loss
    
    predicted = torch.max(y_pred.data, 1)[1] # Add up the number of correct predictions (Indexed off first point)
    batch_correct = (predicted == y_train).sum() # Want to know how many we got correct from this batch
    trn_correct += batch_correct # Keep track of train correct

    # Update params
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print results
    if b%600 == 0:
      print(f'Loss in epoch {i+1} and batch {b} is {loss.item()}')

  train_losses.append(loss)
  train_correct.append(trn_correct)


  # Test
  with torch.no_grad():
    for b, (X_test, y_test) in enumerate(test_loader):
      b += 1
      y_val = model(X_test)
      predicted = torch.max(y_val.data, 1)[1]
      tst_correct += (predicted == y_test).sum()

  loss = criterion(y_val, y_test)
  test_losses.append(loss)
  test_correct.append(tst_correct)

current_time = time.time()

total = current_time - start_time

print(f'Training Took: {int(total/60)} minutes {int(total%60)} seconds')

# Convert to numpy
train_losses = [tl.item() for tl in train_losses]

# Graph loss over epochs
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Test Loss')
plt.title('Loss at Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

#Graph accuracy
plt.plot([t/600 for t in train_correct], label='Training Accuracy')
plt.plot([t/100 for t in test_correct], label='Validation Accuracy')
plt.title('Accuracy at the end of each Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

test_load_everything = DataLoader(test_data, batch_size=10000, shuffle=False)

with torch.no_grad():
  correct = 0
  for X_test, y_test in test_load_everything:
    y_val = model(X_test)
    predicted = torch.max(y_val, 1)[1]
    correct += (predicted == y_test).sum()
  
print(correct.item()/len(test_data)*100, '% correct!')