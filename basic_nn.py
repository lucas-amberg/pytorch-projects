import torch
import torch.nn as nn #NN library from torch
import torch.nn.functional as F

#Create model class which inherits nn module
class Model(nn.Module):
  #Input layer (4 inputs)
  #Hidden layers (1 and 2)
  #Output layer (1 output)
  def __init__(self, in_features = 4, h1=8, h2=9, out_features=3):
    super().__init__() #Instantiates nn module
    self.fc1 = nn.Linear(in_features, h1) #Fully connected from input to layer 1
    self.fc2 = nn.Linear(h1, h2) # etc
    self.out = nn.Linear(h2, out_features)
  
  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.out(x)

    return x

torch.manual_seed(12)

#Create instance

model = Model()