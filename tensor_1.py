import torch
import numpy as np 

#Regular array
my_list = [[1,2,3,4,5], [6,7,8,9,10]]
print(my_list)

#Numpy array
np1 = np.random.rand(3,4)
print(np1)

#2d tensor
tensor_2d = torch.randn(3,4)
print(tensor_2d)

#3d tensor
tensor_3d = torch.zeros(2,3,4)
print(tensor_3d)

#Create tensor from np array
my_tensor = torch.tensor(np1)
print(my_tensor)