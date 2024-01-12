import torch
import numpy as np

tensor_a = torch.tensor([1,2,3,4])
tensor_b = torch.tensor([5,6,7,8])

#Add
print("Add")
print(tensor_a + tensor_b)

#Addition longhand
print("Longhand Add")
print(torch.add(tensor_a, tensor_b))

#Subtraction
print("Subtraction")
print(tensor_a - tensor_b)

#Multiplication
print("Multiplication")
print(tensor_a * tensor_b)

#Division
print("Division")
print(tensor_a / tensor_b)

#Exp
print(torch.pow(tensor_a, tensor_b))