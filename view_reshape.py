import torch
my_torch = torch.arange(10)

print(my_torch)

my_torch = my_torch.reshape(2,5)
print(my_torch)

#Reshape if we don't know number of items
my_torch2 = torch.arange(10)
print(my_torch2)

my_torch2 = my_torch2.reshape(2, -1)
print(my_torch)

my_torch3 = torch.arange(10)
print(my_torch3)

my_torch4 = my_torch3.view(2, 5)
print(my_torch4)

my_torch5 = torch.arange(10)
print(my_torch5)

my_torch6 = my_torch5.reshape(2,5)
print(my_torch6)

my_torch5[1] = 2265

print(my_torch6) # Reference to my_torch5

#View is guaranteed a view/reference to a tensor, reshape can be a view or a copy

# Slices
my_torch7 = torch.arange(10)
print(my_torch7)
print(my_torch7[7])

#Grab slice
my_torch8 = my_torch7.reshape(5,2)
print(my_torch8)

print(my_torch8[:,1])
#Shape as column
print(my_torch8[:,1:])
