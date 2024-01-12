import basic_nn as b_nn
import torch 

new_model = b_nn.Model()

new_model.load_state_dict(torch.load('iris_model_1.pt'))

print(new_model.eval())