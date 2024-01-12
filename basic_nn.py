import torch
import torch.nn as nn #NN library from torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

#Create model class which inherits nn module
class Model(nn.Module):
  #Input layer (4 inputs)
  #Hidden layers (1 and 2)
  #Output layer (1 output)
  def __init__(self, in_features = 4, h1=8, h2=9, h3=9, out_features=3):
    super().__init__() #Instantiates nn module
    self.fc1 = nn.Linear(in_features, h1) #Fully connected from input to layer 1
    self.fc2 = nn.Linear(h1, h2) # etc
    self.fc3 = nn.Linear(h2,h3)
    self.out = nn.Linear(h3, out_features)
  
  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    x = self.out(x)

    return x

torch.manual_seed(12)

#Create instance

model = Model()

# Download iris dataset
url = 'https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv'

my_df = pd.read_csv(url) #Read the data in pandas

# Change type variety of iris to a number instead of string
my_df['variety'] = my_df['variety'].replace('Setosa', 0.0)
my_df['variety'] = my_df['variety'].replace('Versicolor', 1.0)
my_df['variety'] = my_df['variety'].replace('Virginica', 2.0)

print(my_df)

#Train test split
X = my_df.drop('variety', axis=1)
y = my_df['variety']

#Convert to NP array
X = X.values
y = y.values

# Train test split (trains 80%, tests 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

#Convert to tensor
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

#Convert y to tensors long (long -> 64bit int)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

#Set criteria of model to measure error
criterion = nn.CrossEntropyLoss()
#Implement adam optimizer, learning rate (lr): if error doesn't go down after 
#a lot of iterations, we should lower lr
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

#Train model
#Determine number of epochs
epochs = 150
losses = []

for i in range(epochs):
  # Go forward and get a prediction
  y_pred = model.forward(X_train) #Get predicted results

  # Measure the loss/error, will be high early on
  loss = criterion(y_pred, y_train) #Predicted values vs trained values

  #Keep track of losses
  losses.append(loss.detach().numpy())

  #Print every 10 epochs
  if i % 10 == 0:
    print(f'Epoch {i} and loss: {loss}')

  # Do back propagation: take error rate and feed it back 
  # Thru network to fine tune weights, helps it learn better
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

# Graph it out
plt.plot(range(epochs), losses)
plt.ylabel('loss/error')
plt.xlabel('Epoch')
plt.show()

#Evaluate Model on Test Data Set
with torch.no_grad():
  y_eval = model.forward(X_test) #X_test are features from test set, y_eval will be predictions
  loss = criterion(y_eval, y_test) # Find loss
  print(loss)

correct = 0
with torch.no_grad():
  for i, data in enumerate(X_test):
    y_val = model.forward(data)

    if y_test[i] == 0:
      x = "Setosa"
    elif y_test[i] == 1:
      x = 'Versicolor'
    else:
      x = 'Virginica'

    #This will tell us what type of flower class our network thinks it is
    print(f'{i+1}.) {str(y_val)} \t {x} {y_test[i]}')

    #Correct or not?
    if y_val.argmax().item() == y_test[i]:
      correct += 1


print(f'We got {correct} correct out of 30!')


new_iris = torch.tensor([5.9, 3.0, 5.1, 1.8])


with torch.no_grad():
  y_val = model.forward(new_iris)
  print(y_val.argmax().item())


# Save model

torch.save(model.state_dict(), 'iris_model_1.pt')

# Load model:

new_model = Model()
new_model.load_state_dict(torch.load('iris_model_1.pt'))
print(new_model.eval())