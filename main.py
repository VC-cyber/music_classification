from neuralnet import NeuralNetwork
import torch
from torch import nn
from torch import optim
from preprocessing import preprocessing
from preprocessing import preprocessingCV
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

x_train, y_train, x_cv, y_cv, x_test, y_test  = preprocessingCV()



model = NeuralNetwork()
loss_fn = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.7)
optimizer = optim.Adam(model.parameters())
lossArr = [] 
accuracy = []
def trainloop(model, loss, optimizer, x_input, y_input):
    for epoch in range(1000): 
        X = torch.tensor(x_input).float()
        y = torch.tensor(y_input)-1
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        lossArr.append(loss)
        loss.backward()
        optimizer.step()

for i in range(len(x_train)):
    lossArr = []
    trainloop(model, loss_fn, optimizer, x_train[0:i], y_train[0:i])
    pred2 = model(torch.tensor(x_cv).float())
    accuracy.append(torch.argmax(pred2, dim=1).numpy()==y_cv).astype(int).sum() / len(y_cv)
    
    
plt.plot(accuracy)
plt.show()
#pred2 = model(torch.tensor(x_test).float())

#accuracy = (torch.argmax(pred2, dim=1).numpy()==y_test).astype(int).sum() / len(y_test)
#print(accuracy)

