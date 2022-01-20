import torch 
from torch import nn
import matplotlib as plt
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork,self).__init__()
        self.input = nn.Linear(58,256)
        self.hidden_layer1 = nn.Linear(256,256)
        self.dropout1 = nn.Dropout(p=0.1)
        self.hidden_layer2 = nn.Linear(256,256)
        self.output = nn.Linear(256, 10)

    def forward(self,x):
        x = self.input(x)
        x = self.hidden_layer1(x)
        x = F.sigmoid(x)
        x = self.dropout1(x)
        x = self.hidden_layer2(x)
        x = F.sigmoid(x)
        x = self.output(x)
        x = F.sigmoid(x)

        return x




