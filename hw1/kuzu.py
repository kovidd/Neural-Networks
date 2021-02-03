# kuzu.py
# COMP9444, CSE, UNSW
# z5240067, Kovid Sharma

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class NetLin(nn.Module):
    # linear function followed by log_softmax
    def __init__(self):
        super(NetLin, self).__init__()
        self.full_connected = nn.Linear(784, 10) #in_features=28*28, out_features=10

    def forward(self, input):
        flat_input = input.view(input.shape[0], -1)  # make sure inputs are flattened
        # print('flat_input.size : ',flat_input.size())
        output = F.log_softmax(self.full_connected(flat_input), dim=1)  # preserve batch dim
        return output # CHANGE CODE HERE

class NetFull(nn.Module):
    # two fully connected tanh layers followed by log softmax
    def __init__(self):
        super(NetFull, self).__init__()
        # INSERT CODE HERE
        hidden_layers = 150
        # print('Number of Hidden Layers:', hidden_layers)
        self.layer_1 = nn.Linear(784, hidden_layers, bias=True) #in_features=28*28, out_features=200
        self.layer_2 = nn.Linear(hidden_layers, 10, bias=True) #in_features=200, out_features=10
        
    def forward(self, input):
        flat_input = input.view(input.shape[0], -1)
        hidden = torch.tanh(self.layer_1(flat_input))
        output = F.log_softmax(self.layer_2(hidden), dim=1)  # preserve batch dim
        return output # CHANGE CODE HERE

class NetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self):
        super(NetConv, self).__init__()
        # INSERT CODE HERE
        # References :
        # https://towardsdatascience.com/convolutional-neural-network-for-image-classification-with-implementation-on-python-using-pytorch-7b88342c9ca9
        # https://stackoverflow.com/questions/56660546/how-to-select-parameters-for-nn-linear-layer-while-training-a-cnn
        
        # --- Test 2 --- #
        c1 = 32
        c2 = c1 * 2
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=c1, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=5, padding=2)
        f1 = c2 * 4 * 4 # n_features_conv * height * width
        # f2 = int(f1/3)
        f2 = 512
        self.layer_1 = nn.Linear(in_features=f1, out_features=f2, bias=True)
        self.layer_2 = nn.Linear(in_features=f2, out_features=10, bias=True)
        self.maxpool = nn.MaxPool2d(kernel_size=5, stride=2)
        # print('Conv layers: 1', c1, c2)
        # print('Hidden layers:', f1, f2, '10')

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = x.view(x.shape[0], -1)
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_2(x)
        
        output = F.log_softmax(x, dim=1)  # preserve batch dim
  
        return output # CHANGE CODE HERE
