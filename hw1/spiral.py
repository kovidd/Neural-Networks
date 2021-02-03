# spiral.py
# COMP9444, CSE, UNSW
# z5240067, Kovid Sharma

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

'''
    First, the input (x,y) is converted to polar co-ordinates (r,a) 
    with r=sqrt(x*x + y*y), a=atan2(y,x). Next, (r,a) is fed into a 
    fully connected neural network with one hidden layer using tanh 
    activation, followed by a single output using sigmoid activation. 
    The conversion to polar coordinates should be included in your 
    forward() method, so that the Module performs the entire task of 
    conversion followed by network layers.
'''    
class PolarNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(PolarNet, self).__init__()
        # INSERT CODE HERE
        self.l1 = nn.Linear(in_features=2, out_features=num_hid, bias=True)
        self.l2 = nn.Linear(in_features=num_hid, out_features=1, bias=True)

    def forward(self, input):
        # output = 0*input[:,0] # CHANGE CODE HERE
        x = input[:, 0]
        y = input[:, 1]
        temp = x * x + y * y
        r = torch.sqrt(temp).reshape(-1, 1)
        a = torch.atan2(y, x).reshape(-1, 1)
        pos = torch.cat(tensors=(r, a), dim=1)
        self.hid1 = torch.tanh(self.l1(pos))
        output = torch.sigmoid(self.l2(self.hid1))
        return output


'''
    RawNet which operates on the raw input (x,y) without converting 
    to polar coordinates. Your network should consist of two fully 
    connected hidden layers with tanh activation, plus the output layer, 
    with sigmoid activation. The two hidden layers should each have the
    same number of hidden nodes, determined by the parameter num_hid.
'''
class RawNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(RawNet, self).__init__()
        # INSERT CODE HERE
        self.l1 = nn.Linear(in_features=2, out_features=num_hid, bias=True)
        self.l2 = nn.Linear(in_features=num_hid, out_features=num_hid, bias=True)
        self.l3 = nn.Linear(in_features=num_hid, out_features=1, bias=True)

    def forward(self, input):
        # output = 0*input[:,0] # CHANGE CODE HERE
        self.hid1 = torch.tanh(self.l1(input))
        self.hid2 = torch.tanh(self.l2(self.hid1))
        output = torch.sigmoid(self.l3(self.hid2))
        return output

'''
    Using graph_output() as a guide, write a method called 
    graph_hidden(net, layer, node) which plots the activation 
    (after applying the tanh function) of the hidden node with the
    specified number (node) in the specified layer (1 or 2). 
    (Note: if net is of type PolarNet, graph_output() only needs 
     to behave correctly when layer is 1).
    
    Hint: you might need to modify forward() so that the hidden unit
    activations are retained, i.e. replace hid1 = torch.tanh(...) 
    with self.hid1 = torch.tanh(...)

    Use this code to generate plots of all the hidden nodes in PolarNet, 
    and all the hidden nodes in both layers of RawNet, and include 
    them in your report.
'''
def graph_hidden(net, layer, node):
    # plot function computed by model
    plt.clf()
    # INSERT CODE HERE
    xrange = torch.arange(start=-7, end=7.1, step=0.01, dtype=torch.float32)
    yrange = torch.arange(start=-6.6, end=6.7, step=0.01, dtype=torch.float32)
    xcoord = xrange.repeat(yrange.size()[0])
    ycoord = torch.repeat_interleave(yrange, xrange.size()[0], dim=0)
    grid = torch.cat(tensors=(xcoord.unsqueeze(dim=1), ycoord.unsqueeze(dim=1)), dim=1)
    
    with torch.no_grad():
        net.eval()
        output = net(grid)
        # first hidden layer for both networks, 2nd only for RawNet
        if layer == 1:
            pred = (net.hid1[:, node] >= 0).float() #hidden nodes is tanh() rather than sigmoid(
        # second hidden layer, only RawNet
        else:
            pred = (net.hid2[:, node] >= 0).float() #hidden nodes is tanh() rather than sigmoid(
            
        plt.pcolormesh(xrange, yrange, pred.cpu().view(yrange.size()[0], xrange.size()[0]), cmap='Wistia')
        