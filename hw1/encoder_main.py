# encoder_main.py
# COMP9444, CSE, UNSW

from __future__ import print_function
import torch
import torch.utils.data
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import argparse

from encoder_model import EncModel, plot_hidden
from encoder import star16, heart18, target1, target2

import time

start = time.time()

# command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--target',type=str,default='input',help='input, star16, heart18, target1 or target2')
parser.add_argument('--dim',type=int,default=9,help='input dimension')
parser.add_argument('--plot',default=False,action='store_true',help='show intermediate plots')
parser.add_argument('--epochs',type=int, default=1000000, help='max epochs')
parser.add_argument('--stop',type=float, default=0.02, help='loss to stop at')
parser.add_argument('--lr', type=float, default=0.4, help='learning rate')
parser.add_argument('--mom',type=float, default=0.9, help='momentum')
parser.add_argument('--init',type=float, default=0.001, help='initial weights')
parser.add_argument('--cuda',default=False,action='store_true',help='use cuda')

args = parser.parse_args()

#--- Kovid ---#
parser.set_defaults(target='star16')
# parser.set_defaults(target='heart18')
# parser.set_defaults(target='target1')
# parser.set_defaults(target='target2')
# parser.set_defaults(init=0.001)
parser.set_defaults(lr=0.03)

# parser.set_defaults(target='input')
# parser.set_defaults(dim=9)
# parser.set_defaults(plot=True)


args = parser.parse_args()
#--- Kovid ---#

# choose CPU or CUDA 
if args.cuda:
    device = 'cuda'
else:
    device = 'cpu'

# load specified target values
if args.target == 'input':
    target = torch.eye(args.dim)
elif args.target == 'star16':
    target = star16
elif args.target == 'heart18':
    target = heart18
elif args.target == 'target1':
    target = target1
elif args.target == 'target2':
    target = target2
else:
    print('Unknown target: %s' %args.target )
    exit()

num_in  = target.size()[0]
num_out = target.size()[1]

# print('num_in: ',num_in) #16 for star
# print('num_out: ',num_out) #8 for star
# print('\ntarget', target, '\n')

# input is one-hot with same number of rows as target
input = torch.eye(num_in) # makes a 16x16 diagonal matrix of 1's 

xor_dataset  = torch.utils.data.TensorDataset(input,target)
# print('\nxor_dataset', list(xor_dataset), '\n') # basically input and target tensors

train_loader = torch.utils.data.DataLoader(xor_dataset, batch_size=num_in)
# print('\ntrain_loader', list(train_loader), '\n') # basically input and target tensors

# create neural network according to model specification
net = EncModel(num_in,2,num_out).to(device) # CPU or GPU

# initialize weights, but set biases to zero
net.in_hid.weight.data.normal_(0,args.init)
net.hid_out.weight.data.normal_(0,args.init)
net.in_hid.bias.data.zero_()
net.hid_out.bias.data.zero_()

# SGD optimizer
optimizer = torch.optim.SGD(net.parameters(),lr=args.lr,momentum=args.mom)

# plot only at selected epochs
def plot_epoch( epoch ):
    # return epoch in [50,100,150,200,300,500,700,1000,
    #                   1500,2000,3000,5000,7000,10000,
    #                   15000,20000,30000,50000,70000,100000,
    #                   150000,200000,300000,500000,700000,1000000]
    return epoch in[50,100,150,200,300,500,700,1000,
                      1500,2000,3000]

loss = 1.0
epoch = 0
while epoch < args.epochs and loss > args.stop:
    epoch = epoch+1
    for batch_id, (data,target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad() # zero the gradients
        output = net(data)    # apply network
        # print((output))
        loss = F.binary_cross_entropy(output,target)
        loss.backward()       # compute gradients
        optimizer.step()      # update weights
        if epoch % 10 == 0:
            # print('ep%3d: loss = %7.5f' % (epoch, loss.item()))
            a = 10
        if args.plot and plot_epoch( epoch ):
            plot_hidden(net)
            plt.show()
            
print('\nFinal Epoch',epoch)
plot_hidden(net)
plt.show()

print(f'Target : {args.target}')
print(f'Dim : {args.dim}')
print(f'init : {args.init}')
print(f'Learning rate : {args.lr}')
print(f'Momentum : {args.mom}')

end = time.time() - start
print("\nTime:", "{:.2f}".format(end/60), "mins")