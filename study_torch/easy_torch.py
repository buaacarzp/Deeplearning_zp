import torch
# from torch import nn
import numpy as np
'''
review the sample api
'''
X = torch.randn(1,3,256,256)
Y = torch.randn(1,3,256,256)
Z = torch.add(X,Y)
print(Z.shape)
X = torch.nn.Conv2d(in_channels=3,out_channels=8,kernel_size=3,stride=1,padding=0)(X)
X = torch.nn.AvgPool2d(kernel_size=2,stride=2)(X)
X = torch.nn.BatchNorm2d(X.shape[1])(X)
X = torch.flatten(X,1)
X = torch.nn.Linear(X.shape[1],10)(X)
print(X.shape)