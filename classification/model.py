"""
@author HAORAN
time: 2021/2/4

"""
import torch
import torchvision
import torch.nn as nn
import numpy as np

class mynet(nn.Module):
    def __init__(self, input_shape=(512, 512)):
        super(mynet, self).__init__()
        self.input = nn.Sequential(
        )
        self.resnet18 = torchvision.models.resnet18()
        self.output = nn.Sequential(

        )
    def forward(self,x):
        x = self.input(x)
        x = self.resnet18(x)
        x = self.output(x)
        return x

