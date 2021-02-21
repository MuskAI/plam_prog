"""
@author HAORAN
time: 2021/2/4

"""
import torch
import torchvision
import torch.nn as nn
import numpy as np
from torchsummary import summary


class PalmNet(nn.Module):
    def __init__(self, input_shape=(512, 512)):
        super(PalmNet, self).__init__()
        self.numberClass = 6
        self.input_rgb = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )
        self.input_rgb_enhancement = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )
        self.prewitt_4 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )

        self.intput_to_densenet = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=3, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(3),
            nn.ReLU()
        )
        self.densenet121 = torchvision.models.densenet121()

        self.outLayer1 = torch.nn.Sequential(
            torch.nn.Linear(1000, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2))
        self.outLayer2 = torch.nn.Linear(512, self.numberClass)

    def forward(self, rgb, rgb_enhancement, prewitt_4):
        x_rgb = self.input_rgb(rgb)
        x_enhancement = self.input_rgb_enhancement(rgb_enhancement)
        x_prewitt_4 = self.prewitt_4(prewitt_4)
        x = torch.cat([x_rgb, x_enhancement], 1)
        x = torch.cat([x, x_prewitt_4], 1)
        x = torch.cat([x, rgb], 1)

        # TODO 1 准备输入densenet
        x = self.intput_to_densenet(x)
        x = self.densenet121(x)
        x = self.outLayer1(x)
        x = self.outLayer2(x)

        return x


if __name__ == '__main__':
    model = PalmNet().cpu()
    summary(model, [[3, 512, 512], [3, 512, 512], [4, 512, 512]],device="cpu")
