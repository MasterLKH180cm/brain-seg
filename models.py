# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torchvision.models as models


class seg_model(nn.Module):

    def __init__(self, args):
        super(seg_model, self).__init__()  

        ''' declare layers used in this network'''
        self.resnet18 = models.resnet18(pretrained=True)
        self.resnet18 = torch.nn.Sequential(*(list(self.resnet18.children())[:-2]))#.to('cuda:0')

#        self.fc1 = nn.Linear(1000, 512*11*14)
        self.TransConv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.relu1 = nn.ReLU()
        self.TransConv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.relu2 = nn.ReLU()
        self.TransConv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.relu3 = nn.ReLU()
        self.TransConv4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False)
        self.relu4 = nn.ReLU()
        self.TransConv5 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, bias=False)
        self.relu5 = nn.ReLU()
        self.conv = nn.Conv2d(16, 9, kernel_size=1, stride=1, padding=0, bias = True)
        self.fc2 = nn.Linear(9, 9)
        self.relu6 = nn.ReLU()
        self.dropout = nn.Dropout()
        
    def forward(self, img):
        x = self.resnet18(img.transpose(1,3).transpose(2,3))
        x = self.relu1(self.TransConv1(x))
        x = self.relu2(self.TransConv2(x))
        x = self.relu3(self.TransConv3(x))
        x = self.relu4(self.TransConv4(x))
        x = self.relu5(self.TransConv5(x))
        x = self.conv(x)
        
        
        return x

class Improved(nn.Module):

    def __init__(self, args):
        super(Improved, self).__init__()  

        ''' declare layers used in this network'''
        self.googlenet = googlenet(pretrained=True)
        self.googlenet = torch.nn.Sequential(*(list(self.googlenet.children())[:-3]))#.to('cuda:0')
        self.TransConv1 = nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.relu1 = nn.ReLU()
        self.TransConv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.relu2 = nn.ReLU()
        self.TransConv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.relu3 = nn.ReLU()
        self.TransConv4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False)
        self.relu4 = nn.ReLU()
        self.TransConv5 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, bias=False)
        self.relu5 = nn.ReLU()
        self.conv = nn.Conv2d(16, 9, kernel_size=1, stride=1, padding=0, bias = True)

    def forward(self, img):
        x = self.googlenet(self.bn(img.transpose(1,3).transpose(2,3)))

        x = self.relu1(self.TransConv1(x))

        x = self.relu2(self.TransConv2(x))

        x = self.relu3(self.TransConv3(x))

        x = self.relu4(self.TransConv4(x))

        x = self.relu5(self.TransConv5(x))

        x = self.conv(x)

        return x