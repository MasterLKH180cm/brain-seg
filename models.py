# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict

class seg_model(nn.Module):

    def __init__(self, args):
        super(seg_model, self).__init__()  

        ''' declare layers used in this network'''
# =============================================================================
#         self.vgg16 = models.vgg16(pretrained=True)
#         self.vgg16 = torch.nn.Sequential(*(list(self.vgg16.children())[:-2]))#.to('cuda:0')
# #        for p in self.vgg16.parameters():
# #            p.requires_grad = False
# #        self.fc1 = nn.Linear(1000, 512*11*14)
#         self.TransConv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True)
#         self.relu1 = nn.ReLU()
#         self.TransConv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=True)
#         self.relu2 = nn.ReLU()
#         self.TransConv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=True)
#         self.relu3 = nn.ReLU()
#         self.TransConv4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=True)
#         self.relu4 = nn.ReLU()
#         self.TransConv5 = nn.ConvTranspose2d(32, 2, kernel_size=4, stride=2, padding=1, bias=True)
#         self.Sigmoid = nn.Sigmoid()
# =============================================================================
        self.unet=UNet(in_channels=3, out_channels=1, init_features=32)
        print(self.unet.parameters())
    def forward(self, x):
#        print(x.shape)
# =============================================================================
#         x = self.vgg16(x)
#         x = self.relu1(self.TransConv1(x))
#         x = self.relu2(self.TransConv2(x))
#         x = self.relu3(self.TransConv3(x))
#         x = self.relu4(self.TransConv4(x))
#         x = self.Sigmoid(self.TransConv5(x))
# =============================================================================
#        print(x.shape)
        
        
#        return x
        return self.unet(x)


class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )