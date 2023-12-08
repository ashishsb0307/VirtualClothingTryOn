import torch
from torch.utils import data
from os import path as osp
from torchvision import transforms
import json
import numpy as np
from PIL import Image, ImageDraw
from torch import nn
from torch.nn import init
import torchgeometry as tgm
from torch.nn import functional as F
import cv2
import os
from base_network import BaseNetwork

class SegGenerator(BaseNetwork):
    def __init__(self, opt, input_nc, output_nc=13, norm_layer=nn.InstanceNorm2d):
        super(SegGenerator, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(input_nc, 64, kernel_size=3, padding=1), norm_layer(64), nn.ReLU(),
                                nn.Conv2d(64, 64, kernel_size=3, padding=1), norm_layer(64), nn.ReLU())

        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1), norm_layer(128), nn.ReLU(),
                                nn.Conv2d(128, 128, kernel_size=3, padding=1), norm_layer(128), nn.ReLU())

        self.conv3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1), norm_layer(256), nn.ReLU(),
                                nn.Conv2d(256, 256, kernel_size=3, padding=1), norm_layer(256), nn.ReLU())

        self.conv4 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, padding=1), norm_layer(512), nn.ReLU(),
                                nn.Conv2d(512, 512, kernel_size=3, padding=1), norm_layer(512), nn.ReLU())

        self.conv5 = nn.Sequential(nn.Conv2d(512, 1024, kernel_size=3, padding=1), norm_layer(1024), nn.ReLU(),
                                nn.Conv2d(1024, 1024, kernel_size=3, padding=1), norm_layer(1024), nn.ReLU())

        self.up6 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                nn.Conv2d(1024, 512, kernel_size=3, padding=1), norm_layer(512), nn.ReLU())
        self.conv6 = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=3, padding=1), norm_layer(512), nn.ReLU(),
                                nn.Conv2d(512, 512, kernel_size=3, padding=1), norm_layer(512), nn.ReLU())

        self.up7 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                nn.Conv2d(512, 256, kernel_size=3, padding=1), norm_layer(256), nn.ReLU())
        self.conv7 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1), norm_layer(256), nn.ReLU(),
                                nn.Conv2d(256, 256, kernel_size=3, padding=1), norm_layer(256), nn.ReLU())

        self.up8 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                nn.Conv2d(256, 128, kernel_size=3, padding=1), norm_layer(128), nn.ReLU())
        self.conv8 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, padding=1), norm_layer(128), nn.ReLU(),
                                nn.Conv2d(128, 128, kernel_size=3, padding=1), norm_layer(128), nn.ReLU())

        self.up9 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                nn.Conv2d(128, 64, kernel_size=3, padding=1), norm_layer(64), nn.ReLU())
        self.conv9 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), norm_layer(64), nn.ReLU(),
                                nn.Conv2d(64, 64, kernel_size=3, padding=1), norm_layer(64), nn.ReLU(),
                                nn.Conv2d(64, output_nc, kernel_size=3, padding=1))

        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

        self.print_network()
        self.init_weights(opt.init_type, opt.init_variance)
    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.drop(self.conv4(self.pool(conv3)))
        conv5 = self.drop(self.conv5(self.pool(conv4)))

        conv6 = self.conv6(torch.cat((conv4, self.up6(conv5)), 1))
        conv7 = self.conv7(torch.cat((conv3, self.up7(conv6)), 1))
        conv8 = self.conv8(torch.cat((conv2, self.up8(conv7)), 1))
        conv9 = self.conv9(torch.cat((conv1, self.up9(conv8)), 1))
        return self.sigmoid(conv9)

