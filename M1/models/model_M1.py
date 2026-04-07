# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from models.BT3 import BlockBT3 # Import block của bạn từ file BT3.py

class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class M1_CNN(nn.Module):
    def __init__(self, variant=224, num_classes=5):
        super().__init__()
        self.variant = variant
        
        if variant == 224:
            # Variant 1: Input 224x224x3
            self.stem = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2), # -> 112x112x64
                
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)  # -> 56x56x128
            )
            
            self.stages = nn.Sequential(
                BlockBT3(128, 128, s=1),               # -> 56x56x128
                Downsample(128, 256),                  # -> 28x28x256
                BlockBT3(256, 256, s=1),               # -> 28x28x256
                Downsample(256, 512),                  # -> 14x14x512
                BlockBT3(512, 512, s=1),               # -> 14x14x512
                Downsample(512, 1024)                  # -> 7x7x1024
            )
            
        elif variant == 32:
            # Variant 2: Input 32x32x3 (Điều chỉnh nhẹ để khớp kích thước trong sơ đồ)
            self.stem = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU()                              # -> 32x32x64
            )
            
            self.stages = nn.Sequential(
                Downsample(64, 128),                   # -> 16x16x128
                BlockBT3(128, 128, s=1),
                Downsample(128, 256),                  # -> 8x8x256
                BlockBT3(256, 256, s=1),
                Downsample(256, 512),                  # -> 4x4x512
                BlockBT3(512, 512, s=1),
                Downsample(512, 1024)                  # -> 2x2x1024 (Tiệm cận 1x1)
            )

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1) # GAP -> 1x1x1024
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x # Softmax được tính gộp trong nn.CrossEntropyLoss
