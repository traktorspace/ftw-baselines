# Source code of FAUNet
# https://github.com/awadbahaa/FAUNet/tree/main/FAUNet_Train.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def channel_normalize(c, NEWMAX=2000):
    c_max = np.max(c)
    c_min = np.min(c)

    if c_max > NEWMAX:
        c = np.where(c > NEWMAX, NEWMAX, c)
        c_max = NEWMAX

    if c_max > 0:
        c_normalized = (c - c_min) / (c_max - c_min)
    else:
        c_normalized = c.copy()

    return c_normalized


def faunet_normalize(A, CHW=False):
    if CHW:
        c1 = channel_normalize(A[0, :, :])
        c2 = channel_normalize(A[1, :, :])
        c3 = channel_normalize(A[2, :, :])
        A = np.dstack((c1, c2, c3))
        A = A.transpose((2, 0, 1))
    else:
        c1 = channel_normalize(A[:, :, 0])
        c2 = channel_normalize(A[:, :, 1])
        c3 = channel_normalize(A[:, :, 2])
        A = np.dstack((c1, c2, c3))
    return A


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


def creathighpass(nchannel, outchannel, device):
    high = torch.tensor(
        [[0, -0.25, 0], [-0.25, 0, -0.25], [0, -0.25, 0]]
    )  ### make it 1
    high = high.unsqueeze(0).repeat(outchannel, 1, 1)
    high = high.unsqueeze(0).repeat(nchannel, 1, 1, 1)
    high = high.to(device)
    return high


class FAUNet(nn.Module):
    def __init__(self, n_channels, n_classes, device, bilinear=True):
        """
        Official github implementation of FAUNet model
        Reference: https://www.mdpi.com/2072-4292/15/21/5123
        """
        super(FAUNet, self).__init__()

        self.device = device

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        self.up1edge = Up(1024, 512 // factor, bilinear)
        self.up1 = Up(1024, 512 // factor, bilinear)

        self.up2edge = Up(512, 256 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)

        self.up3edge = Up(256, 128 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)

        self.up4edge = Up(128, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)

        self.outc = OutConv(64, n_classes)
        self.outc_edge = OutConv(64, n_classes)

        self.highpass1 = creathighpass(64, 64, self.device)
        self.highpass2 = creathighpass(128, 128, self.device)

        self.attention_weights1 = torch.nn.Parameter(torch.randn(64, 64, 1, 1))
        self.attention_weights2 = torch.nn.Parameter(torch.randn(128, 128, 1, 1))

    def forward(self, x):
        sigma1 = torch.nn.ReLU()
        # Sigma 2 apparently unused, left commented
        # sigma2 = nn.Sigmoid()

        x1 = self.inc(x)
        f_high1 = F.conv2d(x1, self.highpass1, padding="same")

        x2 = self.down1(x1)
        f_high2 = F.conv2d(x2, self.highpass2, padding="same")

        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x_edge = self.up1edge(x5, x4)

        x = self.up2(x, x3)
        x_edge = self.up2edge(x_edge, x3)

        # attention gate2
        #############################################################
        f_high2 = sigma1(f_high2)
        attn_map2 = torch.nn.functional.conv2d(
            f_high2, self.attention_weights2, padding=0
        )
        attn_map2 = torch.nn.functional.softmax(attn_map2, dim=1)
        x2h = attn_map2 * x2
        #############################################################

        x = self.up3(x, x2)
        x_edge = self.up3edge(x_edge, x2h)

        # attention gate1
        #############################################################
        f_high1 = sigma1(f_high1)
        attn_map1 = torch.nn.functional.conv2d(
            f_high1, self.attention_weights1, padding=0
        )
        attn_map1 = torch.nn.functional.softmax(attn_map1, dim=1)
        x1h = attn_map1 * x1
        #############################################################

        x = self.up4(x, x1)
        x_edge = self.up4edge(x_edge, x1h)

        logits = self.outc(x)
        logits_edge = self.outc_edge(x_edge)

        # Returns logits for extent mask and for boundary mask
        return logits, logits_edge
