import torch.nn as nn
import torch.nn.functional as F
from spectral import SpectralNorm


class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, "channels dont divide 2!"
        nc = int(nc / 2)
        return x[:, :nc] * F.sigmoid(x[:, nc:])


def conv1x1(in_planes, out_planes, bias=False):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=bias)


def conv3x3(in_planes, out_planes, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=bias)


# Upsale the spatial size by a factor of 2
def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU(),
    )
    return block


# Keep the spatial size
def Block3x3_relu(in_planes, out_planes):
    block = nn.Sequential(conv3x3(in_planes, out_planes * 2), nn.BatchNorm2d(out_planes * 2), GLU())
    return block


class ResBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            conv3x3(channel_num, channel_num * 2),
            nn.BatchNorm2d(channel_num * 2),
            GLU(),
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num),
        )

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return out


# ############## D networks ##########################
def Block3x3_leakRelu(in_planes, out_planes):
    block = nn.Sequential(SpectralNorm(conv3x3(in_planes, out_planes, bias=True)), nn.LeakyReLU(0.2, inplace=True))
    return block


# Downsale the spatial size by a factor of 2
def downBlock(in_planes, out_planes):
    block = nn.Sequential(
        SpectralNorm(nn.Conv2d(in_planes, out_planes, 4, 2, 1, bias=True)), nn.LeakyReLU(0.2, inplace=True)
    )
    return block


# Downsale the spatial size by a factor of 16
def encode_image_by_16times(ndf):
    layers = []
    layers.append(SpectralNorm(nn.Conv2d(3, ndf, 4, 2, 1, bias=True)))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    layers.append(SpectralNorm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=True)))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    layers.append(SpectralNorm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=True)))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    layers.append(SpectralNorm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=True)))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)
