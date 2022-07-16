import copy

import torch
import torch.nn as nn
from layers import Block3x3_leakRelu
from miscc.config import cfg
from torch.nn import ModuleList


class D_GET_LOGITS(nn.Module):
    def __init__(self, ndf, nef, bcondition=False):
        super(D_GET_LOGITS, self).__init__()
        self.df_dim = ndf
        self.ef_dim = nef
        self.bcondition = bcondition
        if self.bcondition:
            self.jointConv = Block3x3_leakRelu(ndf * 8 + nef, ndf * 8)

        self.outlogits = nn.Sequential(nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4), nn.Sigmoid())

    def forward(self, h_code, c_code=None):
        if self.bcondition and c_code is not None:
            # conditioning output
            c_code = c_code.view(-1, self.ef_dim, 1, 1)
            c_code = c_code.repeat(1, 1, 4, 4)
            # state size (ngf+egf) x 4 x 4

            h_c_code = torch.cat((h_code, c_code), 1)
            # state size ngf x in_size x in_size
            h_c_code = self.jointConv(h_c_code)
        else:
            h_c_code = h_code

        output = self.outlogits(h_c_code)
        return output.view(-1)


class MinibatchStdDev(nn.Module):
    def __init__(self, averaging="all"):
        super().__init__()

        # lower case the passed parameter
        self.averaging = averaging.lower()

        if "group" in self.averaging:
            self.n = int(self.averaging[5:])
        else:
            assert self.averaging in ["all", "flat", "spatial", "none", "gpool"], (
                "Invalid averaging mode %s" % self.averaging
            )

        # calculate the std_dev in such a way that it doesn't result in 0
        # otherwise 0 norm operation's gradient is nan
        self.adjusted_std = lambda x, **kwargs: torch.sqrt(
            torch.mean((x - torch.mean(x, **kwargs)) ** 2, **kwargs) + 1e-8
        )

    def forward(self, x):
        """
        forward pass of the Layer
        :param x: input
        :return: y => output
        """
        shape = list(x.size())
        target_shape = copy.deepcopy(shape)

        # compute the std's over the minibatch
        vals = self.adjusted_std(x, dim=0, keepdim=True)

        # perform averaging
        if self.averaging == "all":
            target_shape[1] = 1
            vals = torch.mean(vals, dim=1, keepdim=True)

        elif self.averaging == "spatial":
            if len(shape) == 4:
                vals = torch.mean(torch.mean(vals, 2, keepdim=True), 3, keepdim=True)

        elif self.averaging == "none":
            target_shape = [target_shape[0]] + [s for s in target_shape[1:]]

        elif self.averaging == "gpool":
            if len(shape) == 4:
                vals = torch.mean(torch.mean(torch.mean(x, 2, keepdim=True), 3, keepdim=True), 0, keepdim=True)
        elif self.averaging == "flat":
            target_shape[1] = 1
            vals = torch.FloatTensor([self.adjusted_std(x)])

        else:  # self.averaging == 'group'
            target_shape[1] = self.n
            vals = vals.view(self.n, self.shape[1] / self.n, self.shape[2], self.shape[3])
            vals = torch.mean(vals, 0, keepdim=True).view(1, self.n, 1, 1)

        # spatial replication of the computed statistic
        vals = vals.expand(*target_shape)

        # concatenate the constant feature map to the input
        y = torch.cat([x, vals], 1)

        # return the computed value
        return y


class DisGeneralConvBlock(nn.Module):
    def __init__(self, in_channels, concat_channels, out_channels):
        super().__init__()
        self.batch_discriminator = MinibatchStdDev()
        self.block = nn.Sequential(
            Block3x3_leakRelu(in_channels + concat_channels, in_channels),
            Block3x3_leakRelu(in_channels, out_channels),
            nn.AvgPool2d(2),
        )

    def forward(self, x):
        out = self.batch_discriminator(x)
        out = self.block(out)
        return out


class MSG_D_NET(nn.Module):
    def __init__(self, depth, b_jcu=True):
        super().__init__()
        ndf = cfg.GAN.DF_DIM
        nef = cfg.TEXT.EMBEDDING_DIM

        self.b_jcu = b_jcu
        self.depth = depth

        self.fRGB_0 = nn.Conv2d(3, ndf, kernel_size=(1, 1), stride=1, padding=0)

        self.layers = ModuleList([])
        for i in range(self.depth):
            if i < 3:
                layer = DisGeneralConvBlock(ndf * (2**i), 4 if i > 0 else 1, ndf * (2 ** (i + 1)))
            else:
                layer = DisGeneralConvBlock(ndf * 8, 4, ndf * 8)
            self.layers.append(layer)

        if self.b_jcu:
            self.UNCOND_DNET = D_GET_LOGITS(ndf, nef, bcondition=False)
        else:
            self.UNCOND_DNET = None
        self.COND_DNET = D_GET_LOGITS(ndf, nef, bcondition=True)

    def forward(self, inputs):
        """
        :param inputs: (multi-scale input images) to the network, list[Tensor]
        :return out => raw predition value
        """

        out = self.fRGB_0(inputs[-1])
        out = self.layers[0](out)

        for x, block in zip(reversed(inputs[:-1]), self.layers[1:]):
            out = torch.cat((x, out), dim=1)
            out = block(out)  # ==> ndf*8 x 4 x 4

        return out
