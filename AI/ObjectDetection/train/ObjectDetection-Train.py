from torch.utils.tensorboard import SummaryWriter as TensorBoardSummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim.lr_scheduler as lr_scheduler
import torch.distributed as dist
import torch.nn.functional as F
from threading import Thread
import torch.optim as optim
from torch.cuda import amp
from copy import deepcopy
from pathlib import Path
import torch.utils.data
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import argparse
import logging
import random
import time
import math
import yaml
import os


logger = logging.getLogger(__name__)




#######################################################################################################
# START: common.py
#######################################################################################################


def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class MP(nn.Module):
    def __init__(self, k=2):
        super(MP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=k)

    def forward(self, x):
        return self.m(x)


class SP(nn.Module):
    def __init__(self, k=3, s=1):
        super(SP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=s, padding=k // 2)

    def forward(self, x):
        return self.m(x)

    
class ReOrg(nn.Module):
    def __init__(self):
        super(ReOrg, self).__init__()

    def forward(self, x):
        return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)


class Concat(nn.Module):
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class Chuncat(nn.Module):
    def __init__(self, dimension=1):
        super(Chuncat, self).__init__()
        self.d = dimension

    def forward(self, x):
        x1 = []
        x2 = []
        for xi in x:
            xi1, xi2 = xi.chunk(2, self.d)
            x1.append(xi1)
            x2.append(xi2)
        return torch.cat(x1+x2, self.d)


class Shortcut(nn.Module):
    def __init__(self, dimension=0):
        super(Shortcut, self).__init__()
        self.d = dimension

    def forward(self, x):
        return x[0]+x[1]


class Foldcut(nn.Module):
    def __init__(self, dimension=0):
        super(Foldcut, self).__init__()
        self.d = dimension

    def forward(self, x):
        x1, x2 = x.chunk(2, self.d)
        return x1+x2


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class RobustConv(nn.Module):
    def __init__(self, c1, c2, k=7, s=1, p=None, g=1, act=True, layer_scale_init_value=1e-6):
        super(RobustConv, self).__init__()
        self.conv_dw = Conv(c1, c1, k=k, s=s, p=p, g=c1, act=act)
        self.conv1x1 = nn.Conv2d(c1, c2, 1, 1, 0, groups=1, bias=True)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(c2)) if layer_scale_init_value > 0 else None

    def forward(self, x):
        x = x.to(memory_format=torch.channels_last)
        x = self.conv1x1(self.conv_dw(x))
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1)) 
        return x


class RobustConv2(nn.Module):
    def __init__(self, c1, c2, k=7, s=4, p=None, g=1, act=True, layer_scale_init_value=1e-6):
        super(RobustConv2, self).__init__()
        self.conv_strided = Conv(c1, c1, k=k, s=s, p=p, g=c1, act=act)
        self.conv_deconv = nn.ConvTranspose2d(in_channels=c1, out_channels=c2, kernel_size=s, stride=s, 
                                              padding=0, bias=True, dilation=1, groups=1)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(c2)) if layer_scale_init_value > 0 else None

    def forward(self, x):
        x = self.conv_deconv(self.conv_strided(x))
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1)) 
        return x
    

def DWConv(c1, c2, k=1, s=1, act=True):
    return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class GhostConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super(GhostConv, self).__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, None, g, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat([y, self.cv2(y)], 1)


class Stem(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Stem, self).__init__()
        c_ = int(c2/2)
        self.cv1 = Conv(c1, c_, 3, 2)
        self.cv2 = Conv(c_, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 2)
        self.pool = torch.nn.MaxPool2d(2, stride=2)
        self.cv4 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x = self.cv1(x)
        return self.cv4(torch.cat((self.cv3(self.cv2(x)), self.pool(x)), dim=1))


class DownC(nn.Module):
    def __init__(self, c1, c2, n=1, k=2):
        super(DownC, self).__init__()
        c_ = int(c1)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2//2, 3, k)
        self.cv3 = Conv(c1, c2//2, 1, 1)
        self.mp = nn.MaxPool2d(kernel_size=k, stride=k)

    def forward(self, x):
        return torch.cat((self.cv2(self.cv1(x)), self.cv3(self.mp(x))), dim=1)


class SPP(nn.Module):
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class Res(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super(Res, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c_, 3, 1, g=g)
        self.cv3 = Conv(c_, c2, 1, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv3(self.cv2(self.cv1(x))) if self.add else self.cv3(self.cv2(self.cv1(x)))


class ResX(Res):
    def __init__(self, c1, c2, shortcut=True, g=32, e=0.5):
        super().__init__(c1, c2, shortcut, g, e)
        c_ = int(c2 * e)


class Ghost(nn.Module):
    def __init__(self, c1, c2, k=3, s=1):
        super(Ghost, self).__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(GhostConv(c1, c_, 1, 1),  # pw
                                  DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),
                                  GhostConv(c_, c2, 1, 1, act=False))
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False),
                                      Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class SPPCSPC(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(SPPCSPC, self).__init__()
        c_ = int(2 * c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))

class GhostSPPCSPC(SPPCSPC):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(2 * c2 * e)
        self.cv1 = GhostConv(c1, c_, 1, 1)
        self.cv2 = GhostConv(c1, c_, 1, 1)
        self.cv3 = GhostConv(c_, c_, 3, 1)
        self.cv4 = GhostConv(c_, c_, 1, 1)
        self.cv5 = GhostConv(4 * c_, c_, 1, 1)
        self.cv6 = GhostConv(c_, c_, 3, 1)
        self.cv7 = GhostConv(2 * c_, c2, 1, 1)


class GhostStem(Stem):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__(c1, c2, k, s, p, g, act)
        c_ = int(c2/2)
        self.cv1 = GhostConv(c1, c_, 3, 2)
        self.cv2 = GhostConv(c_, c_, 1, 1)
        self.cv3 = GhostConv(c_, c_, 3, 2)
        self.cv4 = GhostConv(2 * c_, c2, 1, 1)


class BottleneckCSPA(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super(BottleneckCSPA, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.m(self.cv1(x))
        y2 = self.cv2(x)
        return self.cv3(torch.cat((y1, y2), dim=1))


class BottleneckCSPB(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super(BottleneckCSPB, self).__init__()
        c_ = int(c2)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        x1 = self.cv1(x)
        y1 = self.m(x1)
        y2 = self.cv2(x1)
        return self.cv3(torch.cat((y1, y2), dim=1))


class BottleneckCSPC(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super(BottleneckCSPC, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 1, 1)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(torch.cat((y1, y2), dim=1))


class ResCSPA(BottleneckCSPA):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*[Res(c_, c_, shortcut, g, e=0.5) for _ in range(n)])


class ResCSPB(BottleneckCSPB):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2)
        self.m = nn.Sequential(*[Res(c_, c_, shortcut, g, e=0.5) for _ in range(n)])


class ResCSPC(BottleneckCSPC):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*[Res(c_, c_, shortcut, g, e=0.5) for _ in range(n)])


class ResXCSPA(ResCSPA):
    def __init__(self, c1, c2, n=1, shortcut=True, g=32, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*[Res(c_, c_, shortcut, g, e=1.0) for _ in range(n)])


class ResXCSPB(ResCSPB):
    def __init__(self, c1, c2, n=1, shortcut=True, g=32, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2)
        self.m = nn.Sequential(*[Res(c_, c_, shortcut, g, e=1.0) for _ in range(n)])


class ResXCSPC(ResCSPC):
    def __init__(self, c1, c2, n=1, shortcut=True, g=32, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*[Res(c_, c_, shortcut, g, e=1.0) for _ in range(n)])


class GhostCSPA(BottleneckCSPA):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*[Ghost(c_, c_) for _ in range(n)])


class GhostCSPB(BottleneckCSPB):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2)
        self.m = nn.Sequential(*[Ghost(c_, c_) for _ in range(n)])


class GhostCSPC(BottleneckCSPC):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*[Ghost(c_, c_) for _ in range(n)])


class ImplicitA(nn.Module):
    def __init__(self, channel, mean=0., std=.02):
        super(ImplicitA, self).__init__()
        self.channel = channel
        self.mean = mean
        self.std = std
        self.implicit = nn.Parameter(torch.zeros(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

    def forward(self, x):
        return self.implicit + x


class ImplicitM(nn.Module):
    def __init__(self, channel, mean=1., std=.02):
        super(ImplicitM, self).__init__()
        self.channel = channel
        self.mean = mean
        self.std = std
        self.implicit = nn.Parameter(torch.ones(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

    def forward(self, x):
        return self.implicit * x


class RepConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True, deploy=False):
        super(RepConv, self).__init__()
        self.deploy = deploy
        self.groups = g
        self.in_channels = c1
        self.out_channels = c2
        assert k == 3
        assert autopad(k, p) == 1
        padding_11 = autopad(k, p) - k // 2
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        if deploy:
            self.rbr_reparam = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=True)
        else:
            self.rbr_identity = (nn.BatchNorm2d(num_features=c1) if c2 == c1 and s == 1 else None)
            self.rbr_dense = nn.Sequential(
                nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False),
                nn.BatchNorm2d(num_features=c2),
            )
            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d( c1, c2, 1, s, padding_11, groups=g, bias=False),
                nn.BatchNorm2d(num_features=c2),
            )

    def forward(self, inputs):
        if hasattr(self, "rbr_reparam"):
            return self.act(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        return self.act(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)
    
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return (
            kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid,
            bias3x3 + bias1x1 + biasid,
        )

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch[0].weight
            running_mean = branch[1].running_mean
            running_var = branch[1].running_var
            gamma = branch[1].weight
            beta = branch[1].bias
            eps = branch[1].eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros(
                    (self.in_channels, input_dim, 3, 3), dtype=np.float32
                )
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def repvgg_convert(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        return (
            kernel.detach().cpu().numpy(),
            bias.detach().cpu().numpy(),
        )

    def fuse_conv_bn(self, conv, bn):
        std = (bn.running_var + bn.eps).sqrt()
        bias = bn.bias - bn.running_mean * bn.weight / std
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        weights = conv.weight * t
        bn = nn.Identity()
        conv = nn.Conv2d(in_channels = conv.in_channels,
                              out_channels = conv.out_channels,
                              kernel_size = conv.kernel_size,
                              stride=conv.stride,
                              padding = conv.padding,
                              dilation = conv.dilation,
                              groups = conv.groups,
                              bias = True,
                              padding_mode = conv.padding_mode)
        conv.weight = torch.nn.Parameter(weights)
        conv.bias = torch.nn.Parameter(bias)
        return conv

    def fuse_repvgg_block(self):    
        if self.deploy:
            return
        self.rbr_dense = self.fuse_conv_bn(self.rbr_dense[0], self.rbr_dense[1])
        self.rbr_1x1 = self.fuse_conv_bn(self.rbr_1x1[0], self.rbr_1x1[1])
        rbr_1x1_bias = self.rbr_1x1.bias
        weight_1x1_expanded = torch.nn.functional.pad(self.rbr_1x1.weight, [1, 1, 1, 1])
        if (isinstance(self.rbr_identity, nn.BatchNorm2d) or isinstance(self.rbr_identity, nn.modules.batchnorm.SyncBatchNorm)):
            identity_conv_1x1 = nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    groups=self.groups, 
                    bias=False)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.to(self.rbr_1x1.weight.data.device)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.squeeze().squeeze()
            identity_conv_1x1.weight.data.fill_(0.0)
            identity_conv_1x1.weight.data.fill_diagonal_(1.0)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.unsqueeze(2).unsqueeze(3)
            identity_conv_1x1 = self.fuse_conv_bn(identity_conv_1x1, self.rbr_identity)
            bias_identity_expanded = identity_conv_1x1.bias
            weight_identity_expanded = torch.nn.functional.pad(identity_conv_1x1.weight, [1, 1, 1, 1])            
        else:
            bias_identity_expanded = torch.nn.Parameter( torch.zeros_like(rbr_1x1_bias) )
            weight_identity_expanded = torch.nn.Parameter( torch.zeros_like(weight_1x1_expanded) )            
        self.rbr_dense.weight = torch.nn.Parameter(self.rbr_dense.weight + weight_1x1_expanded + weight_identity_expanded)
        self.rbr_dense.bias = torch.nn.Parameter(self.rbr_dense.bias + rbr_1x1_bias + bias_identity_expanded)
        self.rbr_reparam = self.rbr_dense
        self.deploy = True
        if self.rbr_identity is not None:
            del self.rbr_identity
            self.rbr_identity = None
        if self.rbr_1x1 is not None:
            del self.rbr_1x1
            self.rbr_1x1 = None
        if self.rbr_dense is not None:
            del self.rbr_dense
            self.rbr_dense = None


class RepBottleneck(Bottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, shortcut=True, g=1, e=0.5)
        c_ = int(c2 * e)
        self.cv2 = RepConv(c_, c2, 3, 1, g=g)


class RepBottleneckCSPA(BottleneckCSPA):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*[RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])


class RepBottleneckCSPB(BottleneckCSPB):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2)
        self.m = nn.Sequential(*[RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])


class RepBottleneckCSPC(BottleneckCSPC):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*[RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])


class RepRes(Res):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, shortcut, g, e)
        c_ = int(c2 * e)
        self.cv2 = RepConv(c_, c_, 3, 1, g=g)


class RepResCSPA(ResCSPA):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*[RepRes(c_, c_, shortcut, g, e=0.5) for _ in range(n)])


class RepResCSPB(ResCSPB):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2)
        self.m = nn.Sequential(*[RepRes(c_, c_, shortcut, g, e=0.5) for _ in range(n)])


class RepResCSPC(ResCSPC):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*[RepRes(c_, c_, shortcut, g, e=0.5) for _ in range(n)])


class RepResX(ResX):
    def __init__(self, c1, c2, shortcut=True, g=32, e=0.5):
        super().__init__(c1, c2, shortcut, g, e)
        c_ = int(c2 * e)
        self.cv2 = RepConv(c_, c_, 3, 1, g=g)


class RepResXCSPA(ResXCSPA):
    def __init__(self, c1, c2, n=1, shortcut=True, g=32, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*[RepResX(c_, c_, shortcut, g, e=0.5) for _ in range(n)])


class RepResXCSPB(ResXCSPB):
    def __init__(self, c1, c2, n=1, shortcut=False, g=32, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2)
        self.m = nn.Sequential(*[RepResX(c_, c_, shortcut, g, e=0.5) for _ in range(n)])


class RepResXCSPC(ResXCSPC):
    def __init__(self, c1, c2, n=1, shortcut=True, g=32, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*[RepResX(c_, c_, shortcut, g, e=0.5) for _ in range(n)])


class TransformerLayer(nn.Module):
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)
        self.tr = nn.Sequential(*[TransformerLayer(c2, num_heads) for _ in range(num_layers)])
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2)
        p = p.unsqueeze(0)
        p = p.transpose(0, 3)
        p = p.squeeze(3)
        e = self.linear(p)
        x = p + e
        x = self.tr(x)
        x = x.unsqueeze(3)
        x = x.transpose(0, 3)
        x = x.reshape(b, self.c2, w, h)
        return x


class Focus(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)

    def forward(self, x):
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))


class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class Contract(nn.Module):
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()
        s = self.gain
        x = x.view(N, C, H // s, s, W // s, s)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        return x.view(N, C * s * s, H // s, W // s)


class Expand(nn.Module):
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()
        s = self.gain
        x = x.view(N, s, s, C // s ** 2, H, W)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()
        return x.view(N, C // s ** 2, H * s, W * s)


class NMS(nn.Module):
    conf = 0.25
    iou = 0.45
    classes = None

    def __init__(self):
        super(NMS, self).__init__()

    def forward(self, x):
        return non_max_suppression(x[0], conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)


class autoShape(nn.Module):
    conf = 0.25
    iou = 0.45
    classes = None

    def __init__(self, model):
        super(autoShape, self).__init__()
        self.model = model.eval()

    def autoshape(self):
        print('autoShape already enabled, skipping... ')
        return self

    @torch.no_grad()
    def forward(self, imgs, size=640, augment=False, profile=False):
        t = [time_synchronized()]
        p = next(self.model.parameters())  # for device and type
        if isinstance(imgs, torch.Tensor):  # torch
            with amp.autocast(enabled=p.device.type != 'cpu'):
                return self.model(imgs.to(p.device).type_as(p), augment, profile)
        n, imgs = (len(imgs), imgs) if isinstance(imgs, list) else (1, [imgs])
        shape0, shape1, files = [], [], []
        for i, im in enumerate(imgs):
            f = f'image{i}'
            if isinstance(im, str):
                im, f = np.asarray(Image.open(requests.get(im, stream=True).raw if im.startswith('http') else im)), im
            elif isinstance(im, Image.Image):
                im, f = np.asarray(im), getattr(im, 'filename', f) or f
            files.append(Path(f).with_suffix('.jpg').name)
            if im.shape[0] < 5:
                im = im.transpose((1, 2, 0))
            im = im[:, :, :3] if im.ndim == 3 else np.tile(im[:, :, None], 3)
            s = im.shape[:2]
            shape0.append(s)
            g = (size / max(s))
            shape1.append([y * g for y in s])
            imgs[i] = im
        shape1 = [make_divisible(x, int(self.stride.max())) for x in np.stack(shape1, 0).max(0)]
        x = [letterbox(im, new_shape=shape1, auto=False)[0] for im in imgs]
        x = np.stack(x, 0) if n > 1 else x[0][None]
        x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))
        x = torch.from_numpy(x).to(p.device).type_as(p) / 255.
        t.append(time_synchronized())
        with amp.autocast(enabled=p.device.type != 'cpu'):
            y = self.model(x, augment, profile)[0]
            t.append(time_synchronized())
            y = non_max_suppression(y, conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)
            for i in range(n):
                scale_coords(shape1, y[i][:, :4], shape0[i])
            t.append(time_synchronized())
            return Detections(imgs, y, files, t, self.names, x.shape)


class Detections:
    def __init__(self, imgs, pred, files, times=None, names=None, shape=None):
        super(Detections, self).__init__()
        d = pred[0].device
        gn = [torch.tensor([*[im.shape[i] for i in [1, 0, 1, 0]], 1., 1.], device=d) for im in imgs]
        self.imgs = imgs
        self.pred = pred
        self.names = names
        self.files = files
        self.xyxy = pred
        self.xywh = [xyxy2xywh(x) for x in pred]
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]
        self.n = len(self.pred)
        self.t = tuple((times[i + 1] - times[i]) * 1000 / self.n for i in range(3))
        self.s = shape

    def display(self, pprint=False, show=False, save=False, render=False, save_dir=''):
        colors = color_list()
        for i, (img, pred) in enumerate(zip(self.imgs, self.pred)):
            str = f'image {i + 1}/{len(self.pred)}: {img.shape[0]}x{img.shape[1]} '
            if pred is not None:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()
                    str += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "
                if show or save or render:
                    for *box, conf, cls in pred:
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        plot_one_box(box, img, label=label, color=colors[int(cls) % 10])
            img = Image.fromarray(img.astype(np.uint8)) if isinstance(img, np.ndarray) else img
            if pprint:
                print(str.rstrip(', '))
            if show:
                img.show(self.files[i])
            if save:
                f = self.files[i]
                img.save(Path(save_dir) / f)
                print(f"{'Saved' * (i == 0)} {f}", end=',' if i < self.n - 1 else f' to {save_dir}\n')
            if render:
                self.imgs[i] = np.asarray(img)

    def print(self):
        self.display(pprint=True)
        print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {tuple(self.s)}' % self.t)

    def show(self):
        self.display(show=True)

    def save(self, save_dir='runs/hub/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/hub/exp')
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        self.display(save=True, save_dir=save_dir)

    def render(self):
        self.display(render=True)
        return self.imgs

    def pandas(self):
        new = copy(self)
        ca = 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'
        cb = 'xcenter', 'ycenter', 'width', 'height', 'confidence', 'class', 'name'
        for k, c in zip(['xyxy', 'xyxyn', 'xywh', 'xywhn'], [ca, ca, cb, cb]):
            a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()] for x in getattr(self, k)]
            setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
        return new

    def tolist(self):
        x = [Detections([self.imgs[i]], [self.pred[i]], self.names, self.s) for i in range(self.n)]
        for d in x:
            for k in ['imgs', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
                setattr(d, k, getattr(d, k)[0])
        return x

    def __len__(self):
        return self.n


class Classify(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        super(Classify, self).__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g)
        self.flat = nn.Flatten()

    def forward(self, x):
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)
        return self.flat(self.conv(z))


def transI_fusebn(kernel, bn):
    gamma = bn.weight
    std = (bn.running_var + bn.eps).sqrt()
    return kernel * ((gamma / std).reshape(-1, 1, 1, 1)), bn.bias - bn.running_mean * gamma / std


class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                             stride=1, padding=0, dilation=1, groups=1, deploy=False, nonlinear=None):
        super().__init__()
        if nonlinear is None:
            self.nonlinear = nn.Identity()
        else:
            self.nonlinear = nonlinear
        if deploy:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True)
        else:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False)
            self.bn = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        if hasattr(self, 'bn'):
            return self.nonlinear(self.bn(self.conv(x)))
        else:
            return self.nonlinear(self.conv(x))

    def switch_to_deploy(self):
        kernel, bias = transI_fusebn(self.conv.weight, self.bn)
        conv = nn.Conv2d(in_channels=self.conv.in_channels, out_channels=self.conv.out_channels, kernel_size=self.conv.kernel_size, stride=self.conv.stride, padding=self.conv.padding, dilation=self.conv.dilation, groups=self.conv.groups, bias=True)
        conv.weight.data = kernel
        conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('conv')
        self.__delattr__('bn')
        self.conv = conv    


class OREPA_3x3_RepConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 internal_channels_1x1_3x3=None,
                 deploy=False, nonlinear=None, single_init=False):
        super(OREPA_3x3_RepConv, self).__init__()
        self.deploy = deploy

        if nonlinear is None:
            self.nonlinear = nn.Identity()
        else:
            self.nonlinear = nonlinear
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        assert padding == kernel_size // 2
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.branch_counter = 0
        self.weight_rbr_origin = nn.Parameter(torch.Tensor(out_channels, int(in_channels/self.groups), kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight_rbr_origin, a=math.sqrt(1.0))
        self.branch_counter += 1
        if groups < out_channels:
            self.weight_rbr_avg_conv = nn.Parameter(torch.Tensor(out_channels, int(in_channels/self.groups), 1, 1))
            self.weight_rbr_pfir_conv = nn.Parameter(torch.Tensor(out_channels, int(in_channels/self.groups), 1, 1))
            nn.init.kaiming_uniform_(self.weight_rbr_avg_conv, a=1.0)
            nn.init.kaiming_uniform_(self.weight_rbr_pfir_conv, a=1.0)
            self.weight_rbr_avg_conv.data
            self.weight_rbr_pfir_conv.data
            self.register_buffer('weight_rbr_avg_avg', torch.ones(kernel_size, kernel_size).mul(1.0/kernel_size/kernel_size))
            self.branch_counter += 1
        else:
            raise NotImplementedError
        self.branch_counter += 1
        if internal_channels_1x1_3x3 is None:
            internal_channels_1x1_3x3 = in_channels if groups < out_channels else 2 * in_channels
        if internal_channels_1x1_3x3 == in_channels:
            self.weight_rbr_1x1_kxk_idconv1 = nn.Parameter(torch.zeros(in_channels, int(in_channels/self.groups), 1, 1))
            id_value = np.zeros((in_channels, int(in_channels/self.groups), 1, 1))
            for i in range(in_channels):
                id_value[i, i % int(in_channels/self.groups), 0, 0] = 1
            id_tensor = torch.from_numpy(id_value).type_as(self.weight_rbr_1x1_kxk_idconv1)
            self.register_buffer('id_tensor', id_tensor)

        else:
            self.weight_rbr_1x1_kxk_conv1 = nn.Parameter(torch.Tensor(internal_channels_1x1_3x3, int(in_channels/self.groups), 1, 1))
            nn.init.kaiming_uniform_(self.weight_rbr_1x1_kxk_conv1, a=math.sqrt(1.0))
        self.weight_rbr_1x1_kxk_conv2 = nn.Parameter(torch.Tensor(out_channels, int(internal_channels_1x1_3x3/self.groups), kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight_rbr_1x1_kxk_conv2, a=math.sqrt(1.0))
        self.branch_counter += 1
        expand_ratio = 8
        self.weight_rbr_gconv_dw = nn.Parameter(torch.Tensor(in_channels*expand_ratio, 1, kernel_size, kernel_size))
        self.weight_rbr_gconv_pw = nn.Parameter(torch.Tensor(out_channels, in_channels*expand_ratio, 1, 1))
        nn.init.kaiming_uniform_(self.weight_rbr_gconv_dw, a=math.sqrt(1.0))
        nn.init.kaiming_uniform_(self.weight_rbr_gconv_pw, a=math.sqrt(1.0))
        self.branch_counter += 1
        if out_channels == in_channels and stride == 1:
            self.branch_counter += 1
        self.vector = nn.Parameter(torch.Tensor(self.branch_counter, self.out_channels))
        self.bn = nn.BatchNorm2d(out_channels)
        self.fre_init()
        nn.init.constant_(self.vector[0, :], 0.25)
        nn.init.constant_(self.vector[1, :], 0.25)
        nn.init.constant_(self.vector[2, :], 0.0)
        nn.init.constant_(self.vector[3, :], 0.5)
        nn.init.constant_(self.vector[4, :], 0.5)

    def fre_init(self):
        prior_tensor = torch.Tensor(self.out_channels, self.kernel_size, self.kernel_size)
        half_fg = self.out_channels/2
        for i in range(self.out_channels):
            for h in range(3):
                for w in range(3):
                    if i < half_fg:
                        prior_tensor[i, h, w] = math.cos(math.pi*(h+0.5)*(i+1)/3)
                    else:
                        prior_tensor[i, h, w] = math.cos(math.pi*(w+0.5)*(i+1-half_fg)/3)
        self.register_buffer('weight_rbr_prior', prior_tensor)

    def weight_gen(self):
        weight_rbr_origin = torch.einsum('oihw,o->oihw', self.weight_rbr_origin, self.vector[0, :])
        weight_rbr_avg = torch.einsum('oihw,o->oihw', torch.einsum('oihw,hw->oihw', self.weight_rbr_avg_conv, self.weight_rbr_avg_avg), self.vector[1, :])
        weight_rbr_pfir = torch.einsum('oihw,o->oihw', torch.einsum('oihw,ohw->oihw', self.weight_rbr_pfir_conv, self.weight_rbr_prior), self.vector[2, :])
        weight_rbr_1x1_kxk_conv1 = None
        if hasattr(self, 'weight_rbr_1x1_kxk_idconv1'):
            weight_rbr_1x1_kxk_conv1 = (self.weight_rbr_1x1_kxk_idconv1 + self.id_tensor).squeeze()
        elif hasattr(self, 'weight_rbr_1x1_kxk_conv1'):
            weight_rbr_1x1_kxk_conv1 = self.weight_rbr_1x1_kxk_conv1.squeeze()
        else:
            raise NotImplementedError
        weight_rbr_1x1_kxk_conv2 = self.weight_rbr_1x1_kxk_conv2
        if self.groups > 1:
            g = self.groups
            t, ig = weight_rbr_1x1_kxk_conv1.size()
            o, tg, h, w = weight_rbr_1x1_kxk_conv2.size()
            weight_rbr_1x1_kxk_conv1 = weight_rbr_1x1_kxk_conv1.view(g, int(t/g), ig)
            weight_rbr_1x1_kxk_conv2 = weight_rbr_1x1_kxk_conv2.view(g, int(o/g), tg, h, w)
            weight_rbr_1x1_kxk = torch.einsum('gti,gothw->goihw', weight_rbr_1x1_kxk_conv1, weight_rbr_1x1_kxk_conv2).view(o, ig, h, w)
        else:
            weight_rbr_1x1_kxk = torch.einsum('ti,othw->oihw', weight_rbr_1x1_kxk_conv1, weight_rbr_1x1_kxk_conv2)
        weight_rbr_1x1_kxk = torch.einsum('oihw,o->oihw', weight_rbr_1x1_kxk, self.vector[3, :])
        weight_rbr_gconv = self.dwsc2full(self.weight_rbr_gconv_dw, self.weight_rbr_gconv_pw, self.in_channels)
        weight_rbr_gconv = torch.einsum('oihw,o->oihw', weight_rbr_gconv, self.vector[4, :])    
        weight = weight_rbr_origin + weight_rbr_avg + weight_rbr_1x1_kxk + weight_rbr_pfir + weight_rbr_gconv
        return weight

    def dwsc2full(self, weight_dw, weight_pw, groups):
        t, ig, h, w = weight_dw.size()
        o, _, _, _ = weight_pw.size()
        tg = int(t/groups)
        i = int(ig*groups)
        weight_dw = weight_dw.view(groups, tg, ig, h, w)
        weight_pw = weight_pw.squeeze().view(o, groups, tg)
        weight_dsc = torch.einsum('gtihw,ogt->ogihw', weight_dw, weight_pw)
        return weight_dsc.view(o, i, h, w)

    def forward(self, inputs):
        weight = self.weight_gen()
        out = F.conv2d(inputs, weight, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        return self.nonlinear(self.bn(out))

class RepConv_OREPA(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, padding=1, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False, nonlinear=nn.SiLU()):
        super(RepConv_OREPA, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = c1
        self.out_channels = c2
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        assert k == 3
        assert padding == 1
        padding_11 = padding - k // 2
        if nonlinear is None:
            self.nonlinearity = nn.Identity()
        else:
            self.nonlinearity = nonlinear
        if use_se:
            self.se = SEBlock(self.out_channels, internal_neurons=self.out_channels // 16)
        else:
            self.se = nn.Identity()
        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=k, stride=s,
                                      padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)
        else:
            self.rbr_identity = nn.BatchNorm2d(num_features=self.in_channels) if self.out_channels == self.in_channels and s == 1 else None
            self.rbr_dense = OREPA_3x3_RepConv(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=k, stride=s, padding=padding, groups=groups, dilation=1)
            self.rbr_1x1 = ConvBN(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1, stride=s, padding=padding_11, groups=groups, dilation=1)
            print('RepVGG Block, identity = ', self.rbr_identity)


    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        out1 = self.rbr_dense(inputs)
        out2 = self.rbr_1x1(inputs)
        out3 = id_out
        out = out1 + out2 + out3
        return self.nonlinearity(self.se(out))

    def get_custom_L2(self):
        K3 = self.rbr_dense.weight_gen()
        K1 = self.rbr_1x1.conv.weight
        t3 = (self.rbr_dense.bn.weight / ((self.rbr_dense.bn.running_var + self.rbr_dense.bn.eps).sqrt())).reshape(-1, 1, 1, 1).detach()
        t1 = (self.rbr_1x1.bn.weight / ((self.rbr_1x1.bn.running_var + self.rbr_1x1.bn.eps).sqrt())).reshape(-1, 1, 1, 1).detach()
        l2_loss_circle = (K3 ** 2).sum() - (K3[:, :, 1:2, 1:2] ** 2).sum()
        eq_kernel = K3[:, :, 1:2, 1:2] * t3 + K1 * t1
        l2_loss_eq_kernel = (eq_kernel ** 2 / (t3 ** 2 + t1 ** 2)).sum()
        return l2_loss_eq_kernel + l2_loss_circle

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1,1,1,1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if not isinstance(branch, nn.BatchNorm2d):
            if isinstance(branch, OREPA_3x3_RepConv):
                kernel = branch.weight_gen()
            elif isinstance(branch, ConvBN):
                kernel = branch.conv.weight
            else:
                raise NotImplementedError
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        print(f"RepConv_OREPA.switch_to_deploy")
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.in_channels, out_channels=self.rbr_dense.out_channels,
                                     kernel_size=self.rbr_dense.kernel_size, stride=self.rbr_dense.stride,
                                     padding=self.rbr_dense.padding, dilation=self.rbr_dense.dilation, groups=self.rbr_dense.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity') 


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        nn.init.normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        try:
            x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        except:
            x = (attn.half() @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    B, H, W, C = x.shape
    assert H % window_size == 0, 'feature map h and w can not divide by window size'
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SwinTransformerLayer(nn.Module):
    def __init__(self, dim, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.SiLU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def create_mask(self, H, W):
        img_mask = torch.zeros((1, H, W, 1))
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x):
        _, _, H_, W_ = x.shape
        Padding = False
        if min(H_, W_) < self.window_size or H_ % self.window_size!=0 or W_ % self.window_size!=0:
            Padding = True
            pad_r = (self.window_size - W_ % self.window_size) % self.window_size
            pad_b = (self.window_size - H_ % self.window_size) % self.window_size
            x = F.pad(x, (0, pad_r, 0, pad_b))
        B, C, H, W = x.shape
        L = H * W
        x = x.permute(0, 2, 3, 1).contiguous().view(B, L, C)
        if self.shift_size > 0:
            attn_mask = self.create_mask(H, W).to(x.device)
        else:
            attn_mask = None
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.permute(0, 2, 1).contiguous().view(-1, C, H, W)
        if Padding:
            x = x[:, :, :H_, :W_]
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, c1, c2, num_heads, num_layers, window_size=8):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.blocks = nn.Sequential(*[SwinTransformerLayer(dim=c2, num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2) for i in range(num_layers)])

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        x = self.blocks(x)
        return x


class STCSPA(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super(STCSPA, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        num_heads = c_ // 32
        self.m = SwinTransformerBlock(c_, c_, num_heads, n)

    def forward(self, x):
        y1 = self.m(self.cv1(x))
        y2 = self.cv2(x)
        return self.cv3(torch.cat((y1, y2), dim=1))


class STCSPB(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super(STCSPB, self).__init__()
        c_ = int(c2)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        num_heads = c_ // 32
        self.m = SwinTransformerBlock(c_, c_, num_heads, n)

    def forward(self, x):
        x1 = self.cv1(x)
        y1 = self.m(x1)
        y2 = self.cv2(x1)
        return self.cv3(torch.cat((y1, y2), dim=1))


class STCSPC(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super(STCSPC, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 1, 1)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        num_heads = c_ // 32
        self.m = SwinTransformerBlock(c_, c_, num_heads, n)

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(torch.cat((y1, y2), dim=1))

  
class WindowAttention_v2(nn.Module):

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.,
                 pretrained_window_size=[0, 0]):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True), nn.ReLU(inplace=True), nn.Linear(512, num_heads, bias=False))
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h, relative_coords_w])).permute(1, 2, 0).contiguous().unsqueeze(0)
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
        relative_coords_table *= 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)
        self.register_buffer("relative_coords_table", relative_coords_table)
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01))).exp()
        attn = attn * logit_scale
        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        try:
            x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        except:
            x = (attn.half() @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, ' \
               f'pretrained_window_size={self.pretrained_window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        flops = 0
        flops += N * self.dim * 3 * self.dim
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        flops += N * self.dim * self.dim
        return flops
    
class Mlp_v2(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition_v2(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse_v2(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SwinTransformerLayer_v2(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.SiLU, norm_layer=nn.LayerNorm, pretrained_window_size=0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention_v2(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            pretrained_window_size=(pretrained_window_size, pretrained_window_size))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_v2(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def create_mask(self, H, W):
        img_mask = torch.zeros((1, H, W, 1))
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x):
        _, _, H_, W_ = x.shape
        Padding = False
        if min(H_, W_) < self.window_size or H_ % self.window_size!=0 or W_ % self.window_size!=0:
            Padding = True
            pad_r = (self.window_size - W_ % self.window_size) % self.window_size
            pad_b = (self.window_size - H_ % self.window_size) % self.window_size
            x = F.pad(x, (0, pad_r, 0, pad_b))
        B, C, H, W = x.shape
        L = H * W
        x = x.permute(0, 2, 3, 1).contiguous().view(B, L, C)
        if self.shift_size > 0:
            attn_mask = self.create_mask(H, W).to(x.device)
        else:
            attn_mask = None
        shortcut = x
        x = x.view(B, H, W, C)
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        x_windows = window_partition_v2(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse_v2(attn_windows, self.window_size, H, W)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(self.norm1(x))
        x = x + self.drop_path(self.norm2(self.mlp(x)))
        x = x.permute(0, 2, 1).contiguous().view(-1, C, H, W)
        if Padding:
            x = x[:, :, :H_, :W_]
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        flops += self.dim * H * W
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        flops += self.dim * H * W
        return flops


class SwinTransformer2Block(nn.Module):
    def __init__(self, c1, c2, num_heads, num_layers, window_size=7):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.blocks = nn.Sequential(*[SwinTransformerLayer_v2(dim=c2, num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2) for i in range(num_layers)])

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        x = self.blocks(x)
        return x


class ST2CSPA(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super(ST2CSPA, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        num_heads = c_ // 32
        self.m = SwinTransformer2Block(c_, c_, num_heads, n)

    def forward(self, x):
        y1 = self.m(self.cv1(x))
        y2 = self.cv2(x)
        return self.cv3(torch.cat((y1, y2), dim=1))


class ST2CSPB(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super(ST2CSPB, self).__init__()
        c_ = int(c2)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        num_heads = c_ // 32
        self.m = SwinTransformer2Block(c_, c_, num_heads, n)

    def forward(self, x):
        x1 = self.cv1(x)
        y1 = self.m(x1)
        y2 = self.cv2(x1)
        return self.cv3(torch.cat((y1, y2), dim=1))


class ST2CSPC(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super(ST2CSPC, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 1, 1)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        num_heads = c_ // 32
        self.m = SwinTransformer2Block(c_, c_, num_heads, n)

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(torch.cat((y1, y2), dim=1))


#######################################################################################################
# END: common.py
#######################################################################################################


#######################################################################################################
# START: train.py
#######################################################################################################


def train(hyp, opt, device, tb_writer=None):
    logger.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    save_dir, epochs, batch_size, total_batch_size, weights, rank, freeze = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.global_rank, opt.freeze

    # Directories
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    last = wdir / 'last.pt'
    best = wdir / 'best.pt'
    results_file = save_dir / 'results.txt'

    # Save run settings
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)

    # Configure
    plots = not opt.evolve  # create plots
    cuda = device.type != 'cpu'
    init_seeds(2 + rank)
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
    is_coco = opt.data.endswith('coco.yaml')

    # Logging- Doing this before checking the dataset. Might update data_dict
    loggers = {'wandb': None}  # loggers dict
    if rank in [-1, 0]:
        opt.hyp = hyp  # add hyperparameters
        run_id = torch.load(weights, map_location=device).get('wandb_id') if weights.endswith('.pt') and os.path.isfile(weights) else None
        wandb_logger = WandbLogger(opt, Path(opt.save_dir).stem, run_id, data_dict)
        loggers['wandb'] = wandb_logger.wandb
        data_dict = wandb_logger.data_dict
        if wandb_logger.wandb:
            weights, epochs, hyp = opt.weights, opt.epochs, opt.hyp  # WandbLogger might update weights, epochs if resuming

    nc = 1 if opt.single_cls else int(data_dict['nc'])  # number of classes
    names = ['item'] if opt.single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check

    # Model
    pretrained = weights.endswith('.pt')
    if pretrained:
        with torch_distributed_zero_first(rank):
            attempt_download(weights)  # download if not found locally
        ckpt = torch.load(weights, map_location=device)  # load checkpoint
        model = Model(opt.cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
        exclude = ['anchor'] if (opt.cfg or hyp.get('anchors')) and not opt.resume else []  # exclude keys
        state_dict = ckpt['model'].float().state_dict()  # to FP32
        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(state_dict, strict=False)  # load
        logger.info('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weights))  # report
    else:
        model = Model(opt.cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
    with torch_distributed_zero_first(rank):
        check_dataset(data_dict)  # check
    train_path = data_dict['train']
    test_path = data_dict['val']

    # Freeze
    freeze = [f'model.{x}.' for x in (freeze if len(freeze) > 1 else range(freeze[0]))]  # parameter names to freeze (full or partial)
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print('freezing %s' % k)
            v.requires_grad = False

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / total_batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # scale weight_decay
    logger.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)  # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay
        if hasattr(v, 'im'):
            if hasattr(v.im, 'implicit'):           
                pg0.append(v.im.implicit)
            else:
                for iv in v.im:
                    pg0.append(iv.implicit)
        if hasattr(v, 'imc'):
            if hasattr(v.imc, 'implicit'):           
                pg0.append(v.imc.implicit)
            else:
                for iv in v.imc:
                    pg0.append(iv.implicit)
        if hasattr(v, 'imb'):
            if hasattr(v.imb, 'implicit'):           
                pg0.append(v.imb.implicit)
            else:
                for iv in v.imb:
                    pg0.append(iv.implicit)
        if hasattr(v, 'imo'):
            if hasattr(v.imo, 'implicit'):           
                pg0.append(v.imo.implicit)
            else:
                for iv in v.imo:
                    pg0.append(iv.implicit)
        if hasattr(v, 'ia'):
            if hasattr(v.ia, 'implicit'):           
                pg0.append(v.ia.implicit)
            else:
                for iv in v.ia:
                    pg0.append(iv.implicit)
        if hasattr(v, 'attn'):
            if hasattr(v.attn, 'logit_scale'):   
                pg0.append(v.attn.logit_scale)
            if hasattr(v.attn, 'q_bias'):   
                pg0.append(v.attn.q_bias)
            if hasattr(v.attn, 'v_bias'):  
                pg0.append(v.attn.v_bias)
            if hasattr(v.attn, 'relative_position_bias_table'):  
                pg0.append(v.attn.relative_position_bias_table)
        if hasattr(v, 'rbr_dense'):
            if hasattr(v.rbr_dense, 'weight_rbr_origin'):  
                pg0.append(v.rbr_dense.weight_rbr_origin)
            if hasattr(v.rbr_dense, 'weight_rbr_avg_conv'): 
                pg0.append(v.rbr_dense.weight_rbr_avg_conv)
            if hasattr(v.rbr_dense, 'weight_rbr_pfir_conv'):  
                pg0.append(v.rbr_dense.weight_rbr_pfir_conv)
            if hasattr(v.rbr_dense, 'weight_rbr_1x1_kxk_idconv1'): 
                pg0.append(v.rbr_dense.weight_rbr_1x1_kxk_idconv1)
            if hasattr(v.rbr_dense, 'weight_rbr_1x1_kxk_conv2'):   
                pg0.append(v.rbr_dense.weight_rbr_1x1_kxk_conv2)
            if hasattr(v.rbr_dense, 'weight_rbr_gconv_dw'):   
                pg0.append(v.rbr_dense.weight_rbr_gconv_dw)
            if hasattr(v.rbr_dense, 'weight_rbr_gconv_pw'):   
                pg0.append(v.rbr_dense.weight_rbr_gconv_pw)
            if hasattr(v.rbr_dense, 'vector'):   
                pg0.append(v.rbr_dense.vector)

    if opt.adam:
        optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
    if opt.linear_lr:
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    else:
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # plot_lr_scheduler(optimizer, scheduler, epochs)

    # EMA
    ema = ModelEMA(model) if rank in [-1, 0] else None

    # Resume
    start_epoch, best_fitness = 0, 0.0
    if pretrained:
        # Optimizer
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']

        # EMA
        if ema and ckpt.get('ema'):
            ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
            ema.updates = ckpt['updates']

        # Results
        if ckpt.get('training_results') is not None:
            results_file.write_text(ckpt['training_results'])  # write results.txt

        # Epochs
        start_epoch = ckpt['epoch'] + 1
        if opt.resume:
            assert start_epoch > 0, '%s training to %g epochs is finished, nothing to resume.' % (weights, epochs)
        if epochs < start_epoch:
            logger.info('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                        (weights, ckpt['epoch'], epochs))
            epochs += ckpt['epoch']  # finetune additional epochs

        del ckpt, state_dict

    # Image sizes
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples

    # DP mode
    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    if opt.sync_bn and cuda and rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        logger.info('Using SyncBatchNorm()')

    # Trainloader
    dataloader, dataset = create_dataloader(train_path, imgsz, batch_size, gs, opt,
                                            hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect, rank=rank,
                                            world_size=opt.world_size, workers=opt.workers,
                                            image_weights=opt.image_weights, quad=opt.quad, prefix=colorstr('train: '))
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
    nb = len(dataloader)  # number of batches
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, opt.data, nc - 1)

    # Process 0
    if rank in [-1, 0]:
        testloader = create_dataloader(test_path, imgsz_test, batch_size * 2, gs, opt,  # testloader
                                       hyp=hyp, cache=opt.cache_images and not opt.notest, rect=True, rank=-1,
                                       world_size=opt.world_size, workers=opt.workers,
                                       pad=0.5, prefix=colorstr('val: '))[0]

        if not opt.resume:
            labels = np.concatenate(dataset.labels, 0)
            c = torch.tensor(labels[:, 0])  # classes
            # cf = torch.bincount(c.long(), minlength=nc) + 1.  # frequency
            # model._initialize_biases(cf.to(device))
            if plots:
                #plot_labels(labels, names, save_dir, loggers)
                if tb_writer:
                    tb_writer.add_histogram('classes', c, 0)

            # Anchors
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
            model.half().float()  # pre-reduce anchor precision

    # DDP mode
    if cuda and rank != -1:
        model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank,
                    # nn.MultiheadAttention incompatibility with DDP https://github.com/pytorch/pytorch/issues/26698
                    find_unused_parameters=any(isinstance(layer, nn.MultiheadAttention) for layer in model.modules()))

    # Model parameters
    hyp['box'] *= 3. / nl  # scale to layers
    hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    model.names = names

    # Start training
    t0 = time.time()
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = amp.GradScaler(enabled=cuda)
    compute_loss_ota = ComputeLossOTA(model)  # init loss class
    compute_loss = ComputeLoss(model)  # init loss class
    logger.info(f'Image sizes {imgsz} train, {imgsz_test} test\n'
                f'Using {dataloader.num_workers} dataloader workers\n'
                f'Logging results to {save_dir}\n'
                f'Starting training for {epochs} epochs...')
    torch.save(model, wdir / 'init.pt')
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()

        # Update image weights (optional)
        if opt.image_weights:
            # Generate indices
            if rank in [-1, 0]:
                cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
                iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
                dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx
            # Broadcast if DDP
            if rank != -1:
                indices = (torch.tensor(dataset.indices) if rank == 0 else torch.zeros(dataset.n)).int()
                dist.broadcast(indices, 0)
                if rank != 0:
                    dataset.indices = indices.cpu().numpy()

        # Update mosaic border
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        mloss = torch.zeros(4, device=device)  # mean losses
        if rank != -1:
            dataloader.sampler.set_epoch(epoch)
        pbar = enumerate(dataloader)
        logger.info(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'total', 'labels', 'img_size'))
        if rank in [-1, 0]:
            pbar = tqdm(pbar, total=nb)  # progress bar
        optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward
            with amp.autocast(enabled=cuda):
                pred = model(imgs)  # forward
                if 'loss_ota' not in hyp or hyp['loss_ota'] == 1:
                    loss, loss_items = compute_loss_ota(pred, targets.to(device), imgs)  # loss scaled by batch_size
                else:
                    loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
                if rank != -1:
                    loss *= opt.world_size  # gradient averaged between devices in DDP mode
                if opt.quad:
                    loss *= 4.

            # Backward
            scaler.scale(loss).backward()

            # Optimize
            if ni % accumulate == 0:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)

            # Print
            if rank in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                s = ('%10s' * 2 + '%10.4g' * 6) % (
                    '%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
                pbar.set_description(s)

                # Plot
                if plots and ni < 10:
                    f = save_dir / f'train_batch{ni}.jpg'  # filename
                    Thread(target=plot_images, args=(imgs, targets, paths, f), daemon=True).start()
                    # if tb_writer:
                    #     tb_writer.add_image(f, result, dataformats='HWC', global_step=epoch)
                    #     tb_writer.add_graph(torch.jit.trace(model, imgs, strict=False), [])  # add model graph
                elif plots and ni == 10 and wandb_logger.wandb:
                    wandb_logger.log({"Mosaics": [wandb_logger.wandb.Image(str(x), caption=x.name) for x in
                                                  save_dir.glob('train*.jpg') if x.exists()]})

            # end batch ------------------------------------------------------------------------------------------------
        # end epoch ----------------------------------------------------------------------------------------------------

        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for tensorboard
        scheduler.step()

        # DDP process 0 or single-GPU
        if rank in [-1, 0]:
            # mAP
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])
            final_epoch = epoch + 1 == epochs
            if not opt.notest or final_epoch:  # Calculate mAP
                wandb_logger.current_epoch = epoch + 1
                results, maps, times = test.test(data_dict,
                                                 batch_size=batch_size * 2,
                                                 imgsz=imgsz_test,
                                                 model=ema.ema,
                                                 single_cls=opt.single_cls,
                                                 dataloader=testloader,
                                                 save_dir=save_dir,
                                                 verbose=nc < 50 and final_epoch,
                                                 plots=plots and final_epoch,
                                                 wandb_logger=wandb_logger,
                                                 compute_loss=compute_loss,
                                                 is_coco=is_coco,
                                                 v5_metric=opt.v5_metric)

            # Write
            with open(results_file, 'a') as f:
                f.write(s + '%10.4g' * 7 % results + '\n')  # append metrics, val_loss
            if len(opt.name) and opt.bucket:
                os.system('gsutil cp %s gs://%s/results/results%s.txt' % (results_file, opt.bucket, opt.name))

            # Log
            tags = ['train/box_loss', 'train/obj_loss', 'train/cls_loss',  # train loss
                    'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
                    'val/box_loss', 'val/obj_loss', 'val/cls_loss',  # val loss
                    'x/lr0', 'x/lr1', 'x/lr2']  # params
            for x, tag in zip(list(mloss[:-1]) + list(results) + lr, tags):
                if tb_writer:
                    tb_writer.add_scalar(tag, x, epoch)  # tensorboard
                if wandb_logger.wandb:
                    wandb_logger.log({tag: x})  # W&B

            # Update best mAP
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            if fi > best_fitness:
                best_fitness = fi
            wandb_logger.end_epoch(best_result=best_fitness == fi)

            # Save model
            if (not opt.nosave) or (final_epoch and not opt.evolve):  # if save
                ckpt = {'epoch': epoch,
                        'best_fitness': best_fitness,
                        'training_results': results_file.read_text(),
                        'model': deepcopy(model.module if is_parallel(model) else model).half(),
                        'ema': deepcopy(ema.ema).half(),
                        'updates': ema.updates,
                        'optimizer': optimizer.state_dict(),
                        'wandb_id': wandb_logger.wandb_run.id if wandb_logger.wandb else None}

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if (best_fitness == fi) and (epoch >= 200):
                    torch.save(ckpt, wdir / 'best_{:03d}.pt'.format(epoch))
                if epoch == 0:
                    torch.save(ckpt, wdir / 'epoch_{:03d}.pt'.format(epoch))
                elif ((epoch+1) % 25) == 0:
                    torch.save(ckpt, wdir / 'epoch_{:03d}.pt'.format(epoch))
                elif epoch >= (epochs-5):
                    torch.save(ckpt, wdir / 'epoch_{:03d}.pt'.format(epoch))
                if wandb_logger.wandb:
                    if ((epoch + 1) % opt.save_period == 0 and not final_epoch) and opt.save_period != -1:
                        wandb_logger.log_model(
                            last.parent, opt, epoch, fi, best_model=best_fitness == fi)
                del ckpt

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training
    if rank in [-1, 0]:
        # Plots
        if plots:
            plot_results(save_dir=save_dir)  # save as results.png
            if wandb_logger.wandb:
                files = ['results.png', 'confusion_matrix.png', *[f'{x}_curve.png' for x in ('F1', 'PR', 'P', 'R')]]
                wandb_logger.log({"Results": [wandb_logger.wandb.Image(str(save_dir / f), caption=f) for f in files
                                              if (save_dir / f).exists()]})
        # Test best.pt
        logger.info('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
        if opt.data.endswith('coco.yaml') and nc == 80:  # if COCO
            for m in (last, best) if best.exists() else (last):  # speed, mAP tests
                results, _, _ = test.test(opt.data,
                                          batch_size=batch_size * 2,
                                          imgsz=imgsz_test,
                                          conf_thres=0.001,
                                          iou_thres=0.7,
                                          model=attempt_load(m, device).half(),
                                          single_cls=opt.single_cls,
                                          dataloader=testloader,
                                          save_dir=save_dir,
                                          save_json=True,
                                          plots=False,
                                          is_coco=is_coco,
                                          v5_metric=opt.v5_metric)

        # Strip optimizers
        final = best if best.exists() else last  # final model
        for f in last, best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
        if opt.bucket:
            os.system(f'gsutil cp {final} gs://{opt.bucket}/weights')  # upload
        if wandb_logger.wandb and not opt.evolve:  # Log the stripped model
            wandb_logger.wandb.log_artifact(str(final), type='model',
                                            name='run_' + wandb_logger.wandb_run.id + '_model',
                                            aliases=['last', 'best', 'stripped'])
        wandb_logger.finish_run()
    else:
        dist.destroy_process_group()
    torch.cuda.empty_cache()
    return results


#######################################################################################################
# END: train.py
#######################################################################################################



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolo7.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.p5.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--upload_dataset', action='store_true', help='Upload dataset as W&B artifact table')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval for W&B')
    parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')
    parser.add_argument('--artifact_alias', type=str, default="latest", help='version of dataset artifact to be used')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone of yolov7=50, first3=0 1 2')
    parser.add_argument('--v5-metric', action='store_true', help='assume maximum recall as 1.0 in AP calculation')
    opt = parser.parse_args()

    # Set DDP variables
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    set_logging(opt.global_rank)
    #if opt.global_rank in [-1, 0]:
    #    check_git_status()
    #    check_requirements()

    # Resume
    wandb_run = check_wandb_resume(opt)
    if opt.resume and not wandb_run:  # resume an interrupted run
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        apriori = opt.global_rank, opt.local_rank
        with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
            opt = argparse.Namespace(**yaml.load(f, Loader=yaml.SafeLoader))  # replace
        opt.cfg, opt.weights, opt.resume, opt.batch_size, opt.global_rank, opt.local_rank = '', ckpt, True, opt.total_batch_size, *apriori  # reinstate
        logger.info('Resuming training from %s' % ckpt)
    else:
        # opt.hyp = opt.hyp or ('hyp.finetune.yaml' if opt.weights else 'hyp.scratch.yaml')
        opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
        opt.name = 'evolve' if opt.evolve else opt.name
        opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve)  # increment run

    # DDP mode
    opt.total_batch_size = opt.batch_size
    device = select_device(opt.device, batch_size=opt.batch_size)
    if opt.local_rank != -1:
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        device = torch.device('cuda', opt.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
        assert opt.batch_size % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
        opt.batch_size = opt.total_batch_size // opt.world_size

    # Hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps

    # Train
    logger.info(opt)
    if not opt.evolve:
        tb_writer = None  # init loggers
        if opt.global_rank in [-1, 0]:
            prefix = colorstr('tensorboard: ')
            logger.info(f"{prefix}Start with 'tensorboard --logdir {opt.project}', view at http://localhost:6006/")
            tb_writer = TensorBoardSummaryWriter(opt.save_dir)  # Tensorboard
        train(hyp, opt, device, tb_writer)

    # Evolve hyperparameters (optional)
    else:
        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
        meta = {'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
                'lrf': (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
                'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
                'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
                'warmup_epochs': (1, 0.0, 5.0),  # warmup epochs (fractions ok)
                'warmup_momentum': (1, 0.0, 0.95),  # warmup initial momentum
                'warmup_bias_lr': (1, 0.0, 0.2),  # warmup initial bias lr
                'box': (1, 0.02, 0.2),  # box loss gain
                'cls': (1, 0.2, 4.0),  # cls loss gain
                'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
                'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
                'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
                'iou_t': (0, 0.1, 0.7),  # IoU training threshold
                'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
                'anchors': (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
                'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
                'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
                'hsv_s': (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
                'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
                'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
                'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
                'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
                'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
                'perspective': (0, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
                'flipud': (1, 0.0, 1.0),  # image flip up-down (probability)
                'fliplr': (0, 0.0, 1.0),  # image flip left-right (probability)
                'mosaic': (1, 0.0, 1.0),  # image mixup (probability)
                'mixup': (1, 0.0, 1.0),   # image mixup (probability)
                'copy_paste': (1, 0.0, 1.0),  # segment copy-paste (probability)
                'paste_in': (1, 0.0, 1.0)}    # segment copy-paste (probability)
        
        with open(opt.hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
            if 'anchors' not in hyp:  # anchors commented in hyp.yaml
                hyp['anchors'] = 3
                
        assert opt.local_rank == -1, 'DDP mode not implemented for --evolve'
        opt.notest, opt.nosave = True, True  # only test/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        yaml_file = Path(opt.save_dir) / 'hyp_evolved.yaml'  # save best result here
        if opt.bucket:
            os.system('gsutil cp gs://%s/evolve.txt .' % opt.bucket)  # download evolve.txt if exists

        for _ in range(300):  # generations to evolve
            if Path('evolve.txt').exists():  # if evolve.txt exists: select best hyps and mutate
                # Select parent(s)
                parent = 'single'  # parent selection method: 'single' or 'weighted'
                x = np.loadtxt('evolve.txt', ndmin=2)
                n = min(5, len(x))  # number of previous results to consider
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                w = fitness(x) - fitness(x).min()  # weights
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # Mutate
                mp, s = 0.8, 0.2  # mutation probability, sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([x[0] for x in meta.values()])  # gains 0-1
                ng = len(meta)
                v = np.ones(ng)
                while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = float(x[i + 7] * v[i])  # mutate

            # Constrain to limits
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  # lower limit
                hyp[k] = min(hyp[k], v[2])  # upper limit
                hyp[k] = round(hyp[k], 5)  # significant digits

            # Train mutation
            results = train(hyp.copy(), opt, device)

            # Write mutation results
            print_mutation(hyp.copy(), results, yaml_file, opt.bucket)

        # Plot results
        plot_evolution(yaml_file)
        print(f'Hyperparameter evolution complete. Best results saved as: {yaml_file}\n'
              f'Command to train a new model with these hyperparameters: $ python train.py --hyp {yaml_file}')