import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np

from fcos_core.modeling.dsc.irnn import irnn


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=stride, padding=0, bias=False)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


class Spacial_IRNN(nn.Module):
    def __init__(self, in_channels, alpha=1.0):
        super(Spacial_IRNN, self).__init__()
        self.left_weight = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, groups=in_channels, padding=0)
        self.right_weight = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, groups=in_channels, padding=0)
        self.up_weight = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, groups=in_channels, padding=0)
        self.down_weight = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, groups=in_channels, padding=0)
        self.left_weight.weight = nn.Parameter(torch.tensor([[[[alpha]]]] * in_channels))
        self.right_weight.weight = nn.Parameter(torch.tensor([[[[alpha]]]] * in_channels))
        self.up_weight.weight = nn.Parameter(torch.tensor([[[[alpha]]]] * in_channels))
        self.down_weight.weight = nn.Parameter(torch.tensor([[[[alpha]]]] * in_channels))

    def forward(self, input):
        return irnn()(input, self.up_weight.weight, self.right_weight.weight, self.down_weight.weight,
                      self.left_weight.weight, self.up_weight.bias, self.right_weight.bias, self.down_weight.bias,
                      self.left_weight.bias)


class Attention(nn.Module):
    def __init__(self, in_channels):
        super(Attention, self).__init__()
        self.out_channels = int(in_channels / 2)
        self.convD1 = nn.Conv2d(in_channels, self.out_channels, kernel_size=3, padding=1, stride=1)
        self.relu1 = nn.ReLU()
        self.convD2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1, stride=1)
        self.relu2 = nn.ReLU()
        self.convD3 = nn.Conv2d(self.out_channels, 1, kernel_size=1, padding=0, stride=1)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        out = self.convD1(x)
        out = self.relu1(out)
        out = self.convD2(out)
        out = self.relu2(out)
        out = self.convD3(out)
        out = self.sigmod(out)
        return out


class DSC_Module(nn.Module):
    def __init__(self, in_channels, out_channels, attention=1, alpha=1.0):
        super(DSC_Module, self).__init__()
        self.out_channels = out_channels
        self.irnn1 = Spacial_IRNN(self.out_channels, alpha)
        self.irnn2 = Spacial_IRNN(self.out_channels, alpha)
        self.conv_in = conv1x1(in_channels, in_channels)
        self.convD2 = conv1x1(in_channels * 4, in_channels)
        self.convD3 = conv1x1(in_channels * 4, in_channels)
        self.relu2 = nn.ReLU(True)
        self.attention = attention
        if self.attention:
            self.attention_layer = Attention(in_channels)
        self.relu4 = nn.ReLU(True)


    def forward(self, x):


        #先使用irnn
        out = self.conv_in(x)
        top_up, top_right, top_down, top_left = self.irnn1(out)

        out = torch.cat([top_up, top_right, top_down, top_left], dim=1)
        out = self.convD2(out)
        out = self.relu2(out)

        if self.attention:
            weight = self.attention_layer(out)

        out=x.mul(weight)
        out=self.relu4(out)


        return out


class LayerConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding, relu):
        super(LayerConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.relu is not None:
            x = self.relu(x)

        return x


class Predict(nn.Module):
    def __init__(self, in_planes=32, out_planes=1, kernel_size=1):
        super(Predict, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size)

    def forward(self, x):
        y = self.conv(x)

        return y


# initialization
#         for modules in [self.cls_tower, self.bbox_tower,
#                         self.cls_logits, self.bbox_pred,
#                         self.centerness]:
#             for l in modules.modules():
#                 if isinstance(l, nn.Conv2d):
#                     torch.nn.init.normal_(l.weight, std=0.01)
#                     # torch.nn.init.constant_(l.bias, 0)


if __name__ == '__main__':
    model = DSC_Module(256, 256).double()
    model = model.cuda()

    feature1 = np.random.rand(5, 256, 25, 25)
    feature1 = torch.from_numpy(feature1)
    feature2 = np.random.rand(5, 256, 25, 25)
    feature2 = torch.from_numpy(feature2)
    feature3 = np.random.rand(5, 256, 25, 25)
    feature3 = torch.from_numpy(feature3)

    result = model(feature3.cuda())

    print(result.size())