import torch
import torch.nn as nn
import math
import numpy as np

class ChannelAttention(nn.Module):

    def __init__(self,in_planes,ratio=16):
        super(ChannelAttention,self).__init__()
        self.avg_pool=nn.AdaptiveAvgPool2d(1)
        self.max_pool=nn.AdaptiveMaxPool2d(1)

        self.fc1=nn.Conv2d(in_planes,in_planes//16,1,bias=False)
        self.relu1=nn.ReLU()
        self.fc2=nn.Conv2d(in_planes//16,in_planes,1,bias=False)

        self.sigmoid=nn.Sigmoid()

    def forward(self,x):
        avg_out=self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out=self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out=avg_out+max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self,kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3,7), 'kernel size must be 3 or 7'
        padding=3 if kernel_size==7 else 1

        self.conv1cbam=nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid=nn.Sigmoid()
    def forward(self,x):
        avg_out=torch.mean(x,dim=1,keepdim=True)
        max_out,_=torch.max(x,dim=1,keepdim=True)
        x=torch.cat([avg_out,max_out],dim=1)
        x=self.conv1cbam(x)
        return self.sigmoid(x)

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1cbam = conv3x3(inplanes, planes, stride)
        self.bn1cbam = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2cbam = conv3x3(planes, planes)
        self.bn2cbam = nn.BatchNorm2d(planes)

        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1cbam(x)
        out = self.bn1cbam(out)
        out = self.relu(out)

        out = self.conv2cbam(out)
        out = self.bn2cbam(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


if __name__=='__main__':
    model=BasicBlock(256,256).double()

    featureDim=[100,50,25,13,7]
    featureList=[]
    for x in featureDim:
        temp=np.random.rand(5,256,x,x)
        featureList.append(torch.from_numpy(temp))

    result=[]
    for x in featureList:
        result.append(model(x))

    for x in result:
        print(x.size())


