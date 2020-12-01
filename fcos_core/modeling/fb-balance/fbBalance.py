import torch
import torch.nn as nn
import numpy as np
from fcos_core.modeling.fs_enhancement.rfb import BasicConv
from fcos_core.modeling.fs_enhancement.rfb import BasicRFB as RFB

class fbBalance(nn.Module):
    def __init__(self,in_planes=256,out_planes=256):
        super(fbBalance,self).__init__()
        self.maskRF=RFB(in_planes,out_planes)


    def forward(self,x):


        return x

