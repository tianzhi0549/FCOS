from fcos_core.modeling.dsc.DSC import DSC_Module
from torch import nn
import torch

class dscFeature(nn.Module):
    def __init__(self):
        super(dscFeature, self).__init__()
        self.dsc = DSC_Module(256, 256)
        self.catconv=nn.Conv2d(512,256,kernel_size=1,stride=1,padding=0)
        self.DDrelu=nn.ReLU(True)


    def forward(self,features):

        featureList=[]

        for feature_layer in features:
            layerDsc=self.dsc(feature_layer)
            Ctemp=torch.cat((layerDsc,feature_layer),1)
            layerDsc=self.catconv(Ctemp)

            layerDsc=self.DDrelu(layerDsc)
            featureList.append(layerDsc)

        featureList = tuple(featureList)
        return featureList
