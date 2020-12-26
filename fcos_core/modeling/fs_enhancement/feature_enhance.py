import torch
from torch import nn
from fcos_core.modeling.dsc.dscFeature import dscFeature
from fcos_core.modeling.fs_enhancement.fs_fusion import generateSceneFeatureMap as getScene
# from fcos_core.modeling.fs_enhancement.fs import generateSceneFeatureMap as getScene
from fcos_core.modeling.fs_enhancement.asff import ASFF
from fcos_core.modeling.fs_enhancement.cbam import BasicBlock


class featureInhanceHead(nn.Module):
    def __init__(self):
        super(featureInhanceHead, self).__init__()

        self.enhance=getScene()

        self.dsc=dscFeature()
        self.catDscEnhance=nn.Conv2d(512,256,kernel_size=1,stride=1,padding=0)
        self.BackRelu=nn.ReLU(True)

        self.asff=ASFF()

        self.cbam=BasicBlock(256,256)

        # for modules in [self.enhance, self.dsc,
        #                 self.catDscEnhance, self.asff]:
        #     for l in modules.modules():
        #         if isinstance(l, nn.Conv2d):
        #             torch.nn.init.normal_(l.weight, std=0.01)
        #             torch.nn.init.constant_(l.bias, 0)

    def forward(self,features):
        feature_DSC=self.dsc(features)
        feature_RFB=self.enhance(features)

        featureList=[]

        for x,y,z in zip(feature_DSC,feature_RFB,features):
            # temp=torch.cat((x,y),1)
            # temp=self.catDscEnhance(temp)
            # temp=self.BackRelu(temp)


            # temp=self.asff(x,y,z)
            # temp=self.cbam(temp)
            featureList.append(y)

        features=tuple(featureList)
        return features

def build_feature_enchance():

    model=featureInhanceHead()
    return model