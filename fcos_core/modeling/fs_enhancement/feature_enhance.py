import torch
from torch import nn
from fcos_core.modeling.dsc.dscFeature import dscFeature
from fcos_core.modeling.fs_enhancement.fs_fusion import generateSceneFeatureMap as getScene
# from fcos_core.modeling.fs_enhancement.fs import generateSceneFeatureMap as getScene
from fcos_core.modeling.fs_enhancement.asff import ASFF
from fcos_core.modeling.fs_enhancement.cbam import BasicBlock

from fcos_core.modeling.fs_enhancement.rfb import BasicRFB as RFB
from fcos_core.modeling.fs_enhancement.rfb import BasicConv





class featureInhanceHead(nn.Module):
    def __init__(self):
        super(featureInhanceHead, self).__init__()

        self.enhance=getScene()

        self.dsc=dscFeature()

        #self.combine=convCombine()
        self.combine = asffCombine()


        # self.asff1= ASFF()
        # self.asff2 = ASFF()
        # self.asff3 = ASFF()
        # self.asff4 = ASFF()
        # self.asff5 = ASFF()
        #
        # self.catDscEnhance = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        # self.catTwo1=nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        # self.catTwo2=nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        # self.catTwo3 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        # self.catTwo4 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        # self.catTwo5 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)


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
        # feature_DSC = self.dsc(feature_RFB)
        x=self.combine(feature_DSC,feature_RFB,features)



        # featureList=[]
        # featureList.append(self.asff1(feature_DSC[0],features[0],feature_RFB[0],))
        # featureList.append(self.asff2(feature_DSC[1],features[1],feature_RFB[1],))
        # featureList.append(self.asff3(feature_DSC[2],features[2],feature_RFB[2],))
        # featureList.append(self.asff4(feature_DSC[3],features[3],feature_RFB[3],))
        # featureList.append(self.asff5(feature_DSC[4],features[4],feature_RFB[4],))


        # for x,y,z in zip(feature_DSC,feature_RFB,features):
        #     temp=torch.cat((x,y),1)
        #     temp=self.catDscEnhance(temp)
        #     temp=self.BackRelu(temp)
        #     # temp=self.asff1(z,x,y)
        #     # temp=self.cbam(temp)
        #     featureList.append(temp)


        return x


class convCombine(nn.Module):
    def __init__(self,in_planes=512,out_planes=256):
        super(convCombine,self).__init__()

        self.catTwo1 = BasicConv(in_planes, out_planes, kernel_size=1, stride=1, padding=0)
        self.catTwo2 = BasicConv(in_planes, out_planes, kernel_size=1, stride=1, padding=0)
        self.catTwo3 = BasicConv(in_planes, out_planes, kernel_size=1, stride=1, padding=0)
        self.catTwo4 = BasicConv(in_planes, out_planes, kernel_size=1, stride=1, padding=0)
        self.catTwo5 = BasicConv(in_planes, out_planes, kernel_size=1, stride=1, padding=0)

    def forward(self,feature1,feature2):
        featureList=[]

        temp1=torch.cat((feature1[0],feature2[0]),1)
        temp1=self.catTwo1(temp1)
        featureList.append(temp1)
        del temp1

        temp1 = torch.cat((feature1[1], feature2[1]), 1)
        temp1 = self.catTwo2(temp1)
        featureList.append(temp1)
        del temp1

        temp1 = torch.cat((feature1[2], feature2[2]), 1)
        temp1 = self.catTwo3(temp1)
        featureList.append(temp1)
        del temp1

        temp1 = torch.cat((feature1[3], feature2[3]), 1)
        temp1 = self.catTwo4(temp1)
        featureList.append(temp1)
        del temp1

        temp1 = torch.cat((feature1[4], feature2[4]), 1)
        temp1 = self.catTwo5(temp1)
        featureList.append(temp1)
        del temp1

        features=tuple(featureList)
        del featureList

        return features


class asffCombine(nn.Module):
    def __init__(self,in_planes=512,out_planes=256):
        super(asffCombine,self).__init__()

        self.asff1 = ASFF()
        self.asff2 = ASFF()
        self.asff3 = ASFF()
        self.asff4 = ASFF()
        self.asff5 = ASFF()

    def forward(self,feature1,feature2,x):
        featureList=[]

        featureList.append(self.asff1(feature1[0], feature2[0], x[0], ))
        featureList.append(self.asff2(feature1[1], feature2[1], x[1], ))
        featureList.append(self.asff3(feature1[2], feature2[2], x[2], ))
        featureList.append(self.asff4(feature1[3], feature2[3], x[3], ))
        featureList.append(self.asff5(feature1[4], feature2[4], x[4], ))

        features = tuple(featureList)

        return features





def build_feature_enchance():

    model=featureInhanceHead()
    return model