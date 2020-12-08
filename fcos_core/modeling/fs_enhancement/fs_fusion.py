import torch
import torch.nn as nn
import numpy as np
from fcos_core.modeling.fs_enhancement.rfb import BasicConv
from fcos_core.modeling.fs_enhancement.rfb import BasicRFB as RFB

class generateSceneFeatureMap(nn.Module):
    def __init__(self,in_planes=256,out_planes=256):
        super(generateSceneFeatureMap, self).__init__()
        self.p3_rf=RFB(in_planes,out_planes)
        self.p4_rf=RFB(in_planes*2,out_planes)
        self.p5_rf=RFB(in_planes*2,out_planes)

        self.upsample_2=nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        self.upsample_4=nn.Upsample(scale_factor=4,mode='bilinear',align_corners=True)
        self.downSample=nn.MaxPool2d(2,stride=2)
        self.fusion=fsFusion()

        # for m in self.modules():
        #     if isinstance(m, (nn.Conv2d, nn.Linear)):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_in')




    def forward(self,x):
        p3=x[0]
        p4=x[1]
        p5=x[2]

        P3RF = self.p3_rf(p3)

        p3=self.downSample(p3)
        P4_3=torch.cat((p4,p3),dim=1)
        P4RF = self.p4_rf(P4_3)

        p4=self.downSample(p4)
        P5_4=torch.cat((p5,p4),dim=1)
        P5RF=self.p5_rf(P5_4)

        featureList=[]
        featureList.append(P3RF)
        featureList.append(P4RF)
        featureList.append(P5RF)
        featureList.append(x[3])
        featureList.append(x[4])

        featureList=tuple(featureList)

        return  featureList


        # SceneMap=torch.cat((P5RF,self.downSample(P4RF),self.downSample(self.downSample(P3RF))),dim=1)
        # result=self.fusion(SceneMap)
        # return result






class fsFusion(nn.Module):
    def __init__(self,in_planes=256,out_planes=256):
        super(fsFusion, self).__init__()
        self.scene_P7=BasicConv(in_planes*3,out_planes,kernel_size=1)
        self.scene_P6=BasicConv(in_planes*3,out_planes,kernel_size=1)
        self.scene_P5 =BasicConv(in_planes*3, out_planes, kernel_size=1)
        self.scene_P4 =BasicConv(in_planes*3, out_planes, kernel_size=1)
        self.scene_P3 =BasicConv(in_planes*3, out_planes, kernel_size=1)

        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.downSample = nn.MaxPool2d(2, stride=2,padding=1)


    def forward(self,scene):
        p3_scene=self.scene_P3(self.upsample_4(scene))
        p4_scene=self.scene_P4(self.upsample_2(scene))
        p5_scene=self.scene_P5(scene)
        p6_scene=self.scene_P6(self.downSample(scene))
        p7_scene=self.scene_P7(self.downSample(self.downSample(scene)))

        featureList = []
        featureList.append(p3_scene)
        featureList.append(p4_scene)
        featureList.append(p5_scene)
        featureList.append(p6_scene)
        featureList.append(p7_scene)
        featureList = tuple(featureList)

        return featureList



if __name__=='__main__':
    features=[]
    featureSize=[100,50,25,13,7]
    for i in featureSize:
        feature=np.random.rand(5,256,i,i)
        feature=torch.from_numpy(feature)
        features.append(feature)
    features=tuple(features)

    net=generateSceneFeatureMap().double()
    y=net(features)
    for item in y:
        print(item.size())


