import torch
import torch.nn as nn
import numpy as np
from fcos_core.modeling.fs_enhancement.rfb import BasicConv
from fcos_core.modeling.fs_enhancement.rfb import BasicRFB as RFB
import torch.nn.init as init

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

        for m in self.children():
            # print(m)
            # print('=' * 80)
            # print('\n\n')
            self.weights_init(m)



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

        # featureList=[]
        # featureList.append(P3RF)
        # featureList.append(P4RF)
        # featureList.append(P5RF)
        # featureList.append(x[3])
        # featureList.append(x[4])
        #
        # featureList=tuple(featureList)
        # return  featureList


        SceneMap=torch.cat((P5RF,self.downSample(P4RF),self.downSample(self.downSample(P3RF))),dim=1)
        result=self.fusion(SceneMap,x)
        return result

    def weights_init(self,m):
        # print(m)
        # print('='*80)
        # print('\n\n')
        for key in m.state_dict():
            # print(key)
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal_(m.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    m.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                m.state_dict()[key][...] = 0





class fsFusion(nn.Module):
    def __init__(self,in_planes=256,out_planes=256):
        super(fsFusion, self).__init__()
        self.scene_P7 = BasicConv(in_planes * 3, out_planes, kernel_size=1)
        self.scene_P6 = BasicConv(in_planes * 3, out_planes, kernel_size=1)
        self.scene_P5 = BasicConv(in_planes * 3, out_planes, kernel_size=1)
        self.scene_P4 = BasicConv(in_planes * 3, out_planes, kernel_size=1)
        self.scene_P3 = BasicConv(in_planes * 3, out_planes, kernel_size=1)

        self.relation_P7 = RFB(in_planes*2, out_planes)
        self.relation_P6 = RFB(in_planes*2, out_planes)
        self.relation_P5 = RFB(in_planes*2, out_planes)
        self.relation_P4 = RFB(in_planes*2, out_planes)
        self.relation_P3 = RFB(in_planes*2, out_planes)


        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.downSample = nn.MaxPool2d(2, stride=2,padding=1)


    def forward(self,scene,x):
        p3_scene=self.scene_P3(self.upsample_4(scene))
        p4_scene=self.scene_P4(self.upsample_2(scene))
        p5_scene=self.scene_P5(scene)
        p6_scene=self.scene_P6(self.downSample(scene))
        p7_scene=self.scene_P7(self.downSample(self.downSample(scene)))

        p3 = x[0]
        p4 = x[1]
        p5 = x[2]
        p6 = x[3]
        p7 = x[4]


        featureList = []


        temp=torch.cat((p3_scene,p3),dim=1)
        p3R=self.relation_P3(temp)
        del temp
        p3R=torch.sigmoid(p3R)
        featureList.append(p3R*p3)
        del p3R,p3_scene

        temp = torch.cat((p4_scene, p4), dim=1)
        p4R = self.relation_P4(temp)
        del temp
        p4R =torch.sigmoid(p4R)
        featureList.append(p4R*p4)
        del p4R,p4_scene

        temp = torch.cat((p5_scene, p5), dim=1)
        p5R = self.relation_P5(temp)
        del temp
        p5R = torch.sigmoid(p5R)
        featureList.append(p5R*p5)
        del p5R,p5_scene

        temp = torch.cat((p6_scene, p6), dim=1)
        p6R = self.relation_P6(temp)
        del temp
        p6R=torch.sigmoid(p6R)
        featureList.append(p6R*p6)
        del p6R,p6_scene

        temp = torch.cat((p7_scene, p7), dim=1)
        p7R = self.relation_P7(temp)
        del temp
        p7R=torch.sigmoid(p7R)
        featureList.append(p7R*p7)
        del p7R,p7_scene


        featureList=tuple(featureList)

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
    # for param in net.named_parameters():
    #     print(param)
    # for k, v in net.items():
    #     print(k, v.size(), sep="    ")

    for item in y:
        print(item.size())


