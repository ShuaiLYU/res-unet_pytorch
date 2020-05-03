import torch
import torch.nn as nn
import torch.nn.functional as F
from unet_blocks import  *
from my_resnet import  resnet18,resnet50



class UNet(nn.Module):
    def __init__(self,n_classes,base_channels=64,level=5,padding=0,norm_layer=None,bilinear=True):
        super(UNet, self).__init__()
        self.level=level
        self.base_channels=base_channels
        self.norm_layer=norm_layer
        self.padding=padding
        self.bilinear=bilinear
        self.encoder=self.build_encoder()
        self.decoder=self.build_decoder()
        self.outBlock=nn.Conv2d(base_channels,n_classes,1,1)
    def build_encoder(self):
        return Unet_Encoder(in_channels=3, base_channels=self.base_channels, level=self.level, padding=self.padding)
    def build_decoder(self):
        decoder=nn.ModuleList()
        for i in range(self.level-1): #有 self.level-1 个上采样块
            in_channels= self.base_channels*(2**(self.level-i-1))
            out_channels= self.base_channels*(2**(self.level-i-2))
            decoder.append(UNetUpBlock(in_channels,out_channels,
                                       padding=self.padding,norm_layer= self.norm_layer,bilinear=self.bilinear))
        return  decoder

    def forward(self,x):
        features=self.encoder(x)
        # for feat in features:
        #     print(feat.shape)
        assert len(features)==self.level
        x=features[-1]
        for i,up_block in enumerate(self.decoder):
            x=up_block(x,features[-2-i])
            #print("shape:{}".format(x.shape))
        if self.outBlock is not None:
            x=self.outBlock(x)
        return  x

"""
1. resnet_net 采用了后4个不同尺度的特征图图  level：4
2 不同与原始的unet ,resnet_net在解码部分将padding设置为1 
3. resnet_net的 stride为4，表示 输出的宽和高是输入宽高的 1/4
"""
class Res18_UNet(UNet):
    def __init__(self,pretrained,n_classes,norm_layer=None,bilinear=True):
        self.pretrained=pretrained
        base_channels = 64   # resnet18 和resnet34 这里为64
        level = 4
        padding = 1
        super(Res18_UNet,self).__init__(n_classes,base_channels,level,padding,norm_layer,bilinear)
    def build_encoder(self):
        return resnet18(self.pretrained)

class Res50_UNet(UNet):
    def __init__(self,pretrained,n_classes,norm_layer=None,bilinear=True):
        self.pretrained=pretrained
        base_channels = 256    # resnet50 ，resnet101和resnet152 这里为256
        level = 4
        padding = 1
        super(Res50_UNet,self).__init__(n_classes,base_channels,level,padding,norm_layer,bilinear)
    def build_encoder(self):
        return resnet50(self.pretrained)

if __name__=="__main__":

    ipt=torch.rand(1,3,572,572)

    unet1=UNet(10,base_channels=16,level=5)
    opt = unet1(ipt)
    print(opt.shape)

    unet2=UNet(10,base_channels=16,level=5,padding=1)
    opt = unet2(ipt)
    print(opt.shape)

    ipt=torch.rand(1,3,512,512)

    res18net=Res18_UNet(pretrained=False,n_classes=10)
    opt=res18net(ipt)
    print(opt.shape)

    res50net=Res50_UNet(pretrained=False,n_classes=10)
    opt=res50net(ipt)
    print(opt.shape)