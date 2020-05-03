
import torch
import torch.nn as nn
import torch.nn.functional as F



class ConvBlock(nn.Module):
    """conv-norm-relu"""
    def __init__(self, in_channels, out_channels,kernel_size=3,padding=1, norm_layer=None):
        """
        :param in_channels:  输入通道
        :param out_channels: 输出通道
        :param kernel_size: 默认为3
        :param padding:    默认为1，
        :param norm_layer:  默认使用 batch_norm
        """
        super(ConvBlock,self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            norm_layer(out_channels) if norm_layer is not None else  nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.convblock(x)

class UNetBlock(nn.Module):
    """conv-norm-relu,conv-norm-relu"""
    def __init__(self, in_channels, out_channels,mid_channels=None,padding=0, norm_layer=None):
        """
        :param in_channels:
        :param out_channels:
        :param mid_channels:  默认 mid_channels==out_channels
        :param padding:     缺省设置为padding==0(论文中)
        :param norm_layer: 默认使用 batch_norm
        """
        super(UNetBlock,self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.unetblock=nn.Sequential(
            ConvBlock(in_channels,mid_channels,padding=padding,norm_layer=norm_layer),
            ConvBlock(mid_channels, out_channels,padding=padding,norm_layer=norm_layer)
        )
    def forward(self, x):
        return self.unetblock(x)


class UNetUpBlock(nn.Module):
    """Upscaling then unetblock"""

    def __init__(self, in_channels, out_channels,padding=0,norm_layer=None, bilinear=True):
        """
        :param in_channels:
        :param out_channels:
        :param padding:     缺省设置为padding==0(论文中)
        :param norm_layer: 默认使用 batch_norm
        :param bilinear:  使用双线性插值，或转置卷积
        """

        super(UNetUpBlock,self).__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels , in_channels // 2,1,1)
            )
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = UNetBlock(in_channels, out_channels,padding=padding,norm_layer=norm_layer)


    def crop(self,tensor,target_sz):
        _, _, tensor_height, tensor_width = tensor.size()
        diff_y = (tensor_height - target_sz[0]) // 2
        diff_x = (tensor_width - target_sz[1]) // 2
        return tensor[:, :, diff_y:(diff_y + target_sz[0]), diff_x:(diff_x + target_sz[1])]

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW

        x2=self.crop(x2,x1.shape[2:])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNetDownBlock(nn.Module):
    """maxpooling-unetblock"""

    def __init__(self, in_channels, out_channels,padding=0, norm_layer=None):
        super(UNetDownBlock,self).__init__()

        self.down=nn.Sequential(
            nn.MaxPool2d(2),
            UNetBlock(in_channels, out_channels,padding=padding, norm_layer=norm_layer),
        )
    def forward(self, inputs):
        return self.down(inputs)


class Unet_Encoder(nn.Module):
    def __init__(self, in_channels,base_channels,level,padding=0,norm_layer=None,):
        super(Unet_Encoder,self).__init__()
        self.encoder=nn.ModuleList()
        for i in range(level):
            if i==0:
                #第一层，特征图尺寸和原图大小一致
                self.encoder.append(UNetBlock(in_channels, base_channels*(2**i),
                                              padding=padding,norm_layer=norm_layer))
            else:
                self.encoder.append(UNetDownBlock( base_channels*(2**(i-1)),  base_channels*(2**i),
                                                   padding=padding,norm_layer=norm_layer))

    def forward(self, inputs):
        features=[]
        for block in self.encoder:
            inputs=block(inputs)
            features.append(inputs)
        return features