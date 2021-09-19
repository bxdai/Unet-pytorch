""" Parts of the U-Net model """
#%%
import torch
from torch import cuda
from torch._C import device
from torch.distributed import is_available
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.upsampling import Upsample

#Encoder：使得模型理解了图像的内容，但是丢弃了图像的位置信息。
#Decoder：使模型结合Encoder对图像内容的理解，恢复图像的位置信息。

class DoubleConv(nn.Module):
    """convolution->BN->Relu"""
    def __init__(self,in_channels,out_channels,mid_channels = None):
        super().__init__()

        #这部分和论文不一样,多了个中间输入的channel
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels,mid_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels,out_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self,x):
        return self.double_conv(x)


#编码器encoder
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels,out_channels,maxPool = True):
        super().__init__()

        maxPool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels,out_channels),
        )
        
        #使用卷积代替maxPoling 下采样,这样不会丢失位置信息
        #但是如果网络太深，会产生过拟合
        down_conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            DoubleConv(in_channels,out_channels),
            
        )

        self.downsample = ( maxPool_conv if maxPool  else  down_conv)

      
    def forward(self,x):
        return self.downsample(x)

#解码器Decoder
class Up(nn.Module):
    """Downscaling with maxpool then double conv"""
    
    def __init__(self,in_channels,out_channels, bilinear=True):
        super().__init__()

        #如果是双线性插值，使用普通卷积来减少通道
        if bilinear:
            self.up = Upsample(scale_factor=2,mode='bilinear',align_corners=True)
            self.conv = DoubleConv(in_channels,out_channels,in_channels // 2)
        else:
            #采用转置卷积代替上采样，out_channel 是in_channels的一半
            self.up = nn.ConvTranspose2d(in_channels,in_channels//2,kernel_size=2,stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self,x1,x2):
        #据论文，需要进行融合还原之前的尺寸
        #x1是上采样获得的特征
        #x2是下采样获得的特征
        x1 = self.up(x1)
        if (x1.size(2) != x2.size(2)) or (x1.size(3) != x2.size(3)):
            #input is CHW
            #这个是解决填充不一致的问题
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]
            #print('sizes',x1.size(),x2.size(),diffX // 2, diffX - diffX//2, diffY // 2, diffY - diffY//2)
            x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                            diffY // 2, diffY - diffY//2))
            #print("pad x1:",x1.size())
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class outlayer(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=1)
    def forward(self,x):
        x =self.conv(x)
        return x



class UNet(nn.Module):
    def __init__(self,n_classes,n_channels = 3,bilinear=True) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.n_channels = n_classes
        self.bilinear = bilinear

        self.start = DoubleConv(n_channels,64)

        self.down1 = Down(64,128)
        self.down2 = Down(128,256)
        self.down3 = Down(256,512)
        #self.down4 = Down(512,1024)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.final_conv = outlayer(64, n_classes)

        # self.up1 = Up(1024, 512,bilinear)
        # self.up2 = Up(512, 256,bilinear)
        # self.up3 = Up(256, 128 ,bilinear)
        # self.up4 = Up(128, 64,bilinear)
        # self.final_conv = outlayer(64, n_classes)

    def forward(self,x):
        x0 = self.start(x) #3-64
        #print(x0.shape)
        x1 = self.down1(x0)#64-128
        #print(f"x1.shape:\n{x1.shape}")
        x2 = self.down2(x1)#128-246
        #print(f"x2.shape:\n{x2.shape}")
        x3 = self.down3(x2)#256-512
        #print(f"x3.shape:\n{x3.shape}")
        x4 = self.down4(x3)#512-1024
        #print(f"x4.shape:\n{x4.shape}")

        x = self.up1(x4, x3)#1024-512
        #print(f"x.shape:\n{x.shape}")
        x = self.up2(x, x2)#512-256
        #print(f"x.shape:\n{x.shape}")
        x = self.up3(x, x1)#256-128
        #print(f"x.shape:\n{x.shape}")
        x = self.up4(x, x0)#128-64
        #print(f"x.shape:\n{x.shape}")
        logits = self.final_conv(x)
        #print(f"logits:\n{logits.shape}")
        return logits



if __name__ == '__main__':
    net = UNet(n_channels=1,n_classes=2,bilinear=False)
    dev = ('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(dev)
    x = torch.randn(1,1,572,572)
    out = net(x).to(dev)
    #print(net)
    print(out.shape)


     




    
# %%
