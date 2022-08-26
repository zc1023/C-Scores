
import torchvision
import torch
import torch.nn as nn
import random



def ConvRNRelu(in_channels,out_channels,kernel_size,stride=1):
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size//2,
            ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

class Block(nn.Module):
    # out_channels = out_channels1+out_channels2
    def __init__(self,in_channels,out_channels1,out_channels2) -> None:
        super(Block,self).__init__()
        self.branch1 = ConvRNRelu(in_channels=in_channels,out_channels=out_channels1,kernel_size=1)
        self.branch2 = ConvRNRelu(in_channels=in_channels,out_channels=out_channels2,kernel_size=3)
    def forward(self,x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        return torch.cat((out1,out2),dim=1)

def Stage1(in_channels):
    # out_channels = 160
    return nn.Sequential(
        Block(in_channels=in_channels,out_channels1=32,out_channels2=32),
        Block(in_channels=32+32,out_channels1=32,out_channels2=48),
        ConvRNRelu(in_channels=32+48,out_channels=160,kernel_size=3,stride=2),
    )
def Stage2(in_channels):
    # out_channels = 240
    return nn.Sequential(
        Block(in_channels=in_channels,out_channels1=112,out_channels2=48),
        Block(in_channels=112+48,out_channels1=96,out_channels2=64),
        Block(in_channels=96+64,out_channels1=80,out_channels2=80),
        Block(in_channels=80+80,out_channels1=48,out_channels2=96),
        ConvRNRelu(in_channels=48+96,out_channels=240,kernel_size=3,stride=2),
    )
def Stage3(in_channels):
    # out_channels =176+160
    return nn.Sequential(
        Block(in_channels=in_channels,out_channels1=176,out_channels2=160),
        Block(in_channels=176+160,out_channels1=176,out_channels2=160),
    )

class Inception(nn.Module):
    def __init__(self,class_num=10) -> None:
        super(Inception,self).__init__()
        self.Conv = ConvRNRelu(in_channels=3,out_channels=96,kernel_size=3)
        self.Stage1 = Stage1(in_channels=96)
        self.Stage2 = Stage2(in_channels=160)
        self.Stage3 = Stage3(in_channels=240)
        self.GlobalMaxPool = nn.AdaptiveMaxPool2d(1)
        self.Linear = nn.Linear(in_features=176+160,out_features=class_num)
    def forward(self,x):
        x = self.Conv(x)
        x = self.Stage1(x)
        x = self.Stage2(x)
        x = self.Stage3(x)
        x = self.GlobalMaxPool(x)
        x = x.view(x.size(0),-1)
        out = self.Linear(x)
        return out
if __name__ =="__main__":
    model = Inception()
    input = torch.randn(1024,3,32,32)
    output = model(input)
    print(output)
