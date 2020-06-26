import torch
import torch.nn as nn
import torchvision
import numpy as np
def Conv1(in_planes, places, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes,out_channels=places,kernel_size=7,stride=stride,padding=3, bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
class Bottleneck(nn.Module):
    def __init__(self,in_places,places, stride=1,downsampling=False, expansion = 4):
        super(Bottleneck,self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places,out_channels=places,kernel_size=1,stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places*self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places*self.expansion),
        )

        if self.downsampling: #对输入进行升通道
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(places*self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        residual = x
        out = self.bottleneck(x)

        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out) 
        return out

class ResNet(nn.Module):
    def __init__(self,blocks, num_classes=1000, expansion = 4):
        super(ResNet,self).__init__()
        self.expansion = expansion
        self.conv1 = Conv1(in_planes = 3, places= 64)
        self.layer1 = self.make_layer(in_places = 64, places= 64, block=blocks[0], stride=1)
        self.layer2 = self.make_layer(in_places=256,places=128,block =blocks[1],stride=2)
        self.layer3 = self.make_layer(in_places=512,places=256,block =blocks[2],stride=2)
        self.layer4 = self.make_layer(in_places=1024,places=512,block =blocks[3],stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(2048,num_classes)
    def make_layer(self, in_places, places, block, stride): #针对每一个stage
        layers = []
        layers.append(Bottleneck(in_places, places,stride, downsampling =True))
        for i in range(1, block):
            #将输入和输出通道数目置换
            layers.append(Bottleneck(places*self.expansion, places)) #从第二个bottenneck开始，上一个输入通道是当前的4倍数
        return nn.Sequential(*layers)
    def forward(self,X):
        # print(X.shape)
        X = self.conv1(X)
        # print(X.shape)
        X = self.layer1(X)
        X = self.layer2(X)
        X = self.layer3(X)
        X = self.layer4(X)
        X = self.avgpool(X)
        X = X.view(X.size(0), -1)
        X = self.fc(X)
        return X
inputs = torch.randn(1, 3, 224, 224)
blocks =[3,4,6,3]
model  = ResNet(blocks)
channel_in = model.fc.in_features
print(channel_in)
class_num = 2
model.fc = nn.Sequential(
    nn.Linear(channel_in, class_num),
    nn.ReLU(),
    # nn.Dropout(0.4),
    # nn.Linear(256, class_num),
    nn.LogSoftmax(dim=1)
)
print(model)
# for k in model.modules():
#     print(k)
# out =model(inputs)
# print(out[0][0])
# model.load_state_dict(torch.load('./model/resnet_50.pth'),False)
# out =model(inputs)
# print(out[0][0])
