import torch
import torch.nn as nn
# from torchvision.models import vgg16_bn
from IPython import embed
class VGG16(nn.Module):
    
    def __init__(self,num_classes=1000):
        super(VGG16,self).__init__()
        self.num_classes = num_classes
        self.cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        self.Backbone = self.backbone(self.cfg)
        self.Head = self.head(self.num_classes)
        
    def head(self,num_classes):
        return nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def backbone(self,cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self,x):
        X = self.Backbone(x)
        X = X.view(x.size(0), -1)
        X = self.Head(X)
        return X

vgg16 =VGG16()
print(vgg16)
if __name__ =="__main__":
    x = torch.randn([1,3,255,255])
    out = vgg16(x)
    print(out.shape)