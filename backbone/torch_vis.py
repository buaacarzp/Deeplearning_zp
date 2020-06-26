import torch
from torchvision.models import resnet50
from torchviz import make_dot
model = resnet50()
print(model)
X =torch.rand(1,3,255,255)
# g=make_dot(model(X.requires_grad_(True)),params=dict(list(model.named_parameters())+ [('x', X)]))
g=make_dot(model(torch.rand(1,3,255,255)),params=dict(model.named_parameters()))
count=0
for name,params in model.named_parameters():
    # params.requires_grad = False
    if name.split('.')[0]=='fc':
        count+=1
        params.requires_grad = False
print(count)
for name,params in model.named_parameters():
    if name.split('.')[0]=='fc':
        count+=1
        print(params.requires_grad)
g.view()
