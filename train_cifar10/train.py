from torchvision.models import resnet18
import sys 
import os 
sys.path.append("/mnt/sda1/wenmei_space/zhoupeng/Deeplearning_zp")
# print(sys.path)
# from backbone.resnet_50_torch import ResNet as resnet50
import torch
import torch.nn as nn
from dataload import dataloader
import torch.optim as optim
model = resnet18()
channel_in = model.fc.in_features
class_num = 10
model.fc = nn.Sequential(
    nn.Linear(channel_in, class_num))
model.load_state_dict(torch.load("/mnt/sda1/wenmei_space/zhoupeng/Deeplearning_zp/models/save_model/14_BestModel.pth"))


trainset,testset,trainloader,testloader =dataloader()
# print(trainset[0][0].shape)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.9)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)

model =nn.DataParallel(model)
model.to(device)
for epoch in range(30):  # loop over the dataset multiple times
    print("epoch=",epoch)
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        if len(labels)<=1:continue
        # print(len(labels))
        inputs = inputs.to(device)
        labels =labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
    #predict acc
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            if len(labels)<=1:continue
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
    if (epoch+1)%5==0:
        PATH = '/mnt/sda1/wenmei_space/zhoupeng/Deeplearning_zp/models/save_model'
        PATH = os.path.join(PATH,str(epoch)+"_"+"BestModel.pth")
        torch.save(model.state_dict(), PATH)

print('Finished Training')

