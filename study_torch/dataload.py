import torchvision
import torchvision.transforms as transforms
import torch
def dataloader():
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='data', train=True,
                                            download=False, transform=transform)
    #使用多gpu的时候，确保train和test的bacth_size能被样本总数整除，或者加上drop_last
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,
                                            shuffle=True, num_workers=2,drop_last=True)

    testset = torchvision.datasets.CIFAR10(root='data', train=False,
                                        download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=16,
                                            shuffle=False, num_workers=2,drop_last=True)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainset,testset,trainloader,testloader
if __name__ == "__main__":
    trainset,testset,trainloader,testloader = dataloader()
    print(len(trainset)/16)
    