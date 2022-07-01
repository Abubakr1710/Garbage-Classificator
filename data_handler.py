from cgi import test
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
torch.manual_seed(0)

def dataloader(pth):

    train_transform = transforms.Compose([
        transforms.Resize((120,120)),
        transforms.RandomRotation(20),
        transforms.RandomResizedCrop(100),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        ])

    test_transform = transforms.Compose([
        transforms.Resize((120, 120)),
        transforms.CenterCrop(100),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        ])

    
    trainset = datasets.ImageFolder(pth+'/train', transform=train_transform)
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)

    testset = datasets.ImageFolder(pth + '/test', transform=test_transform)
    testloader = DataLoader(testset, batch_size=32, shuffle=False)

    torch.save(trainset.classes, 'num_classes.pth')

    return trainloader, testloader

    