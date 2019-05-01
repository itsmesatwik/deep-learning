import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F

import torchvision.datasets as datasets
import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable

import math
import os
import time
import numpy as np

class Discriminator(nn.Module):
    def __init__(self):
        super(Descriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=196, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm(normmalized_shape=[196,32,32]),
            nn.LeakyReLU(0.02))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=2, padding=1),
            nn.LayerNorm(normmalized_shape=[196,16,16]),
            nn.LeakyReLU(0.02))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm(normmalized_shape=[196,16,16]),
            nn.LeakyReLU(0.02))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=2, padding=1),
            nn.LayerNorm(normmalized_shape=[196,8,8]),
            nn.LeakyReLU(0.02))
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm(normmalized_shape=[196,8,8]),
            nn.LeakyReLU(0.02))
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm(normmalized_shape=[196,8,8]),
            nn.LeakyReLU(0.02))
        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm(normmalized_shape=[196,8,8]),
            nn.LeakyReLU(0.02))
        self.conv8 = nn.Sequential(
            nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=2, padding=1),
            nn.LayerNorm(normmalized_shape=[196,4,4]),
            nn.LeakyReLU(0.02))
        self.pool = nn.MaxPool2d(kernel_size=4, stride=4)
        self.fc1 = nn.Linear(196,1)
        self.fc2 = nn.Linear(196,10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.pool(x)
        x = x.veiw(x.size(0), -1)
        out1 = self.fc1(x)
        out2 = self.fc2(x)
        return out1, out2

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(100, 196*4*4),
            nn.BatchNorm1d(196*4*4),
            nn.ReLU())
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=196, out_channels=196, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(196),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(196),
            nn.ReLU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(196),
            nn.ReLU())
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(196),
            nn.ReLU())
        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=196, out_channels=196, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(196),
            nn.ReLU())
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(196),
            nn.ReLU())
        self.conv7 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=196, out_channels=196, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(196),
            nn.ReLU())
        self.conv8 = nn.Sequential(
            nn.Conv2d(in_channels=196, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Tanh())

    def forward(self, x):
        x = self.fc1(x)
        x = x.view(-1, 196, 4, 4)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        return x


batch_size = 128

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.7, 1.0), ratio=(1.0,1.0)),
    transforms.ColorJitter(
            brightness=0.1*torch.randn(1),
            contrast=0.1*torch.randn(1),
            saturation=0.1*torch.randn(1),
            hue=0.1*torch.randn(1)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainset = torchvision.datasets.CIFAR10(root='./', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(root='./', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)


model =  Discriminator()
model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

EPOCH = 200

for ep in range(EPOCH):
    running_loss = 0.0
    if(epoch==50):
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate/10.0
    if(epoch==75):
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate/100.0

    for batch_idx, (X_train_batch, Y_train_batch) in enumerate(trainloader):

        if(Y_train_batch.shape[0] < batch_size):
            continue

        X_train_batch = Variable(X_train_batch).cuda()
        Y_train_batch = Variable(Y_train_batch).cuda()
        _, output = model(X_train_batch)

        loss = criterion(output, Y_train_batch)
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
torch.save(model,'cifar10.model')





