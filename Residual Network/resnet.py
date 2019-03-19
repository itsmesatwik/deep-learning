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



import matplotlib.pyplot as plt


transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(0.4),
    transforms.RandomVerticalFlip(0.4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainset = torchvision.datasets.CIFAR100(root='../', train=True, 
                                        download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR100(root='../', train=False, 
                                       download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)



def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,
        stride=stride,padding=1,bias=False)


class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):

        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return out



class ResNet(nn.Module):

    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = conv3x3(in_channels=3, out_channels=32)
        self.bn = nn.BatchNorm2d(num_features=32)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=0.02)

        # block of Basic Blocks
        self.conv2_x = self.make_block(2, 32, 32)
        self.conv3_x = self.make_block(4, 32, 64, stride=2)
        self.conv4_x = self.make_block(4, 64, 128, stride=2)
        self.conv5_x = self.make_block(2, 128, 256, stride=2)

        self.maxpool = nn.MaxPool2d(kernel_size=4, stride=1)
        self.fc_layer = nn.Linear(256, 100)

        

    def make_block(self, blocks, in_channels, out_channels, stride=1):
        downsample = None
        if (stride != 1) or (in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(num_features=out_channels)
            )

        layers = []
        layers.append(
            BasicBlock(in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2_x(out)
        out = self.conv3_x(out)
        out = self.conv4_x(out)
        out = self.conv5_x(out)

        out = self.maxpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc_layer(out)

        return out


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet().to(device)
criterion = nn.CrossEntropyLoss()

EPOCHS=100
LR=0.001
batch_size=128


optimizer = optim.Adam(model.parameters(), lr=LR)


for epoch in range(EPOCHS):
    epoch_time = time.time()
    running_loss = 0.0
    train_accu = []
    for i, data in enumerate(trainloader, 0):
        images, labels = data
        images = Variable(images).to(device)
        labels = Variable(labels).to(device)
        optimizer.zero_grad()


        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        prediction = outputs.data.max(1)[1]
        accuracy = ( float( prediction.eq(labels.data).sum() ) /float(batch_size))*100.0
        train_accu.append(accuracy)

    print("Time Taken Per Epoch: ", (time.time() - epoch_time))
    print("Train Accuracy in Epoch ", epoch, " ", np.mean(train_accu))

    if ((epoch+1)%10==0):
        correct = 0
        total = 0
        for data in testloader:
            inputs, labels = data
            inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct+=predicted.eq(labels.data).sum()

        print("Epoch: ", epoch, ' Accuracy of the network on the test images: ', (100.0 * float(correct) / float(total)))



			

