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



def resnet18(pretrained = True) :
    model_urls = {'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth'}
    model = torchvision.models.resnet.ResNet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2])
    if pretrained :
        model.load_state_dict(torch.utils.model_zoo.load_url(model_urls['resnet18'], model_dir ='./'))
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS=100
LR=0.001
batch_size=128

loss_func = nn.CrossEntropyLoss()
model = resnet18().to(device)
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
        loss = model(outputs, labels)
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


