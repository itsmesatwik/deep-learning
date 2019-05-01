#USES ADAM, DROPOUT, DATA AUGMENTATION AND COMPARING DROPOUT ACCURACY WITH HEURISTIC AND MONTECARLO




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


# Hyper-parameters
num_epochs = 150
batch_size = 128




transform_train = transforms.Compose([
    #transforms.RandomResizedCrop(DIM, scale=(0.7, 1.0), ratio=(1.0,1.0)),
    transforms.RandomHorizontalFlip(0.4),
    transforms.RandomVerticalFlip(0.4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainset = torchvision.datasets.CIFAR10(root='../', train=True, 
                                        download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='../', train=False, 
                                       download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)



print("Class")

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride = 1, padding = 2) 
        self.conv_layer2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride = 1, padding = 2) 
        self.conv_layer3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride = 1, padding = 2)
        self.conv_layer4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride = 1, padding = 2)
        self.conv_layer5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride = 1, padding = 2)
        self.conv_layer6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride = 1, padding = 0)
        self.conv_layer7 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride = 1, padding = 0)
        self.conv_layer8 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride = 1, padding = 0)

        self.fc1 = nn.Linear(1024, 500)
        self.fc2 = nn.Linear(500, 10)
        #self.fc3 = nn.Linear(420,10)
        self.dropout1 = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.4)
        self.dropout3 = nn.Dropout(0.4)
        self.dropout4 = nn.Dropout(0.4)

        self.batch_layer1 = nn.BatchNorm2d(64)
        self.batch_layer2 = nn.BatchNorm2d(64)
        self.batch_layer3 = nn.BatchNorm2d(64)
        self.batch_layer4 = nn.BatchNorm2d(64)
        self.batch_layer5 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2,2)
        self.pool2 = nn.MaxPool2d(2,2)
        self.pool3 = nn.MaxPool2d(2,2)

    def forward(self, x):
        x = F.relu(self.conv_layer1(x))
        x = self.batch_layer1(x)
        x = F.relu(self.conv_layer2(x))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = F.relu(self.conv_layer3(x))
        x = self.batch_layer2(x)
        x = F.relu(self.conv_layer4(x))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        x = F.relu(self.conv_layer5(x))
        x = self.batch_layer3(x)
        x = F.relu(self.conv_layer6(x))
        x = self.dropout3(x)
        
        x = F.relu(self.conv_layer7(x))
        x = self.batch_layer4(x)
        x = F.relu(self.conv_layer8(x))
        x = self.batch_layer5(x)
        #x = self.pool3(x)
        x = self.dropout4(x)
        
        
        x = x.view(-1, 64*4*4)
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        x = F.relu(self.fc2(x))
        return x

print("Class Ends")

conv_nn = ConvNet()
conv_nn.cuda()
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(conv_nn.parameters(), lr = 0.001) #using ADAM

test_accuracy = []





# Train the model
for epoch in range(150):
    
    epoch_time = time.time()
    running_loss = 0.0
    accuracy_list = []
    for batch_idx, data in enumerate(trainloader, 0):
        X_train_batch, labels = data
        
        X_train_batch = Variable(X_train_batch).cuda()
        labels = Variable(labels).cuda()

        optimizer.zero_grad()
        outputs = conv_nn(X_train_batch)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()
        arg_max = outputs.data.max(1)[1]
        accuracy = (float(arg_max.eq(labels.data).sum())/ float(batch_size))
        accuracy_list.append(accuracy)

        running_loss += loss.item()
        #print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / batch_size))

    print("Time Taken Per Epoch: ", (time.time() - epoch_time))
    print("Accuracy in Epoch ", epoch, " ", np.mean(accuracy_list))

print("Training Finished")


# Test the Model

conv_nn.eval()




for data in testloader:
    X_test_batch, Y_test_batch = data
    X_test_batch, Y_test_batch = Variable(X_test_batch).cuda(), Variable(Y_test_batch).cuda()
    output = conv_nn(X_test_batch)
    _, pred = torch.max(output.data,1)
    accuracy = (float(pred.eq(Y_test_batch.data).sum())/ float(batch_size))
    test_accuracy.append(accuracy)

final_accuracy = np.mean(test_accuracy)

print("Final Accuracy: ", final_accuracy)
