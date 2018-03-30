import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from collections import Counter

########################################################################
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),

    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
batch_size = 10
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)

# the number of nets for federated learning
net_number = 5
indices = list(range(len(trainset)))
len = len(trainset) // net_number
train_idx = [None]*net_number
#determine the trainset for each net
trainsets = [None]*net_number
trainloader = [None]*net_number
for i in range(net_number):
    train_idx[i] = indices[i*len:(i+1)*len]

for i in range(net_number):
    trainsets[i] = SubsetRandomSampler(train_idx[i])

for i in range(net_number):
    trainloader[i] = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler= trainsets[i],
                                          shuffle=False, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

########################################################################
# Let us show some of the training images, for fun.

########################################################################
# 2. Define a Convolution Neural Network
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Copy the neural network from the Neural Networks section before and modify it to
# take 3-channel images (instead of 1-channel images as it was defined).

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import varitional_dropout as vd
seed = 1337
class Net(nn.Module):
    def __init__(self, net= None):
        super(Net, self).__init__()

        torch.manual_seed(seed)
        self.conv1 = vd.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = vd.Conv2d(6, 16, 5)
        self.fc1 = vd.Full(16 * 5 * 5, 120)
        self.fc2 = vd.Full(120, 84)
        self.fc3 = vd.Full(84, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        y = self.conv1.dkl_dp() + self.conv2.dkl_dp() + self.fc1.dkl_dp() + \
            self.fc2.dkl_dp() + self.fc3.dkl_dp()
        return x, y



class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        torch.manual_seed(seed)
        self.conv1 = vd.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = vd.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class MobileNet(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]

    def __init__(self, num_classes=10):
        torch.manual_seed(seed)
        super(MobileNet, self).__init__()
        self.conv1 = vd.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.linear = vd.Full(1024, num_classes)

    def _make_layers(self, in_planes):
        torch.manual_seed(seed)
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        y = self.conv1.dkl_dp()
        for i in range(13):
            y += self.layers[i].conv1.dkl_dp()
            y += self.layers[i].conv2.dkl_dp()
        y += self.linear.dkl_dp()
        return out, y







net = [None]*net_number

for i in range(net_number):
    net[i] = MobileNet()
    net[i] = net[i].cuda()





########################################################################
# 3. Define a Loss function and optimizer
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Let's use a Classification Cross-Entropy loss and SGD with momentum.

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = [None]*net_number
for i in range(net_number):
    optimizer[i] = optim.Adam(net[i].parameters(), lr = 0.001)



########################################################################
# 4. Train the network
# ^^^^^^^^^^^^^^^^^^^^
#
# This is when things start to get interesting.
# We simply have to loop over our data iterator, and feed the inputs to the
# network and optimize.

# print('net1 fc3 weight')
# print(net1.fc3.weight)
# print('net2 fc3 weight')
# print(net2.fc3.weight)
# print('net3 fc3 weight')
# print(net3.fc3.weight)
vd.phase = 0
for epoch in range(100):  # loop over the dataset multiple times
    running_loss = [None]*net_number
    for i in range(net_number):
        running_loss[i] = 0.0

    dataiter = [None]*net_number
    for i in range(net_number):
        dataiter[i] = iter(trainloader[i])
    print (epoch + 1)
    vd.phase = True
    for k, data1 in enumerate(trainloader[0], 0):
        inputs = [None]*net_number
        labels = [None]*net_number
        for i in range(net_number):
            inputs[i], labels[i] = dataiter[i].next()
            inputs[i], labels[i] = Variable(inputs[i].cuda()), Variable(labels[i].cuda())

        for i in range(net_number):
            optimizer[i].zero_grad()

        #define output and loss
        outputs = [None]*net_number
        y = [None]*net_number
        loss = [None]*net_number
        # forward + backward + optimize
        loss = [None] * net_number
        # forward + backward + optimize
        for i in range(net_number):
            outputs[i], y[i] = net[i](inputs[i])
            loss[i] = criterion(outputs[i], labels[i])

        for i in range(net_number):
            loss[i].backward(retain_graph=True)
        grad_of_params = [None]*10
        for i in range(net_number):
            grad_of_params[i] = {}
        grad_of_params_total = {}
        grad_of_params = {}
        #calculate the sum of the gradients for all nets
        for i in range(net_number):
            for name, parameter in net[i].named_parameters():

                if 'log' not in name:
                    if i == 0:
                        grad_of_params[name] = parameter.grad.data
                    else:
                        grad_of_params[name] = grad_of_params[name] + parameter.grad.data


        #calculate the average gradeints
        for i in range(net_number):
            for name, parameter in net[i].named_parameters():
                if 'log' not in name:
                    parameter.grad = Variable(grad_of_params[name] / (net_number * 1.0))


        for i in range(net_number):
            optimizer[i].step()

        log_alphas = [None]*net_number
        sparesity = [None]*net_number
        for i in range(net_number):
            log_alphas[i] = vd.gather_logalphas(net[i])
            sparesity[i] = vd.sparseness(log_alphas[i])

        for i in range(net_number):
            running_loss[i] += loss[i].data[0]
        # print statistics

        if k % 100 == 0:    # print every 2000 mini-batches
            for j in range(net_number):
                net_name = 'net' + str(j + 1)
                print('net %d [%d, %5d] loss %d: %.3f, sparsity %d: %.3f' %
                      (j + 1, epoch + 1, k + 1, j+ 1, running_loss[j] / 2000, j + 1, sparesity[j]))
                running_loss[j] = 0.0

    correct = [None]*net_number
    for i in range(net_number):
        correct[i] = 0
    total = 0
    predicted = [None]*net_number
    vd.phase = False
    for data in testloader:
        images, labels = data
        for i in range(net_number):
            outputs[i], y[i] = net[i](Variable(images.cuda()))
            _, predicted[i] = torch.max(outputs[i].data, 1)
            correct[i] += (predicted[i] == labels.cuda()).sum()
        total += labels.size(0)

    for i in range(net_number):
        print('Net %d Accuracy of the network on the 10000 test images: %.3f %%' %(
                i + 1, 100*correct[i] / total))



print('Finished Training')





