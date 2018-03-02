import torch
import torchvision
import torchvision.transforms as transforms
from collections import Counter
########################################################################
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128*3,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)
print (len(trainset))
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

########################################################################
# Let us show some of the training images, for fun.
import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


########################################################################
# 2. Define a Convolution Neural Network
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Copy the neural network from the Neural Networks section before and modify it to
# take 3-channel images (instead of 1-channel images as it was defined).

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net1 = Net()
net2 = Net()
net3 = Net()
########################################################################
# 3. Define a Loss function and optimizer
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Let's use a Classification Cross-Entropy loss and SGD with momentum.

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer1 = optim.SGD(net1.parameters(), lr=0.001, momentum=0.9)
optimizer2 = optim.SGD(net2.parameters(), lr=0.001, momentum=0.9)
optimizer3 = optim.SGD(net3.parameters(), lr=0.001, momentum=0.9)

########################################################################
# 4. Train the network
# ^^^^^^^^^^^^^^^^^^^^
#
# This is when things start to get interesting.
# We simply have to loop over our data iterator, and feed the inputs to the
# network and optimize.
j = 0
for epoch in range(50):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        inputs1, labels1 = inputs[0:128], labels[0:128]
        inputs2, labels2 = inputs[128:256], labels[128:256]
        inputs3, labels3 = inputs[256:384], labels[256:384]
        # wrap them in Variable
        inputs1, labels1 = Variable(inputs1), Variable(labels1)
        inputs2, labels2 = Variable(inputs2), Variable(labels2)
        inputs3, labels3 = Variable(inputs3), Variable(labels3)
        # zero the parameter gradients
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        # forward + backward + optimize
        outputs1 = net1(inputs1)
        outputs2 = net2(inputs2)
        outputs3 = net3(inputs3)
        loss1 = criterion(outputs1, labels1)
        loss2 = criterion(outputs2, labels2)
        loss3 = criterion(outputs3, labels3)
        loss1.backward()
        loss2.backward()
        loss3.backward()
        grad_of_params1 = {}
        grad_of_params2 = {}
        grad_of_params3 = {}
        grad_of_params = {}
        for name, parameter in net1.named_parameters():
            grad_of_params1[name] = parameter.grad
            grad_of_params[name] = parameter.grad.data
        for name, parameter in net2.named_parameters():
            grad_of_params2[name] = parameter.grad
            grad_of_params[name] = grad_of_params[name]+ parameter.grad.data
        for name, parameter in net3.named_parameters():
            grad_of_params3[name] = parameter.grad
            grad_of_params[name] = grad_of_params[name]+ parameter.grad.data

        #grad_of_params = dict(Counter(grad_of_params1) + Counter(grad_of_params2) + Counter(grad_of_params3))
        updates = {}
        for key, value in grad_of_params.iteritems():
            value = value / 3.0
            shape = value.shape
            value_array = value.numpy()
            dim = value_array.ndim
            length = value_array.shape[0]
            if dim == 1:
                norml1 = [np.sum(np.fabs(value_array[idx])) for idx in
                              range(length)]  # calculate the filter rank using l1
                pos = np.argsort(norml1)[length - 1]
                updates[key] = pos
            elif dim == 2:
                norml1 = [np.sum(np.fabs(value_array[idx,:])) for idx in
                              range(length)]  # calculate the filter rank using l1
                pos = np.argsort(norml1)[length - 1]
                updates[key] = pos
            elif dim == 4:
                norml1 = [np.sum(np.fabs(value_array[idx,:, :, :,])) for idx in
                              range(length)]  # calculate the filter rank using l1
                pos = np.argsort(norml1)[length - 1]
                updates[key] = pos
        #print updates

        #for key in grad_of_params.keys():
            #print key
            #print grad_of_params1[key],grad_of_params2[key],grad_of_params3[key]
        #print len(grad_of_params)
        for name, parameter in net1.named_parameters():
            grad_of_params1[name] = parameter.grad

            tensor = grad_of_params[name][updates[name]]

            if grad_of_params1[name].data.numpy().ndim == 1:
                tensor1 = torch.Tensor(1)
                tensor1[0] = tensor
                tensor = tensor1
            #print tensor
            tensor = Variable(tensor)
            grad_of_params1[name][updates[name]] = tensor

        if epoch > 4:
            for name, parameter in net2.named_parameters():
                grad_of_params2[name] = parameter.grad
                tensor = grad_of_params[name][updates[name]]
                if grad_of_params2[name].data.numpy().ndim == 1:
                    tensor1 = torch.Tensor(1)
                    tensor1[0] = tensor
                    tensor = tensor1
                #print tensor
                tensor = Variable(tensor)
                grad_of_params2[name][updates[name]] = tensor

            for name, parameter in net3.named_parameters():
                grad_of_params3[name] = parameter.grad
                tensor = grad_of_params[name][updates[name]]
                if grad_of_params3[name].data.numpy().ndim == 1:
                    tensor1 = torch.Tensor(1)
                    tensor1[0] = tensor
                    tensor = tensor1
                #print tensor
                tensor = Variable(tensor)
                grad_of_params3[name][updates[name]] = tensor

        optimizer1.step()
        optimizer2.step()
        optimizer3.step()

        # print statistics
        running_loss += loss1.data[0]
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

########################################################################
# 5. Test the network on the test data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We have trained the network for 2 passes over the training dataset.
# But we need to check if the network has learnt anything at all.
#
# We will check this by predicting the class label that the neural network
# outputs, and checking it against the ground-truth. If the prediction is
# correct, we add the sample to the list of correct predictions.
#
# Okay, first step. Let us display an image from the test set to get familiar.

dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

########################################################################
# Okay, now let us see what the neural network thinks these examples above are:

outputs = net1(Variable(images))

########################################################################
# The outputs are energies for the 10 classes.
# Higher the energy for a class, the more the network
# thinks that the image is of the particular class.
# So, let's get the index of the highest energy:
_, predicted = torch.max(outputs.data, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

########################################################################
# The results seem pretty good.
#
# Let us look at how the network performs on the whole dataset.

correct1 = 0
correct2 = 0
correct3 = 0
total = 0
for data in testloader:
    images, labels = data
    outputs1 = net1(Variable(images))
    outputs2 = net2(Variable(images))
    outputs3 = net3(Variable(images))
    _, predicted1 = torch.max(outputs1.data, 1)
    _, predicted2 = torch.max(outputs2.data, 1)
    _, predicted3 = torch.max(outputs3.data, 1)
    total += labels.size(0)
    correct1 += (predicted1 == labels).sum()
    correct2 += (predicted2 == labels).sum()
    correct3 += (predicted3 == labels).sum()

print('Net1 Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct1 / total))

print('Net2 Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct2 / total))

print('Net3 Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct3 / total))
########################################################################
# That looks waaay better than chance, which is 10% accuracy (randomly picking
# a class out of 10 classes).
# Seems like the network learnt something.
#
# Hmmm, what are the classes that performed well, and the classes that did
# not perform well:

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
for data in testloader:
    images, labels = data
    outputs = net1(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    c = (predicted == labels).squeeze()
    for i in range(4):
        label = labels[i]
        class_correct[label] += c[i]
        class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))