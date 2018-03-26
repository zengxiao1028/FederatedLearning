import torch
import torchvision
import argparse
import torchvision.transforms as transforms
from torchvision import datasets, transforms
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
########################################################################
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].

# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#
# trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                         download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
#                                           shuffle=True, num_workers=2)
#
# testset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                        download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=4,
#                                          shuffle=False, num_workers=2)
#
# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                    help='input batch size for training (default: 4)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
trainloader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('./fashion_data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
testloader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('./fashion_data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)

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
import test_vd as vd
seed = 1337

class Net(nn.Module):
    def __init__(self, net= None):
        super(Net, self).__init__()

        torch.manual_seed(seed)
        self.conv1 = vd.Conv2d(1, 6, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = vd.Conv2d(6, 16, 5, padding=2)
        self.fc1 = vd.Full(16 * 7 * 7, 1024)
        self.fc2 = vd.Full(1024, 10)
        #self.fc3 = vd.Full(84, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 7 * 7)
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        x = self.fc2(x)
        y = self.conv1.dkl_dp() + self.conv2.dkl_dp() + self.fc1.dkl_dp() + \
            self.fc2.dkl_dp()
        return x, y

net = Net()
net = net.cuda()
import torch.optim as optim
log_alphas = vd.gather_logalphas(net)
x = vd.sparseness(log_alphas)
print(x)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
vd.phase = True  #in the training stage
for epoch in range(20):  # loop over the dataset multiple times

    running_loss = 0.0
    vd.phase = True
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

        # zero the parameter gradientstest.py:72
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs, y = net(inputs)
        loss = criterion(outputs, labels)
        log_alphas = vd.gather_logalphas(net)
        loss = loss + 1./ 50000 * y
        loss.backward(retain_graph=True)
        optimizer.step()
        x = vd.sparseness(log_alphas)


        # print statistics
        running_loss += loss.data[0]
        if i % 100 == 0:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f sparsity: %.4f' %
                  (epoch + 1, i + 1, running_loss / 2000, vd.sparseness(log_alphas)))
            running_loss = 0.0

    correct = 0
    total = 0
    vd.phase = False
    for data in testloader:
        images, labels = data
        outputs, y = net(Variable(images.cuda()))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.cuda()).sum()

    print(correct)

    print('Accuracy of the network on the 10000 test images: %.3f %%' % (
            100 * correct / total))

print('Finished Training')

vd.phase = False  #in the testing stage

correct = 0
total = 0
for data in testloader:
    images, labels = data
    outputs,y = net(Variable(images.cuda()))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels.cuda()).sum()

print(correct)

print('Accuracy of the network on the 10000 test images: %.3f %%' % (
    100 * correct / total))


