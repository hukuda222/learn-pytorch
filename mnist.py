import torch as t
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import numpy as np


class Net(nn.Module):
    def __init__(self, batch_size=100):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, (5, 5))
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.batch_size = batch_size

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(self.batch_size, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


if __name__ == "__main__":

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))]
    )

    trainset, testset = [torchvision.datasets.MNIST(
        root='./data', train=b, download=True,
        transform=transform) for b in [True, False]]

    trainloader, testloader = [t.utils.data.DataLoader(
        dataset, batch_size=100, shuffle=b, num_workers=2)
        for dataset, b in zip([trainset, testset], [True, False])]

    classes = tuple(np.linspace(0, 9, 10, dtype=np.uint8))

    net = Net()
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(10):
        loss = 0
        print(epoch)
        for inputs, labels in trainloader:
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss += loss.item()
    correct = 0
    total = 0
    with t.no_grad():
        for inputs, labels in testloader:
            outputs = net(inputs)
            _, pred = t.max(outputs.data, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    print(correct / total)
