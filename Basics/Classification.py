import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision
import torch.utils.data as Data
from torchvision import transforms


# Hyper parameters
EPOCH = 3
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = True

train_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=True,
    transform=transforms.ToTensor(),    # (0, 1)
    download=DOWNLOAD_MNIST
)

train_loader = Data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

test_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=False,
    transform=transforms.ToTensor(),
    download=DOWNLOAD_MNIST
)

test_loader = Data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

print(train_data.data[0])
print(train_data.targets[0])

# plot one example
print(train_data.data.size())
print(train_data.targets.size())
plt.imshow(train_data.data[0].numpy(), cmap='gray')
plt.title('%i' % train_data.targets[0])
plt.show()


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(  # (1, 28, 28)
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,  # if stride = 1, padding = (kernel_size-1)/2
            ),  # -> (16, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)     # -> (16, 14, 14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),     # -> (32, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(2),        # -> (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)   # (batch, 32, 7, 7)
        x = x.view(x.size(0), -1)   # (batch, 32 * 7 * 7)
        output = self.out(x)
        return output


cnn = CNN()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
cnn.to(device)
print(cnn)

optimizer = optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for batch, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)
        output = cnn(x)
        loss = loss_func(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 100 == 0:
            print('Epoch: {}, Batch: {}, Loss: {}'.format(epoch, batch, loss.item()))
print('Finished Training')

print('Now Testing...')
# training accuracy
correct = 0
total = 0
with torch.no_grad():
    for data in train_loader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = cnn(images)
        _, predict = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predict == labels).sum().item()
    print('Training Accuracy: %.2f %%' % (100 * correct / total))

# test accuracy
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = cnn(images)
        _, predict = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predict == labels).sum().item()
    print('Test Accuracy: %.2f %%' % (100 * correct / total))
