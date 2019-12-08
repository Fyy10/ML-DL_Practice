import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision
import torch.utils.data as Data
from torchvision import transforms

train_data = torchvision.datasets.MNIST(root='./mnist', train=True, transform=transforms.ToTensor(), download=False)
train_loader = Data.DataLoader(train_data, batch_size=50, shuffle=True, num_workers=0)

test_data = torchvision.datasets.MNIST(root='./mnist', train=False, transform=transforms.ToTensor())
test_loader = Data.DataLoader(test_data, batch_size=50, shuffle=True, num_workers=0)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size=28,
            hidden_size=64,
            num_layers=2,
            batch_first=True
        )
        self.out = nn.Linear(64, 10)

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)   # x (batch, time_step, input_size)
        out = self.out(r_out[:, -1, :])     # (batch, time_step, input_size)
        return out


rnn = RNN()
rnn.to(device)
print(rnn)

optimizer = optim.Adam(rnn.parameters(), lr=0.01)
loss_fun = nn.CrossEntropyLoss()

for epoch in range(3):
    for step, (x, y) in enumerate(train_loader):
        x = x.view(-1, 28, 28)      # (batch, time_step, input_size)
        x = x.to(device)
        y = y.to(device)
        output = rnn(x)
        loss = loss_fun(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print('Epoch: {} Step: {} Loss: {}'.format(epoch, step, loss))
print('Finished Training')

print('Now Testing...')
# training accuracy
correct = 0
total = 0
with torch.no_grad():
    for data in train_loader:
        images, labels = data
        images = images.view(-1, 28, 28)
        images = images.to(device)
        labels = labels.to(device)
        outputs = rnn(images)
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
        images = images.view(-1, 28, 28)
        outputs = rnn(images)
        _, predict = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predict == labels).sum().item()
    print('Test Accuracy: %.2f %%' % (100 * correct / total))

