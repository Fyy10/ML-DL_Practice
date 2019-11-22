import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input channel, 6 output channels, 3*3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)   # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can simply specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    # backward function has been defined automatically using autograd

    def num_flat_features(self, x):
        size = x.size()[1:]     # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def main():
    net = Net()
    input = torch.randn(1, 1, 32, 32)
    output = net(input)
    print(output)
    target = torch.randn(10)
    print(target)
    target = target.view(1, -1)     # make it the same shape as output
    print(target)
    criterion = nn.MSELoss()
    loss = criterion(output, target)
    print(loss)
    print(loss.grad_fn)
    print(loss.grad_fn.next_functions[0][0])
    print(loss.grad_fn.next_functions[0][0].next_functions[0][0])
    net.zero_grad()
    print('conv1.bias.grad before backward', net.conv1.bias.grad)
    loss.backward()
    print('conv1.bias.grad after backward', net.conv1.bias.grad)
    learning_rate = 0.01
    for f in net.parameters():
        f.data.sub_(f.grad.data * learning_rate)
    # create an optimizer
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    # in the training loop
    optimizer.zero_grad()       # zero the gradient buffers?
    output = net(input)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()        # update weights


if __name__ == '__main__':
    main()
