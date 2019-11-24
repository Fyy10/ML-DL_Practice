# 第6周学习报告

## 学习内容

- PyTorch基础

## 学习收获

### Tensor

相当于numpy的ndarray:

```python
from __future__ import print_function
import torch
# create an uninitialized matrix
x = torch.empty(5, 3)
# construct an randomly initialized matrix
x = torch.rand(5, 3)
# construct an matrix filled with zeros of dtype long
x = torch.rand(5, 3, dtype=torch.long)
# construct tensor from data
x = torch.tensor([5.5, 3])
# 利用已有的tensor产生新的tensor
x = x.new_ones(5, 3, dtype=torch.double)
x = torch.randn_like(x, dtype=torch.float)
# get size
print(x.size())
```

Tensor Operations:

```python
x = torch.rand(5, 3)
y = torch.rand(5, 3)
print(x + y)
print(torch.add(x, y))
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)
# add x to y
y.add_(x)
print(y)
# 后面带下划线的函数会改变tensor的值，如x.copy_(y), x.t_()
# resize/reshape the tensor: torch.view()
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)
# 单元素的tensor可以转换为python的数值
x = torch.rand(1)
print(x)
print(x.item())
```

将torch tensor转化为numpy array：

```python
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)
# 注意a和b会共享相同的内存空间（如果tensor在CPU上），改变一个将同时改变另一个
```

将numpy array转化为torch tensor：

```python
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
print(a)
print(b)
# a和b仍然共享相同的内存空间
```

CPU上所有除了CharTensor以外的tensor都可以与numpy相互转换。

CUDA tensors：

通过`.to()`方法可以将tensor转移到任何设备(device)上。

```python
# run the cell only if CUDA device is available
# 使用torch.device进行tensor在GPU上的进出
if torch.cuda.is_available():
    device = torch.device("cuda")
    y = torch.ones_like(x, device=device)
    x = x.to(device)
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))
```

### Autograd: Automatic Differentiation

当tensor的`.requires_grad=True`时，它将会记录在上面的所有操作，所有计算完成后，调用`.backward()`进行梯度的自动计算，并被放在`.grad`中。

可以调用`.detach()`来停止tensor记录操作。

所有由`Function`创建的`Tensor`都会有一个`.grad_fn`作为`Function`的引用。

```python
x = torch.ones(2, 2, requires_grad=True)
y = x + 2
print(y.grad_fn)
z = y * y + 3
out = z.mean()
print(z, out)
```

使用`.requires_grad_()`来改变tensor的`.requires_grad`(默认是False)

```python
a = torch.randn(2, 2)
a = ((a * 3) / (a + 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)
```

### 构建一个NN

1. 定义含可变参数的NN
2. Iterate over a dataset of inputs
3. 计算NN的输出
4. 计算Loss
5. Propagate gradients back into network's parameters
6. Update weights of NN

Define a network

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input channel, 6 output channels, 3*3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b (仿射变换)
        self.fc1 = nn.Linear(16*6*6, 120)   # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can simply specify a single number
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]     # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)
```

`nn.Conv2d`接收一个4D tensor (sample \* channel \* height \* width)

backward函数已经被自动定义了（通过autograd）

模型的可学习参数可通过`net.parameters()`返回

```python
params = list(net.parameters())
print(len(params))
print(params[0].size())		# conv1's weight
```

获取NN的输出：

```python
net = Net()
input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)
```

计算Loss（使用nn package）

```python
output = net(input)
target = torch.randn(10)
target = target.view(1, -1)     # 使target与output的shape相同
criterion = nn.MSELoss()
loss = criterion(output, target)
print(loss)
```

利用Loss进行反向传播（Back propagation）：

```python
net.zero_grad()
print(net.conv1.bias.grad)
loss.backward()
print(net.conv1.bias.grad)
```

注：反向传播并不会更新参数，只是把当前的梯度信息保存在`.grad`中，真正更新参数需要调用`optimizer.step()`

更新Weights：

```python
# 简单的Gradient Descent (stochastic gradient descent)
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)
```

```python
# 自选optimizer
import torch.optim as optim
# create an optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)
# in the training loop:
optimizer.zero_grad()
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()
```

### Training a Classifier

Packages for loading data on images: Pillow, OpenCV

`torchvision.datasets`和`torchvision.utils.data.DataLoader`

#### Steps of training a classifier

1. Load and normalize data from datasets using `torchvision`
2. Define a CNN
3. Define a loss function
4. Train the network on the training data
5. Test the network on the test data

Load and normalize data:

```python
import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
```

Train the network

```python
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get input, data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
print('Finished Training')
```

保存训练好的network

```python
# save trained model
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)
```

Load and test the network

```python
# load the saved network
net = Net()
PATH = './cifar_net.pth'
net.load_state_dict(PATH)
# test all
correct = 0     # number of correct predicts
total = 0       # number of total
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predict = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predict == labels).sum().item()
print('Accuracy: %d %%' % (100 * correct / total))
```

## 疑问和困难

1. 网络训练的速度很慢，很久才能出结果，采用了CUDA加速但是与CPU直接计算速度差异并不大
2. 训练好的网络的准确率并不理想
3. 对于`.to(device)`的使用仍然一知半解（可能因此导致训练速度慢）
4. 对于`forward`, `backward`, `optimizer`的理解和运用还不到位
5. 网络的训练仅仅是基于数据的学习，而真正有效的学习过程应该是基于规则和示例（少量数据）的学习
