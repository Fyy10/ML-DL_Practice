# 第8周学习报告

## 学习内容

- matplotlib
- PyTorch-简单RNN的实现

## 学习收获

### matplotlib

画出简单的一元函数的图像，可以使用plot，用法与MATLAB类似

用`np.linspace()`生成一段区间上均匀分布的值

用`plt.figure()`定义一个图像窗口，`plt.plot()`画出(x, y)的曲线，`plt.show()`显示图像

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(1, 10, 100)
y = x**2 + 1
plt.figure()
plt.plot(x, y)
plt.show()
```

也可以一次性画多个函数图像再显示

```python
y1 = x**2 + 1
y2 = x
plt.figure(num=1)
plt.plot(x, y1, c='r')
plt.plot(x, y2, c='g')
plt.show()
```

坐标轴的设置

用`plt.xlim()`设置x轴范围，`plt.xlabel()`设置x轴名称，`plt.ylim()`设置y轴范围，`plt.ylabel()`设置y轴名称

用`plt.xticks()`和`plt.yticks()`设置坐标轴刻度

```python
new_ticks = np.linspace(0, 100, 11)
print(new_ticks)
plt.xticks(new_ticks)
plt.yticks([1, 2, 3, 4, 5], [r'$num_1$', r'$num_2$', r'$num_3$', r'$num_4$', r'$num_5$'])
plt.xlim(-1, 2)
plt.ylim(2, 3)
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
```

坐标轴位置的修改

`plt.gca()`获取坐标轴信息，`.spines`设置边框，`.set_color()`设置边框颜色，默认白色

`.xaxis.set_ticks_position`设置x轴刻度数字或名称的位置(`top`, `bottom`, `both`, `default`, `none`)

用`.spines`设置边框，`.set_position`设置边框位置(`outward`, `axes`, `data`)

```python
ax = plt.gca()
ax.spines['top'].set_color('red')
ax.spines['bottom'].set_position(('data', 0))
ax.spines['left'].set_position(('data', 0))
plt.show()
```

显示图例(`legend`)

```python
plt.figure()
l1, = plt.plot(x, y1, label='line 1')
l2, = plt.plot(x, y2, label='line 2')
print(l1, l2)
# plt.legend(loc='upper left')
plt.legend(handles=[l1, l2], labels=['line 1', 'line 2'], loc='best')
```

loc中可使用的参数如下：

```python
'best': 0
'upper right': 1
'upper left': 2
'lower left': 3
'lower right': 4
'right': 5
'center left': 6
'center right': 7
'lower center': 8
'upper center': 9
'center': 10
```

在画好的图像中进行标注(annotation):

- plt.annotate
- plt.text

```python
def fun(input_x):
    return 2 * input_x + 1


# some basic plotting
x = np.linspace(-3, 3, 100)
y = fun
plt.figure('Annotations')
plt.plot(x, y(x), label=r'$y = 2 x + 1$')
plt.legend(loc='best')
ax = plt.gca()
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['bottom'].set_position(('data', 0))
ax.spines['left'].set_position(('data', 0))
x0 = 1
y0 = y(x0)
plt.plot([x0, x0], [0, y0], 'k--', linewidth=2.5)
plt.scatter([x0], [y0], s=50, c='b')
# make annotations
plt.annotate(r'$2x+1=%s$' % y0, xy=(x0, y0), xycoords='data', xytext=(+30, -30), textcoords='offset points', fontsize=16, arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
# xy: position of point to annotate
# xycoords='data': choose the position based on the data
# xytext: the position relative to the point

# put text
plt.text(-4, 3, r'$This\ is\ a\ special\ point$', fontdict={'size': 16, 'color': 'r'})
plt.show()
```

使用`plt.scatter`绘制散点图

```python
# Scatter
plt.figure('Scatter')
n = 1024    # data size
x = np.random.randn(n)
y = np.random.randn(n)
T = np.arctan2(x, y)    # for color value
plt.scatter(x, y, c=T, alpha=0.5)
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.xticks(())  # ignore x ticks
plt.yticks(())  # ignore y ticks
plt.show()
```

使用`plt.bar`绘制柱状图

```python
# Bar
plt.figure('Bar')
n = 12
x = np.arange(n)
y1 = np.random.uniform(0.0, 1.0, n)
y2 = np.random.uniform(0.0, 1.0, n)

plt.bar(x, +y1, facecolor='#9999ff', edgecolor='white')
plt.bar(x, -y2, facecolor='#ff9999', edgecolor='white')

# ha: horizontal alignment
# va: vertical alignment
for xa, ya in zip(x, y1):
    plt.text(xa, ya + 0.05, '%.2f' % ya, ha='center', va='bottom')

for xa, ya in zip(x, y2):
    plt.text(xa, -ya - 0.05, '%.2f' % ya, ha='center', va='top')

plt.xlim(-1, n)
plt.xticks(())
plt.ylim(-1.25, 1.25)
plt.yticks(())

plt.show()
```

使用`plt.contour`绘制等高线

```python
# Contour
def f(x, y):
    return (1 - x / 2 + x**5 + y**3) * np.exp(-x**2 - y**2)


plt.figure('Contour')
n = 256
x = np.linspace(-3, 3, n)
y = np.linspace(-3, 3, n)
X, Y = np.meshgrid(x, y)
# contour filling
plt.contourf(X, Y, f(X, Y), 10, alpha=0.75, cmap=plt.cm.hot)
# contour
C = plt.contour(X, Y, f(X, Y), 10, colors='black', linewidth=0.5)
plt.clabel(C, inline=True, fontsize=10)
plt.xticks(())
plt.yticks(())
plt.show()
```

将numpy array显示为Image

```python
# Image
x = np.random.rand(3, 3)
print(x)
plt.imshow(x, cmap='bone', interpolation='nearest', origin='lower')
plt.colorbar(shrink=0.92)
plt.xticks(())
plt.yticks(())
plt.show()
```

如果绘制三维图形，则需要另外import Axes3D

```python
from mpl_toolkits.mplot3d import Axes3D
# plot 3D
fig = plt.figure()
ax = Axes3D(fig)
x = np.arange(-4, 4, 0.25)
y = np.arange(-4, 4, 0.25)
x, y = np.meshgrid(x, y)
r = np.sqrt(x**2 + y**2)
z = np.sin(r)
# rstride: row
# cstride: column
ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
ax.contourf(x, y, z, zdir='z', offset=-2, cmap=plt.get_cmap('rainbow'))
ax.set_zlim(-2, 2)
plt.show()
```

### RNN

数据库的读取和预处理在之前已经学过，假设已经预载好了train data和test data。

之前实现了CNN，现在实现RNN只需在class里进行一些修改（将CNN层换成RNN层）

```python
# Classification
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
# Regression
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(
            input_size=1,
            hidden_size=32,
            num_layers=2,
            batch_first=True
        )
        self.out = nn.Linear(32, 1)

    def forward(self, x, h_state):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, output_size/hidden_size)
        r_out, h_state = self.rnn(x, h_state)
        outs = []
        for time_step in range(r_out.size(1)):
            outs.append(self.out(r_out[:, time_step, :]))
        return torch.stack(outs, dim=1), h_state
```

训练与测试步骤与CNN相同。

## 疑问和困难

1. 自己编写的网络不够灵活，缺乏构造神经网络的经验
2. 尽管能看懂教程里的示例代码，但是完全不看示例和参考资料就写不出程序了
3. 实际处理的问题必然不会像示例那么简单，数据的收集和处理还需要多加练习
