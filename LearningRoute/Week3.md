# 第3周学习报告

## 学习内容

- python模块的使用
- 机器学习入门

## 学习收获

### python

#### 使用模块

当目录中存在\_\_init\_\_.py时，这个目录会被当作包处理，\_\_init\_\_.py本身也是模块，模块名就是目录名。访问包中的模块则类似C++中对象加.的方式。

导入模块

```python
import sys
print sys.argv

try:
    import cStringIO as StringIO
except ImportError:
    import StringIO
```

安装第三方模块

```shell
pip install scikit-learn
```

#### 面向对象特性

```python
class MyClass(object): #括号里的是继承的对象，如果没有就填object
#object是所有类最终都会继承的类
    def __init__(self, num1, num2): #类似构造函数，必须有self，相当于C++中this指针指向的对象
        self.num1 = 1
        self.num2 = 2

    def display(self): #必须有self
        print num1, num2
```

python是动态语言，允许对一个类的实例绑定任何数据，因此同一个类的两个实例所拥有的变量可以不同。

在变量名前面加两个下划线`_`说明变量是私有成员，类外不可访问。

双下划线开头和结尾的变量不是私有变量。

继承的概念与C++相同。

多态：开闭原则

- 对扩展开放：允许新增子类
- 对修改封闭：不需要修改依赖父类型的函数

使用type()获取对象类型，用isinstance()确定变量是否在类的继承链上（只看祖先）。

用dir()获取对象的所有属性和方法，返回一个包含字符串的list。

注意以下的写法：

```python
class MyClass(object):
    def __display__():
        pass
    def other():
        pass

a = MyClass()
a.__display__()
display(a) #如果用双下划线作为方法名的开头和结尾，则可以把方法当作普通函数使用
a.other()
```

#### 文件操作

读写文件

```python
#读文件
f = open('filename.txt', 'r') #以只读方式打开文本文件
f.read()
f.close()
f = open('filename.bin', 'rb') #以只读方式打开二进制文件
f.close()
#非ASCII编码的文本文件只能通过二进制方式打开再解码
f = open('filename.txt', 'rb')
f.read().decode('gbk')
f.close()

#写文件
f = open('filename.txt', 'w') #以写入方式打开文本文件
f.write('HaHa')
f.close() #一定要有close，只有当文件被close了，才能保证内存里的内容完全被写入到硬盘里
```

#### python小结

至此python基础部分的学习暂时告一段落了，后面的面向对象高级编程、图形界面和网络编程等内容等用到的时候再根据需要去看。

### 机器学习

#### Lecture 0

人工智能（目标） $\rightarrow$ 机器学习（手段） $\rightarrow$ 深度学习（分支）

生物和机器的行为：

- 先天（人为设定）:Write the program for learning
- 后天（机器学习）:Looking for a function from data

通过数据训练寻找 $\hat{f}$ ，$\hat{f}(\text{Information}) = \text{Result}$ 。$\hat{f}$ 表示真实的函数，$f^*$ 表示通过训练找到的函数。

Steps of supervised learning:

1. Modeling $\rightarrow$ Define a set of functions
2. The goodness of function $\rightarrow$ Loss function $L$
3. Pick the best function $\rightarrow f^*$

Learning Map:

- Supervised Learning: 大量data
- Semi-supervised Learning: 较少labeled data
- Transfer Learning: 少量data和大量无关data
- Unsupervised Learning: 无labeled data
- Reinforcement Learning: 不固定输出，只给输出打分(Learn from critics)
- Structured Learning $\leftarrow$ May be the majority of real cases

输出的Classification:

- Binary Classification: Yes/No
- Multi-class Classification: Class 1, Class 2, ...

#### Lecture 1 Regression

符号的表示

用 $x_i$ 的下标来表示一个输入的某一个分量，用 $x^i$ 的上标来表示一个完整的object编号，即

$$
x^i = (x_1, x_2, \cdots , x_n)
$$

$$
\hat{f}(x^i) = \hat{y}^i
$$

带^表示是真实值（真实数据）。

Step 1: Modeling

以Linear model为例：

$$
y = b + \sum w_i x_i
$$

其中 $b, w_1, w_2, \cdots$ 是function的parameter，$b$ 为bias，$w_i$ 为weight，由一组参数来确定一个函数，不同的参数构成一个function set。模型越复杂，参数越多，function set就越大，就越可能包含最优的那个函数。

为了增大function set，可以增加新的 $x_i$ 或者增加非线性项（次方项）。

Step 2: Goodness of function

自定义一个Loss function $L(b, w)$ 来衡量一个函数有多不好。可以借用统计学中方差的概念来当作Loss function（有一点小区别）。

$$
L(b, w) = \sum_{k=1}^n (\hat{y}^k - (b + wx^k))^2
$$

$L(b, w)$ 的可视化操作：$L(b, w)$ 是一个二元函数，可以在坐标系中画出来，形象地表示出 $L(b, w)$ 在每一处的值。w-b图中的一个点则表示一个回归方程。

Step 3: Pick the best function - Gradient Descent

原问题是一个最优化问题：

$$
f^* = \text{args} \min_f(L(f))
$$

很自然想到沿着梯度反方向可以很快找到局部最优解。

Gradient Descent

1. 随机选取一个点 $(b^0, w^0)$
2. 计算梯度 $\nabla L = (\frac{\partial L}{\partial b}, \frac{\partial L}{\partial w})$
3. 更新 $b, w$ : $b^i = b^{i-1} - \eta \frac{\partial L}{\partial b}|_{(b,w)=(b^{i-1},w^{i-1})}$ ，$w^i$ 同理

$\eta$ 是learning rate，用于控制步长。 $\eta$ 固定时，梯度越大，则跨度越大。

> 注意可能会出现对training data符合很好，但是与testing data偏差较大的情况，称为Over fitting

用 $\text{error} = \sum |\hat{y}^i - y^i|$ 来反映与testing data的偏差。error过大则需重新设计model。

复杂的model可以降低training data的error rate，但是对于testing data却不一定（可能出现over fitting）

有可能设计的model(set of function)里面并没有包含正确的 $f\rightarrow$ Redesign the model

Regularization - Redefine the loss function

$$
L = \sum_i(\hat{y}^i - (b + \sum w_i x_i))^2 + \lambda \sum(w_i)^2
$$

考虑到更平滑的loss function会更好（$w_i$ 较小），使得单一分量对结果产生的影响更小，output对输入的变化更不敏感（受noises的影响较小）。

Select $\lambda$ to obtain the best model

Conclusion:

Gradient Descent

Over fitting & Regularization

Validation

Gradient Descent的实际运用：

```python
import numpy as np
import matplotlib.pyplot as plt

x_data = [ 338., 333., 328., 207., 226., 25., 179., 60., 208., 606.]
y_data = [ 640., 633., 619., 393., 428., 27., 193., 66., 226., 1591.]
# ydata = b + w * xdata

x = np.arange(-200, -100, 1)	#bias
y = np.arange(-5, 5, 0.1)		#weight
z = np.zeros((len(x), len(y)))
X, Y = np.meshgrid(x, y)
for i in range(len(x)):
	for j in range(len(y)):
		b = x[i]
		w = y[j]
		z[j][i] = 0
		for n in range(len(x_data)):
			z[j][i] = z[j][i] + (y_data[n] - b - w*x_data[n])**2
		z[j][i] = z[j][i]/len(x_data)

b = -120
w = -4

lr = 1
iteration = 100000

b_history = [b]
w_history = [w]

lr_b = 0
lr_w = 0

for i in range(iteration):
	b_grad = 0
	w_grad = 0
	for n in range(len(x_data)):
		b_grad = b_grad - 2.0*(y_data[n] - b - w*x_data[n])*1.0
		w_grad = w_grad - 2.0*(y_data[n] - b - w*x_data[n])*x_data[n]
	b_grad = b_grad / len(x_data)
	w_grad = w_grad / len(x_data)

	lr_b = lr_b + b_grad ** 2
	lr_w = lr_w + w_grad ** 2

	b = b - lr/np.sqrt(lr_b) * b_grad
	w = w - lr/np.sqrt(lr_w) * w_grad

	b_history.append(b)
	w_history.append(w)

plt.contourf(x, y, z, 50, alpha=0.5, cmap=plt.get_cmap('jet'))
plt.plot(b_history, w_history, 'o-', ms=3, lw=1.5, color='black')
plt.xlim(-200, -100)
plt.ylim(-5, 5)
plt.xlabel(r'$b$', fontsize=16)
plt.ylabel(r'$w$', fontsize=16)
plt.show()
print b, w
```

#### Lecture 2 Error

Error due to:

- bias
- variance

Estimator:

$\hat{y} = \hat{f}(x)$ , training data $\rightarrow f^*$ ，则 $f^*$ 是 $\hat{f}$ 的一个estimator

假设 $X \sim N(\mu, \sigma^2)$ : $\text{mean} = \mu \quad \text{variance} = \sigma^2$。

Estimator: N sample points: { $x^1, x^2, \cdots , x^N$ }

$m = \frac{1}{N}\sum x_i \neq \mu \quad E(m) = E(\frac{1}{N}\sum x_i) = \frac{1}{N}\sum E(x^i) = \mu$

$E(m) = \mu$ : unbiased

$var[m] = \frac{\sigma^2}{N}$ (depends on the number of samples)

$s^2 = \frac{1}{N}\sum(x^i - m)^2 \quad E(s^2) = \frac{N-1}{N}\sigma^2 \neq \sigma^2$ biased estimator

$E(f^*) = \bar{f}$ , $\bar{f}$ 和 $f^*$ 的偏差代表了variance，$\bar{f}$ 和 $\hat{f}$ 的偏差代表了bias

将data分为两部分，分别作为training data和testing data，每次试验选取不同的training data，得到多个 $f^*$。

Same model, different training data $\Rightarrow$ diff $f^*$

Simple model $\rightarrow$ low variance & high bias (vise versa) $\rightarrow$ less influenced by the sampled data

但是function set更小，bias偏大

Variance大：over fitting

- 能fit training data
- 不能fit testing data

处理方法：

1. 增加data（有效，不实际）
2. generate假的training data（自己制作data）
3. Regularization (smoothen) 相当于调整function space，可能增大bias

Bias大：under fitting

- 不能fit training data

处理方法：

- 增加模型复杂度（增加数据量并没有用，$\hat{f}$ 不在function set里）

Model Selection: trade-off between bias and variance

不能通过手上的testing set的error rate来决定采用哪个model。

N-fold Cross Validation: 将training data分割成n份，每次任取一份作为validation来通过error rate评价模型的好坏（validation不是testing set）

## 疑问和困难

1. 缺乏实践的经验，能够理解理论，但是把理论知识转化为代码比较困难
2. 没有找到合适的示例代码，希望能有一些具体的代码供参考和学习
