# 第9周周报

## 本周计划

- RNN/LSTM/seq2seq 学习
- 设计一个用于机器自动写唐诗/宋词的模型

## 分析与设计

### RNN/LSTM

#### RNN

RNN每次接受一个输入和上一次的hidden state，输出一个output和hidden state（这个hidden state将作为下一次的输入），如此反复即可由一串输入得到一串输出，因为有了传递hidden state的过程，所以RNN的输出不但与当前的输入有关，还与前面的输入有关（信息藏在hidden state中）。

关于RNN的实现，PyTorch里已经有现成可用的模块`torch.nn.RNN`，RNN中的所有参数，包括产生output的参数和产生hidden state的参数都可以通过数据训练来确定。

#### LSTM: Long Short-Term Memory

最普通的RNN是不可用的，因为经过多次输入输出后，会出现梯度爆炸/梯度消失的现象，参数的传递在多次输入输出的过程中被指数级放大/缩小，所以前面的信息很难保留到后面。

LSTM在RNN的基础上，除了要输出hidden state以外，还输出一个cell state，用于确定下一阶段的输入是忘掉还是记住，对输入进行选择性的记忆，因此可以用于输入序列更长的情况。

在PyTorch中，可以用`torch.nn.LSTM`来实现一个LSTM的网络，产生cell state和hidden state的参数都可以通过训练来确定。

### seq2seq

seq2seq是encoder-decoder结构的网络，输入一个序列，输出一个序列

encoder将不定长输入序列编码成定长的状态向量 $c$ ，decoder再将编码后的向量解码成不定长的输出序列

encoder和decoder均采用RNN/LSTM，便于处理序列性的信息

整个seq2seq完成了从不定长序列到不定长序列的转换，输入输出的长度可变

构建整个encoder-decoder结构有两种方式：

1. 将 $c$ 作为decoder每次循环的输入
2. 仅将 $c$ 作为decoder第一次的输入

### 自动写诗模型的设计

设计一个能自动写唐诗/宋词的神经网络，采用seq2seq模型（encoder-decoder架构），输入诗的开头，让机器自动写完整首诗。

网络结构如下：

诗的开头 $\rightarrow Encoder \rightarrow State\ Vector \rightarrow Decoder \rightarrow$ 完整的诗

需要对数据进行预处理，加入特殊字符用于判断开始和结束

#### Encoder结构

1. embedding层：将输入序列转化为向量
2. LSTM层：输入向量，输出状态向量 $c$

#### Decoder结构

1. embedding层：将状态向量映射为另一向量（具体作用还不明，参考了PyTorch的示例程序）
2. 激活函数：可以用relu和tanh等，具体用哪个根据实际情况来确定
3. LSTM层：将向量Decode为一个输出序列

#### 模型的初步实现

参照PyTorch关于RNN和LSTM的示例程序初步实现了Encoder和Decoder

关于数据的获取和预处理暂时放到后面

##### Encoder的实现

按照前面的设计写出了Encoder的大致框架，其具体结构根据后续情况再修改

```python
import torch
import torch.nn as nn


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)

    def forward(self, input_seq, hidden):
        output = self.embedding(input_seq).view(1, 1, -1)
        output, hidden = self.lstm(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)
```

##### Decoder的实现

```python
import torch
import torch.nn as nn


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_seq, hidden):
        output = self.embedding(input_seq).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.lstm(output, hidden)
        output = self.softmax(self.fc(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)
```

再将Encoder和Decoder合并到一个模型中，方便训练（目前的设想）

```python
import torch.nn as nn
import Decoder
import Encoder


class Seq2Seq(nn.Module):
    def __init__(self):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder.EncoderRNN(input_size=INPUT_SIZE,
                                          hidden_size=HIDDEN_SIZE)
        self.decoder = Decoder.DecoderRNN(hidden_size=HIDDEN_SIZE,
                                          output_size=OUTPUT_SIZE)

    def forward(self, in_seq, hidden):
        state, hidden = self.encoder(in_seq, hidden)
        output = self.decoder(state, hidden)
        return output
```

现在网络的基本结构已经确定，接下来将进行数据的收集和处理。

## 当前困难/困惑

1. 还没有加入Attention的部分
2. 如何确定诗的结束和格式
3. 通常情况下输入的序列很短，为什么不直接使用Embedding的结果作为state vector（需要实现代码后做实验确定）
4. 为什么seq2seq的结构适合于输入短，输出长的情况
5. 为什么Decoder中还需要embedding而不是直接使用状态向量作为LSTM的输入
6. 如何将Decoder中LSTM输出的output和hidden state转化为诗词
