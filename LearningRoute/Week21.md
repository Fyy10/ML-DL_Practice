# 第21周周报

## 学习内容

- 数据处理中用到的一些包
- Image Caption 和 VQA 中 attention 的理解

## 学习收获

### Xshell

首先稍微学了一下Xshell的操作

Xshell的基本用法很简单，只需要新建会话，填写服务器地址，用户名和密码就可以自动登录了，用Xftp可以很方便地进行文件传输

用本地的虚拟机进行了一些测试，可以在虚拟机外通过Xshell登录到虚拟机里进行操作

### Regular Expression

在vim中进行字符串的查找和替换经常会用到正则表达式（包括Image Caption和VQA的代码中也有用到），于是看了一些简单的RE规则

[一个RE的练习网站](https://regexr.com/)

正则表达式及其含义

|字符|含义|
|---|---|
|\\|转义|
|.|匹配单个字符|
|\*|匹配任意次，可以是0次，如.\*匹配任意字符串|
|^|匹配首字符串|
|$|匹配尾字符串|
|[...]|匹配中括号中的某个字符，[^...]表示不匹配任意字符，中间可用`-`表示范围，如[A-Z]|
|\\{n, m\\}|匹配[n, m]次，\\{n\\}匹配n次，\\{n,\\}最小匹配次数为n|
|\\(\\)|定义一个匹配位置，在后部可以引用该位置。如\(ab\).*\1表示ab字符串包夹了一个任意字符串|
|\\n|引用已定义的位置，可以从\\1到\\9|
|+|匹配至少一次|
|?|匹配0次或1次|
|\||或|
|()|匹配括号内整个字符串|

特别地：

- [[:alnum:]]字符+数字
- [[:alpha:]]字符
- [[:digit:]]数字
- [[:lower:]]小写字符
- [[:upper:]]大写字符

vim的正则表达式规则略有不同

```vim
" 允许使用正则表达式
set magic
" 查看vim的正则表达式规则
:help magic
```

vim常用替换命令：

```vim
:%s/from/to/gc
```

### 数据处理使用的python包

下面简要地学习一下数据处理中用到的一些python包，由于详细学习十分花时间和精力，而且不是很有必要，所以这里只是学一下基本用法，细致具体的内容可以查doc或者在实际运用中慢慢学习和理解

#### h5py

[官网](https://www.h5py.org/)

hdf5二进制数据格式的python接口

hdf5是包含两种对象的容器，group（类似文件夹）和dataset（类似numpy array）

```python
import h5py

# open a hdf5 file
f = h5py.File('h5file.hdf5', 'r')
# check keys
list(f.keys())
dataset = f['mydata']
# dataset是hdf5的dataset，与numpy array类似
dataset.shape
dataset.dtype
```

group的操作与linux中访问文件路径的方式相同

```python
dataset = f['data/dataset']
```

#### tqdm

[官网](https://tqdm.github.io/)

用于显示进度条，只需要将可迭代的对象用tqdm包装

```python
from tqdm import tqdm

for i in tqdm(range(100)):
    pass
```

#### json

[官网](https://www.json.org/json-zh.html)

json是一种轻量级数据交换格式，易于人阅读和编写，也易于机器解析和生成

```python
import json

# 将python对象编码成json
data = [{'a': 1, 'b': 2}]
j = json.dumps(data)
print(j)
# 将json解码
text = json.loads(jsonData)
```

一般只对python的list和dict进行编码和解码

### Attention

#### Image Caption

在某一个time step中

输入为encoder_out和decoder_hidden

```python
# in Attention
def forward(self, encoder_out, decoder_hidden):
    # encoder_out: [batch_size, num_pixels, encoder_dim]
    # decoder_hidden: [batch_size, decoder_dim]
    att1 = self.encoder_att(encoder_out)
    # att1: [batch_size, num_pixels, attn_dim]
    # 用linear层将encoder_dim变换到attn_dim
    att2 = self.decoder_attn(decoder_hidden)
    # att2: [batch_size, attn_dim]
    # 用linear层将decoder_dim变换到attn_dim
    att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)
    # att: [batch_size, num_pixels]
    # 将att1和att2对应相加后，用relu激活，然后通过linear层将所有的attn_dim映射到一维
    alpha = self.softmax(att)
    # alpha: [batch_size, num_pixels]
    # 经过softmax得到alpha，表示各个pixel的权重
    # 给encoder_out加权
    attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
    # attention_weighted_encoding: [batch_size, encoder_dim]
    # 把encoder_out的所有pixel全部加起来(why?)
    # 猜测：用所有pixel加权后的和来表示某一个encoder_dim (filter) 的权重
    return attention_weighted_encoding, alpha
```

#### Visual Question Answering

下面两篇文章仍然有待研究：

1. [Stacked Attention Networks for Image Question Answering](https://arxiv.org/pdf/1511.02274.pdf)
2. [Show, Ask, Attend, and Answer: A Strong Baseline For Visual Question Answering](https://arxiv.org/abs/1704.03162)

## 疑问和困难

1. 看懂了Image Caption的attention实现过程，但是不明白为什么这样设计attention，为什么这样就体现了attention
2. 同样是把某一维的所有信息整合到一起，有的地方采用linear，有的地方是求和（尽管是特殊情况的linear）
3. 为什么含义完全不相同embedding与attention_weighted_encoding可以直接拼接后放入LSTMCell进行计算（尽管从shape上分析，代码没有问题）
4. vqa的attention仍然比较迷茫
