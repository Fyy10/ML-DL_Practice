# 第10周周报

## 本周计划

- PyTorch实现自动写唐诗的程序
- 用不同的神经网络结构训练，挑选效果较好的

## 实现方法

### 自动写唐诗

为了避免在收集数据方面花太多时间，这里直接用网上现成的，已经处理好的数据`tang.npz`进行训练

测试一下数据的内容：

```python
# This code is only used for testing the data
import numpy as np

DataSet = np.load('tang.npz', allow_pickle=True)
data = DataSet['data']
print(data.shape)
print(type(data))
ix2word = DataSet['ix2word'].item()
print(type(ix2word))
word2ix = DataSet['word2ix'].item()
print(type(word2ix))
```

输出如下：

```python
(57580, 125)
<class 'numpy.ndarray'>
<class 'dict'>
<class 'dict'>
```

可知数据里已有字符和数字相互映射的字典

构造DataLoader：

```python
data = torch.from_numpy(data)
data_loader = DataLoader(data, batch_size=Config.batch_size, shuffle=True, num_workers=Config.num_workers)
```

训练模型（在训练Seq2Seq模型时遇到了一些问题，先从较为简单的CharRNN开始）

#### CharRNN

将一些常用的常量/Hyper parameters放在`Config`类中，方便使用

```python
# CharRNN model
class CharRNN(nn.Module):
    def __init__(self, voc_size, embedding_dim, hidden_dim):
        super(CharRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(voc_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=Config.num_layers)
        self.fc = nn.Linear(hidden_dim, voc_size)

    def forward(self, in_seq, hidden=None):
        seq_len, batch_size = in_seq.size()
        # in_seq: [seq_len, batch_size]
        # set hidden to all 0 if hidden is None
        if hidden is None:
            h0 = in_seq.data.new(Config.num_layers, batch_size, self.hidden_dim).fill_(0).float()
            c0 = in_seq.data.new(Config.num_layers, batch_size, self.hidden_dim).fill_(0).float()
        else:
            h0, c0 = hidden
        embeds = self.embeddings(in_seq)
        # [seq_len, batch_size, embedding_dim]
        output, hidden = self.lstm(embeds, (h0, c0))
        # output: [seq_len, batch_size, hidden_dim]
        # hn/cn: [num_layers, batch_size, hidden_dim]
        output = self.fc(output.view(seq_len * batch_size, -1))
        # output: [seq_len * batch_size, voc_size]
        return output, hidden
```

```python
# enable cuda device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# define model (CharRNN)
model = CharRNN(len(word2ix), Config.embedding_dim, Config.hidden_dim)
model = model.to(device)

# optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# get formal trained model
if Config.model_path:
    model.load_state_dict(torch.load(Config.model_path))

# train
for epoch in range(Config.epoch):
    for step, data_ in enumerate(data_loader):
        # data_: [batch_size, seq_len]
        data_ = data_.long(),transpose(1, 0).contiguous()   # data_: [seq_len, batch_size]
        data_ = data_.to(device)

        optimizer.zero_grad()
        # input_ = 0..n-2 / target = 1..n-1
        input_, target = data_[:-1, :], data_[1:, :]
        output, _ = model(input_)
        # output: [seq_len * batch_size, voc_size]
        # target: [seq_len, batch_size]
        loss = criterion(output, target.view(-1))
        loss.backward()
        if step % 100 == 0:
            print('Epoch: {}, Step: {}, Loss: {}'.format(epoch, step, loss.item()))

    # save model
    torch.save(model.state_dict(), '%s_%s.pth' % (Config.model_prefix, epoch))

# save model
torch.save(model.state_dict(), Config.model_path)
print('Finished Training')
```

经过一段时间的训练后，得到训练结果`CheckPoints/tang_final.pth`，再读取训练结果生成诗句

生成诗句：

```python
# prefix_words表示诗的前缀词，用于控制诗的感情基调（设想是这样的），默认为None
# given the first few words
def generate(model, start_words, ix2word, word2ix, prefix_words=None):
    # to store the whole poetry
    poetry = list(start_words)
    len_start_words = len(start_words)
    # the first input word should be <START>
    in_seq = torch.Tensor([word2ix['<START>']]).view(1, 1).long()
    # in_seq: [1, 1]
    in_seq = in_seq.to(Config.device)
    hidden = None

    # if given prefix_words
    if prefix_words:
        for word in prefix_words:
            output, hidden = model(in_seq, hidden)
            in_seq = in_seq.data.new([word2ix[word]]).view(1, 1)

    # 前缀词的最后一个词作为输入，hidden是前缀词生成的hidden
    # start_words为诗的开头
    # 限制诗的最大长度为max_gen_len
    for i in range(Config.max_gen_len):
        # hidden contains information of start_words
        output, hidden = model(in_seq, hidden)
        # start_words -> poetry
        if i < len_start_words:
            w = poetry[i]
            in_seq = in_seq.data.new([word2ix[w]]).view(1, 1)
        else:
            # get top index of word (the word with maximum probability)
            top_index = output.data[0].topk(1)[1][0].item()
            w = ix2word[top_index]
            poetry.append(w)
            in_seq = in_seq.data.new([top_index]).view(1, 1)
        # End of poem
        if w == '<EOP>':
            del poetry[-1]
            break
        # format the poem
        if w == '。' or w == '？':
            poetry.append('\n')
    # 去掉多余的换行（print自带换行）
    if poetry[-1] == '\n':
        del poetry[-1]
    return poetry
# 返回的poetry是存储了诗句的list
```

测试一下生成的结果：

```python
def write_poem():
    print('Loading model...')
    dataset = np.load(Config.data_path, allow_pickle=True)
    ix2word = dataset['ix2word'].item()
    word2ix = dataset['word2ix'].item()
    model = CharRNN(len(ix2word), Config.embedding_dim, Config.hidden_dim)
    model.load_state_dict(torch.load(Config.model_path, Config.device))
    print('Done!')
    while True:
        start_words = str(input())
        gen_poetry = ''.join(generate(model, start_words, ix2word, word2ix, Config.prefix_words))
        print(gen_poetry)
```

生成的诗句如下（片段）：

> 随风潜入夜\
> 随风潜入夜，密景景无穷。\
> 泫蔼无人夜，清和不可闻。\
> 乍疑萤上月，稍与露中光。\
> 乍想疑红烛，先亏照绿杨。\
> 叶枝犹有色，柳絮不胜霜。\
> 露重花犹落，风轻叶未干。\
> 剪黄犹带萼，剪䌽不成香。\
> 自昔承明盛，今来乐所从。\
> \
> 接天绿叶无穷碧\
> 接天绿叶无穷碧，长风吹浪如流水。\
> 初疑秋风吹落日，复入巖下烧茶竹。\
> 珪璋特下尤不欺，此时自古称不得。\
> 古来世业皆俊才，李生稅与何人来。\
> 我有一言不相识，何人得荅长生诀。\
> 昔时为客荐芳菲，君看此花不成死。\
> 君今此曲君莫言，君今此曲何所为。

测试的时候发现，很多时候并不能自动生成诗的结尾`<EOP>`（目前未找到原因）

#### 回到Seq2Seq的训练

之前对Seq2Seq的context vector理解有误，context vector实际上是由encoder的hidden state加权平均得到的，而不是encoder的output（此处仍有实验的价值）

对于context vector的来源，网上的资料说法并不统一，下面尝试通过不同途径得到context vector来进行测试

##### 将Encoder的output作为context vector

Encoder输入的数据size为(seq_len, batch_size)

embedding后(seq_len, batch_size, embedding_dim)

lstm后size为(seq_len, batch_size, hidden_dim)作为output

而Decoder接受的size为(seq_len, batch_size)，考虑用一个Linear层将hidden_dim压缩到1维

```python
class Seq2Seq(nn.Module):   # still some problems
    def __init__(self, voc_size, hidden_dim):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder.EncoderRNN(input_size=voc_size,
                                          hidden_size=hidden_dim)
        # try to map hidden_dim to 1 dim (may be problem here)
        self.fc = nn.Linear(hidden_dim, 1)
        self.decoder = Decoder.DecoderRNN(hidden_size=hidden_dim,
                                          output_size=voc_size)

    def forward(self, in_seq, hidden=None):
        # last hidden state of encoder used as the context vector (how? & why?)
        # in_seq: [seq_len, batch_size]
        output, (hidden, cell) = self.encoder(in_seq, hidden)
        # output: [seq_len, batch_size, hidden_dim]
        # h_n/c_n: [num_layers, batch_size, hidden_dim]
        output = self.fc(output).long()
        # output: [seq_len, batch_size, 1]
        output = output[:, :, 0]
        # output: [seq_len, batch_size]
        context = output
        output, hidden = self.decoder(context, (hidden, cell))
        return output, hidden
```

此时的Seq2Seq模型的输入输出的形式可以认为跟前面的CharRNN是一样的，所以训练和生成代码不变

训练一个Epoch后（时间有限），测试结果如下：

> 横看成岭侧成峰\
> 横看成岭侧成峰\<START\>\<START\>\<START\>\<START\>\<START\>\<START\>\<START\>\<START\>\<START>\<START\>\<START\>\<EOP>\<START\>\<START\>\<START\>\<START\>\<START\>\<START\>\<START\>\<EOP\>\<START\>\<START\>\<START\>\<START\>\<START\>\<START\>\<START\>\<EOP\>\<START\>\<START\>\<START\>\<START\>\<START\>\<START\>\<START\>\<START\>\<EOP\>\<START\>\<START\>\<START\>\<START\>\<START\>\<START\>\<START\>\<START\>\<EOP>

直接生成会出现一行`<START>`后结束，如果不去掉`<EOP>`则得到上述结果，目前还未找到原因。

## 当前困难/疑惑

1. 训练的速度很慢，即便使用GPU加速，训练一个epoch仍需20min左右，难以在较短的时间里确定模型是否正确
2. 尽管使用了Encoder-Decoder结构，但感觉上跟Seq2Seq的本意有些不同
3. 对于context vector的来源，网上资料有不同的说法（[hidden state加权平均(或final hidden state)作为Decoder的hidden state](https://medium.com/@bgg/seq2seq-pay-attention-to-self-attention-part-1-d332e85e9aad)/[hidden state加权平均(或final hidden state)作为Decoder的input sequence](https://towardsdatascience.com/attn-illustrated-attention-5ec4ad276ee3)/[target](https://medium.com/analytics-vidhya/intuitive-understanding-of-seq2seq-model-attention-mechanism-in-deep-learning-1c1c24aace1e)）
