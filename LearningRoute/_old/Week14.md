# 第14周周报

## 本周计划

- 给seq2seq模型加入attention机制
- cs231n视频

## 学习内容

### Attention

#### Attention机制

通过一个attention层计算出attention weight，求出encoder所有输出的加权平均作为decoder的输入，使得decoder每次只关注encoder输出的一部分，即decoder会选择性地使用encoder的输出

attention weight可以通过上一次的hidden state和input的embedding计算得到，可以使用一个Linear层来计算attention weight，lstm的输入则由attention weight和encoder output共同得到

attention decoder原理图：

![attn](Week14_attn.png)

#### Attention Decoder

```python
class AttnDecoder(nn.Module):
    # 假设输入的最大长度为max_length，即为attention weight的长度
    def __init__(self, hidden_size, output_size, max_length=Config.max_length):
        super(AttnDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        # hidden state + encoder output
        self.attn = nn.Linear(2 * self.hidden_size, self.max_length)
        # encoder output(with attention) + embedding
        self.attn_combine = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, inp, hidden, encoder_outputs):
        embedded = self.embedding(inp).view(1, 1, -1)

        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.lstm(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=Config.device)
```

给诗词的seq2seq模型加入attention之后，训练速度更慢了，于是没有训练完。代码在逻辑上目前没有发现问题，如果使用pytorch tutorial中提供的英法翻译的数据进行训练，结果比未加入attention时略好（不是很明显）。

### cs231n

#### Lecture 1: CNN for Visual Recognition

Computer vision

Input Image $\rightarrow$ Primal Sketch $\rightarrow$ 2 1/2-D sketch $\rightarrow$ 3D model representation

Image Segmentation

#### Lecture 2: Image Classification

Image Classification

Challenges

Data-Driven approach

1. collect dataset of image and labels
2. ML -> classifier
3. Evaluate (new images)

```python
def classifier(image):
    # magic (model)
    return class_of_image


def train(image):
    # ML memorize data/label
    return model


def predict(model, test_image):
    # model predict (predict label of the most similar training image)
    return test_labels
```

Distance Metric to compare images

L1 distance: $d_1(I_1, I_2) = \sum\limits_p|I_1^p - I_2^p|$

test image - training image = pixel-wise absolute value

Manhattan distance

L2 distance: $d_2(I_1, I_2) = \sqrt{\sum\limits_p (I_1^p - I_2^p)^2}$

Euclidean distance

k-nearest neighbor (KNN):

L1 distance: boundary tend to follow coordinate axes

Hyper parameters: choices about the algorithm that we set rather than learn (very problem-dependent)

Setting hyperparameters:

Idea 1: Choose one that works best on the data

BAD: k=1 always works perfectly on training data

Idea 2: data -> training + testing

Choose hyperparameters that work best on test data

BAD: No idea how it will perform on new data

Idea 3: data -> train + validation + test

Choose one that work best on validation and evaluate on test (Better!)

Use the test data at last

Cross Validation (Idea 4)

data -> folds + test

try each fold as validation and average the results

Usually for small datasets, but not used too frequently in deep learning

KNN on images is never used

1. very slow
2. Distance metrics on pixels are not informative

Curse of dimensionality

Image Classification:

training set of images and labels

predict labels on the test set

KNN: predict based on nearest training examples

Linear Classification (simple but important)

Parametric Approach

image $\rightarrow f(x, W) \rightarrow$ 1/0 numbers of giving class scores

$f(x, W) = Wx + b$

Linear Classifier is only learning ***one*** template for each class

Define a (linear) score function

#### Lecture 3: Loss Functions and Optimization

Challenges of recognition

Loss functions: tells how good (bad) the classifier is

A dataset of examples: $\{(x_i, y_i)\}^N_{i=1}$

Loss over the dataset: $L = \frac{1}{N}\sum\limits_iL_i(f(x_i, W), y_i)$

Multi-class SVM loss:

score vector: $s_i = f(x_i, W)$

$L_i = \sum\limits_{j \neq y_i} 0 \quad (s_{y_i} \geq s_j + 1)$

$L_i = \sum\limits_{j \neq y_i} s_j - s_{y_i} - 1 \quad (others)$

$L_i = \sum\limits_{j \neq y_i} \max(0, s_j - s_{y_i} - 1)$

Hinge loss

Loss = data loss + regularization loss

data loss: model predictions should match training data

regularization loss: model should be simple -> work on test data

Regularizations:

$L_2: R(W) = \sum\sum W^2$

$L_1: R(W) = \sum\sum |W|$

Elastic net ( $L_1 + L_2$ ): $R(W) = \sum\sum(\beta W^2 + |W|)$

max norm, dropout, batch normalization, stochastic depth...

Softmax Classifier (Multinomial Logistic Regression)

maximize the log likelihood

$L_i = -\log P(Y = y_i | X = x_i)$

Optimizing

- Numerical gradient: approximate, slow, easy to write (for debugging)
- Analytic gradient: exact, fast, error-prone

Gradient descent

```python
# Vanilla gradient descent
while True:
    wrights_grad = evaluate_gradient(loss_fun, data, weights)
    weights += -step_size * weight_grad     # perform parameter update
```

Stochastic Gradient Descent (SGD)

Full sum is expensive when N is large!

Approximate sum using a minibatch of examples (32/64/128 common)

```python
# Vanilla Minibatch Gradient Descent
while True:
    data_batch = sample_training_data(data, 256)    # sample 256 examples
    weights_grad = evaluate_gradient(loss_fun, data_batch, weights)
    weights += -step_size * weights_grad    # perform parameter update
```

Aside: Image Features

Image $\rightarrow$ Feature Representations $\rightarrow$ Class

Image Features: Motivation <- feature transform

Histogram of Oriented Gradients (HoG)

Bag of Words
