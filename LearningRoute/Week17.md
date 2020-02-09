# 第17周周报

## 学习内容

- cs231n assignments

## 学习收获

### Assignment 1

#### k nearest neighbor (kNN)

计算data之间pixel-wise的l2 distance，选出距离最近的k个training data，根据这k个data的label来决定test data的class

train只需要把training data保存下来

```python
# in class k-NN classifier
def train(self, X, y):
    self.X_train = X
    self.y_train = y
```

计算l2 distance

$$
d_2 (I_1, I_2) = \sqrt{\sum_p (I_1^p - I_2^p)^2}
\\
d_2 (I_1, I_2)= \sqrt{\sum_p ((I_1^p)^2 - 2 I_1^p \cdot I_2^p + (I_2^p)^2)}
$$

```python
num_test = X.shape[0]
num_train = self.X_train.shape[0]
dists = np.zeros((num_test, num_train))
# vectorized implement
dists += np.sum(X ** 2, axis=1).reshape(-1, 1)
dists -= 2 * np.dot(X, self.X_train.T)
dists += np.sum(self.X_train ** 2, axis=1).reshape(1, -1)
dists = np.sqrt(dists)
```

predict只需要根据距离从小到大排序，截取前k个，再进行统计

```python
def predict_labels(self, dists, k=1):
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in range(num_test):
        closest_y = []
        closest_y = self.y_train[np.argsort(dists[i])[:k]]
        y_pred[i] = np.bincount(closest_y).argmax()
    return y_pred
```

#### support vector machine (SVM)

SVM Loss:

$$
s_j = f(x_i, W)_j
\\
L_i = \sum_{j \neq y_i} \max (0, s_j - s_{y_i} + \Delta)
$$

其中 $\Delta$ 是hyper parameter

Loss = data loss + regularization loss

$$
L = \frac{1}{N}\sum_i L_i + \lambda R(W)
\\
R(W) = \sum_k \sum_l W^2_{k, l}
$$

implement of SVM loss

```python
def svm_loss_naive(W, X, y, reg):
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1    # note that delta = 1
            # max(0, margin)
            if margin > 0:
                loss += margin
                dW[:, j] += X[i].T
                dW[:, y[i]] -= X[i].T

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # svm loss
    num_train = X.shape[0]
    num_classes = W.shape[1]
    scores = X.dot(W)   # (N, C)
    correct_class_score = scores[range(num_train), list(y)].reshape(-1, 1)  # (N, 1)
    margins = np.maximum(0, scores - correct_class_score + 1)   # note that delta = 1
    margins[range(num_train), list(y)] = 0
    loss = np.sum(margins) / num_train + reg * np.sum(W * W)

    # gradient
    coeff_mat = np.zeros((num_train, num_classes))
    coeff_mat[margins > 0] = 1
    # the following line can be unnecessary
    # coeff_mat[range(num_train), list(y)] = 0
    coeff_mat[range(num_train), list(y)] -= np.sum(coeff_mat, axis=1)

    dW = np.dot(X.T, coeff_mat)
    dW = dW / num_train + 2 * reg * W

    return loss, dW
```

Stochastic Gradient Descent

```python
# in a step of SGD:
# choose some random data
idx = np.random.choice(num_train, batch_size, replace=True)
X_batch = X[idx, :]
y_batch = y[idx]

loss, grad = self.loss(X_batch, y_batch, reg)
loss_history.append(loss)

self.W -= learning_rate * grad
```

#### Softmax

Softmax function: $f_j(z) = \frac{e^{z_j}}{\sum_ke^{z_k}}$

Softmax cross entropy Loss ( $f$ 表示class score, $f_j$ 表示第j个元素的class score):

$$
f(x_i; W) = Wx_i
\\
L_i = - \log (\frac{e^{f_{y_i}}}{\sum_j e^{f_j}})
\quad or \quad
L_i = - f_{y_i} + \log \sum_j e^{f_j}
\\
L = \frac{1}{N}\sum_i L_i + \lambda R(W)
$$

Probabilistic interpretation:

$$
P(y_i | x_i; W) = \frac{e^{f_{y_i}}}{\sum_j e^{f_j}}
$$

Implement:

```python
def softmax_loss_vectorized(W, X, y, reg):
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    num_train = X.shape[0]
    num_classes = W.shape[1]

    scores = X.dot(W)   # (N, C)
    shifted_scores = scores - np.max(scores, axis=1).reshape(-1, 1)
    softmax_out = np.exp(shifted_scores) / np.sum(np.exp(shifted_scores), axis=1).reshape(-1, 1)
    loss = -np.sum(np.log(softmax_out[range(num_train), list(y)]))
    loss /= num_train
    loss += reg * np.sum(W ** 2)

    # delta softmax
    dS = softmax_out.copy()
    # if j == y[i]
    dS[range(num_train), list(y)] += -1
    dW = np.dot(X.T, dS)
    dW = dW / num_train + reg * W

    return loss, dW
```

#### Neural Networks

Architecture of neural networks:

input (vector) $\rightarrow$ first layer $\rightarrow$ ReLu $\rightarrow$ second layer $\rightarrow$ Softmax $\rightarrow$ prediction

通过forward计算loss和grad，用backward更新weights

```python
# forward
def loss(self, X, y=None, reg=0.0):
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape

    # Compute the forward pass
    scores = None

    # ReLu
    h_out = np.maximum(0, X.dot(W1) + b1)   # (N, D) * (D, H) = (N, H)
    scores = np.dot(h_out, W2) + b2     # (N, C)

    # If the targets are not given then jump out, we're done
    if y is None:
        return scores

    # Compute the loss
    loss = None

    shifted_scores = scores - np.max(scores, axis=1).reshape(-1, 1)
    softmax_out = np.exp(shifted_scores) / np.sum(np.exp(shifted_scores), axis=1).reshape(-1, 1)
    loss = -np.sum(np.log(softmax_out[range(N), list(y)]))
    loss /= N
    loss += reg * (np.sum(W1 ** 2) + np.sum(W2 ** 2))

    dscores = softmax_out.copy()
    dscores[range(N), list(y)] -= 1
    dscores /= N
    grads['W2'] = np.dot(h_out.T, dscores) + 2 * reg * W2
    grads['b2'] = np.sum(dscores, axis=0)

    dh = dscores.dot(W2.T)  # (N, H)
    dh_ReLu = (h_out > 0) * dh
    grads['W1'] = np.dot(X.T, dh_ReLu) + 2 * reg * W1
    grads['b1'] = np.sum(dh_ReLu, axis=0)

    return loss, grads
```

update weights $W_{i+1} = W_i - learning\_rate * \frac{\partial L}{\partial W}|_{W = W_i}$ :

```python
self.params['W1'] += - learning_rate * grads['W1']
self.params['b1'] += - learning_rate * grads['b1']
self.params['W2'] += - learning_rate * grads['W2']
self.params['b2'] += - learning_rate * grads['b2']
```

predict:

```python
h_out = np.maximum(0, X.dot(self.params['W1'])) + self.params['b1']
scores = h_out.dot(self.params['W2']) + self.params['b2']
y_pred = np.argmax(scores, axis=1)
```

由于NN的hyper-parameter较多，采用cross-validation + brute force search的方法确定合适的hyper-parameter，先大范围后小范围，采用二分法缩减搜索区间。

```python
best_net = None
best_val = -1
best_lr = 0
best_rs = 0
best_hd = 0
input_size = 32 * 32 * 3
num_classes = 10

learning_rates =  np.linspace(1.6, 1.7, 10) * 1e-3
regularization_strengths = [0, 0.25, 0.5, 0.75, 1]
hidden_dims = [64, 128]

print('Begin searching...')
for lr in learning_rates:
    for rs in regularization_strengths:
        for hd in hidden_dims:
            nn = TwoLayerNet(input_size, hd, num_classes)
            nn.train(X_train, y_train, X_val, y_val, num_iters=2000, learning_rate=lr, reg=rs, verbose=False)
            val_acc = (nn.predict(X_val) == y_val).mean()
            if val_acc > best_val:
                best_val = val_acc
                best_lr = lr
                best_rs = rs
                best_hd = hd
                best_net = nn
                print('Current hyperparameters: lr={}, rs={}, hd={}, val_acc={}'.format(lr, rs, hd, val_acc))

print('Done!')
print('lr: ', best_lr)
print('rs: ', best_rs)
print('hd: ', best_hd)
```

### Assignment 2

#### Fully-connected NN

分别实现各个layer的forward, activation和backward，再组合成整个model

$input \rightarrow hidden\ layer\ 1\ (with\ activation)\rightarrow \cdots (All\ with\ activation) \rightarrow hidden\ layer\ N \rightarrow output\ layer \rightarrow Softmax \rightarrow nun\_classes$

```python
# layers.py
# forward
x_vec = x.reshape(x.shape[0], -1)   # (N, D)
out = np.dot(x_vec, w) + b  # (N, M)

# backward
N, D = x.shape[0], w.shape[0]
dx = np.dot(dout, w.T).reshape(x.shape)     # (N, M) * (M, D) = (N, D)
dw = np.dot(x.reshape(N, -1).T, dout)   # (D, N) * (N, M) = (D, M)
db = np.sum(dout, axis=0)
```

Two layer neural network

```python
class TwoLayerNet(object):
    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        self.params = {}
        self.reg = reg

        # initialize
        self.params['W1'] = np.random.randn(input_dim, hidden_dim) * weight_scale
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = np.random.randn(hidden_dim, num_classes) * weight_scale
        self.params['b2'] = np.zeros(num_classes)

    def loss(self, X, y=None):
        scores = None

        hout, h_cache = affine_relu_forward(X, self.params['W1'], self.params['b1'])
        scores, out_cache = affine_forward(hout, self.params['W2'], self.params['b2'])

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}

        # loss and gradient
        loss, dscores = softmax_loss(scores, y)
        loss += 0.5 * self.reg * (np.sum(self.params['W1'] ** 2) + np.sum(self.params['W2'] ** 2))
        dx2, dw2, db2 = affine_backward(dscores, out_cache)
        grads['W2'] = dw2 + self.reg * self.params['W2']
        grads['b2'] = db2
        dx1, dw1, db1 = affine_relu_backward(dx2, h_cache)
        grads['W1'] = dw1 + self.reg * self.params['W1']
        grads['b1'] = db1

        return loss, grads
```

以上network的architecture跟assignment1的相同：

input (vector) $\rightarrow$ hidden layer $\rightarrow$ ReLu $\rightarrow$ output layer $\rightarrow$ Softmax $\rightarrow$ prediction

Training & Testing

```python
# optimizer config
opconf = {'learning_rate': 1e-3}
solver = Solver(model, data, print_every=100, optim_config=opconf, lr_decay=0.9, num_epochs=20)
solver.train()
test_accuracy = solver.check_accuracy(data['X_test'], data['y_test'])
print('Test:', test_accuracy)
```

上述所有代码均在jupyter notebook上运行和debug，使用方法如下

```python
# in shell
jupyter notebook
```

之后浏览器会自动打开并跳转到根目录，运行相应的`ipynb`文件，`Shift+Enter`运行相应代码块

## 疑问和困难

- 除了做cross-validation和暴力搜索以外，不知道更好的寻找hyper parameter的方式
