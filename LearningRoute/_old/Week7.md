# 第7周学习报告

## 学习内容

- PyTorch数据处理及可视化

## 学习收获

### 数据库读取

从csv文件读取（用pandas包）：

```python
import panda as pd
from skimage import io
# read CSV and get annotations in an (N, 2) array, N is the number of landmarks
landmarks_frame = pd.read_csv('data/faces/face_landmarks.csv')

n = 65	# choose the nth image, not the N above
img_name = landmarks_frame.iloc[n, 0]
print(img_name)
landmarks = landmarks_frame.iloc[n, 1:].as_matrix()
landmarks = landmarks.astype('float').reshape(-1, 2)
print(landmarks)


# 显示image and landmarks
def show_landmarks(image, landmarks):
	plt.imshow(image)
	plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
	plt.pause(0.001)


plt.figure(img_name)
show_landmarks(io.imread(os.path.join('data/faces/', img_name)), landmarks)
plt.show()
```

Dataset类

使用`torch.utils.data.Dataset`来表示一个数据库（dataset），自定义的datase应继承`Dataset`，并且override下列方法：

- `__len__`: return the size of the dataset
- `__getitem__`: to support the indexing such that dataset[i] can be used for ith sample

```python
# Create a dataset for face landmarks
# Read the csv in __init__ but read the images in __getitem__ to save memory (read image as is required)
# sample: a dict {'image': image, 'landmarks': landmarks}
class FaceLandmarksDataset(Dataset):
    """Face Landmarks Dataset"""
    def __init__(self, csv_file, root_dir, transform=None):
        """
        :param csv_file (string): Path to read csv file with annotations
        :param root_dir (string): Directory of all images
        :param transform (callable, optional): Optional transform to be applied
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_pos = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx, 0])

        image = io.read(img_pos)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample
```

从自定义的数据库中读取数据并显示：

```python
face_dataset = FaceLandmarkDataset(csv_file='data/faces/face_landmarks.csv', root_dir='data/faces/')

for i in range(len(face_dataset)):
	sample = face_dataset[i]
	print(i, sample['image'].shape, sample['landmarks'].shape)
```

设置Transforms:

- Rescale: rescale the image to the same shape
- RandomCrop: crop from image randomly (data augmentation数据增强，防止过拟合)
- ToTensor: convert numpy images to torch images (need to swap axis)

```python
# 实现Rescale类，进行image的缩放
class Rescale(object):
    """Rescale the image in a sample to a given size"""
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):		# 为了能将Rescale以类似函数的方式调用
        image, landmarks = sample['image'], sample['landmarks']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:	# set the value of the shorter side and not change h/w
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images, x and y axes are 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]		# landmarks中的x和y与image中的h和w是相反的

        return {'image': img, 'landmarks': landmarks}

# 实现RandomCrop类，进行image的随机裁剪
class RandomCrop(object):
    """Crop randomly the image in a sample"""
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size

    def __ceil__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h, left: left + new_w]
        landmarks = landmarks - [left, top]		# why??
        return {'image': image, 'landmarks': landmarks}

# 将sample的numpy image转化为torch tensor image
class ToTensor(object):
    """Convert ndarrays in sample to tensors"""
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis since:
        # numpy image: H * W * C
        # torch image: C * H * W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image), 'landmarks': torch.from_numpy(landmarks)}

# Usage
scale = Rescale(256)
crop = RandomCrop(128)
```

Compose transforms

Apply the transforms on a sample

使用`torchvision.transforms.Compose`包来compose transforms

```python
scale = Rescale(256)
crop = RandomCrop(128)
composed = transforms.Compose([scale, crop])

# Apply each of transforms on sample (compose transforms)
fig = plt.figure()
sample = face_dataset[n]	# nth sample
for i, transfm in enumerate([scale, crop, composed]):
    transformed_sample = transfm(sample)

    ax = plt.subplot(1, 3, i + 1)
    plt.tight_layout()
    ax.set_title(type(transfm).__name__)
    show_landmarks(**transformed_sample)
plt.show()
```

Iterating through the dataset

Create a dataset with composed transforms

每次从dataset中sample时，进行以下操作：

- An image is read from file
- Transforms are applied on the read image
- data is augmented on sampling since one of the transforms is random

```python
# Create a dataset with composed transforms
transformed_dataset = FaceLandmarksDataset(csv_file='data/faces/face_landmarks.csv', root_dir='data/faces/', transform=transforms.Compose([Rescale(256), RandomCrop(224), ToTensor()]))

# Iterating through the dataset
for i in range(len(transformed_dataset)):
	sample = transformed_dataset[i]
	print(i, sample['image'].size(), sample['landmarks'].size())

	if i == 3:
		break
```

但是在迭代取样整个数据库的过程中，有很多操作没有进行：

- Batching the data
- Shuffling the data
- Load the data in parallel using `multiprocessing` workers

`torch.utils.data.DataLoader`是一个提供了上述特征的迭代器，可以用DataLoader来进行数据库的迭代

```python
# 在Windows上，num_workers只能设置为0，否则会出现pipe error
dataloader = DataLoader(transformed_dataset, batch_size=4, shuffle=True, num_workers=0)


# Show a batch of data
def show_landmarks_batch(sample_batched):
    """Show image with landmarks for a batch of samples"""
    images_batch, landmarks_batch = sample_batched['image'], sample_batched['landmarks']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)
    grid_border_size = 2

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose(1, 2, 0))

    for i in range(batch_size):
        plt.scatter(landmarks_batch[i, :, 0].numpy() + i * im_size + (i + 1) * grid_border_size, landmarks_batch[i, :, 1].numpy() + grid_border_size, s=10, marker='.', c='r')
        plt.title('Batch from dataloader')


for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['image'].size(), sample_batched['landmarks'].size())

    # observe one (4th) batch and stop
    if i_batch == 3:
        plt.figure()
        show_landmarks_batch(sample_batched)
        plt.axis('off')
        plt.ioff()
        plt.show()
        break
```

可以使用`torchvision.datasets.ImageFolder`来读取文件夹中的图片作为数据，组成dataset

### 模型可视化

使用TensorBoard进行model训练过程的可视化操作

#### Setup TensorBoard

```python
from torch.utils.tensorboard import SummaryWriter
# default 'log_dir' is "runs"
# writing information to TensorBoard
writer = SummaryWriter('runs/fashion_mnist_experiment_1')
# will create a folder: runs/fashion_mnist_experiment_1
```

#### Writing to TensorBoard

```python
# Randomly pick some training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# create grid of images
img_grid = torchvision.utils.make_grid(images)

# write to TensorBoard
writer.add_image('four_fashion_mnist_images', img_grid)
```

将信息写到TensorBoard中后，在命令行中运行以下命令

```shell
tensorboard --logdir=runs
```

然后用浏览器访问[https://localhost:6006](https://localhost:6006)即可看到TensorBoard

#### Inspect the model using TensorBoard

将模型可视化

```python
writer.add_graph(net, images)
writer.close()
```

#### Adding a "Projector" to TensorBoard

To visualize the lower dimensional representation of higher dimensional data (via `add_embedding` method)

```python
# select random images and their target indices
images, labels = select_n_random(trainset.data, trainset.targets)

# get the class labels for each image
class_labels = [classes[lab] for lab in labels]

# log embeddings
features = images.view(-1, 28 * 28)
writer.add_embedding(features,
                     metadata=class_labels,
                     label_img=images.unsqueeze(1))
writer.close()
```

## 疑问和困难

1. 在读取数据库和数据的预处理方面还比较模糊，需要多加练习
2. matplotlib库的使用还不熟练
3. 在给TensorBoard添加Projector时，显示出来的页面是空白的
