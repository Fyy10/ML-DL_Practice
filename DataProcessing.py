from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils

import warnings
warnings.filterwarnings('ignore')

plt.ion()   # interactive mode

landmarks_frame = pd.read_csv('data/faces/face_landmarks.csv')
print(landmarks_frame)

n = 65  # choose the nth image
img_name = landmarks_frame.iloc[n, 0]
print(img_name)
landmarks = landmarks_frame.iloc[n, 1:].as_matrix()
print(landmarks)
print(landmarks.reshape(-1, 2))
landmarks = landmarks.astype('float').reshape(-1, 2)
print(landmarks)

print('Image name: {}'.format(img_name))
print('Landmarks shape: {}'.format(landmarks.shape))
print('First 4 Landmarks: {}'.format(landmarks[:4]))


def show_landmarks(image, landmarks):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)    # pause to see the plot update


plt.figure(img_name)
show_landmarks(io.imread(os.path.join('data/faces/', img_name)), landmarks)
plt.show()


# Create a dataset for face landmarks
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

        image = io.imread(img_pos)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample


face_dataset = FaceLandmarksDataset(csv_file='data/faces/face_landmarks.csv', root_dir='data/faces/')

for i in range(len(face_dataset)):
    sample = face_dataset[i]

    print(i, sample['image'].shape, sample['landmarks'].shape)

    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    show_landmarks(**sample)

    if i == 3:
        plt.show()
        break


class Rescale(object):
    """Rescale the image in a sample to a given size"""
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:   # set the value of shorter side and not change h/w
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images, x and y axes are 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmarks': landmarks}


class RandomCrop(object):
    """Crop randomly the image in a sample"""
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h, left: left + new_w]
        landmarks = landmarks - [left, top]     # why??
        return {'image': image, 'landmarks': landmarks}


class ToTensor(object):
    """Convert ndarrays in sample to tensors"""
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis since:
        # numpy image: H * W * C
        # torch image: C * H * W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image), 'landmarks': torch.from_numpy(landmarks)}


scale = Rescale(256)
crop = RandomCrop(128)
composed = transforms.Compose([scale, crop])

# Apply each of transforms on sample (compose transforms)
fig = plt.figure()
sample = face_dataset[n]
for i, transfm in enumerate([scale, crop, composed]):
    transformed_sample = transfm(sample)

    ax = plt.subplot(1, 3, i + 1)
    plt.tight_layout()
    ax.set_title(type(transfm).__name__)
    show_landmarks(**transformed_sample)
plt.show()

transformed_dataset = FaceLandmarksDataset(
    csv_file='data/faces/face_landmarks.csv',
    root_dir='data/faces/',
    transform=transforms.Compose([Rescale(256), RandomCrop(224), ToTensor()])
)

for i in range(len(transformed_dataset)):
    sample = transformed_dataset[i]
    print(i, sample['image'].size(), sample['landmarks'].size())

    if i == 3:
        break

# num_workers has to be 0 on Windows (known bug in PyTorch)
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
