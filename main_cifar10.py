import os
import pickle

import numpy as np
import pandas as pd
import torch
import torchvision
import tarfile
from torchvision.datasets.utils import download_url
import torchvision.transforms as transforms
import random
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


# Download the dataset
dataset_rul = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

download_url(dataset_rul, '.')

# Extract from archive
with tarfile.open('./cifar-10-python.tar.gz', 'r:gz') as tar:
    def is_within_directory(directory, target):
        
        abs_directory = os.path.abspath(directory)
        abs_target = os.path.abspath(target)
    
        prefix = os.path.commonprefix([abs_directory, abs_target])
        
        return prefix == abs_directory
    
    def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
    
        for member in tar.getmembers():
            member_path = os.path.join(path, member.name)
            if not is_within_directory(path, member_path):
                raise Exception("Attempted Path Traversal in Tar File")
    
        tar.extractall(path, members, numeric_owner=numeric_owner) 
        
    
    safe_extract(tar, path="./data")

data_dir = './data/cifar-10-batches-py'

# Look inside the dataset directory

path_data_batch_1 = os.path.join(data_dir, 'data_batch_1')


def load_cifar10(file_name):
    with open('./data/cifar-10-batches-py/' + file_name, 'rb') as file:
        batch = pickle.load(file, encoding='latin1')
        features = batch['data']
        labels = batch['labels']
        return features, labels


# Load files
batch_1, labels_1 = load_cifar10('data_batch_1')
batch_2, labels_2 = load_cifar10('data_batch_2')
batch_3, labels_3 = load_cifar10('data_batch_3')
batch_4, labels_4 = load_cifar10('data_batch_4')
batch_5, labels_5 = load_cifar10('data_batch_5')

test, label_test = load_cifar10('test_batch')

# Merge files
x_train = np.concatenate([batch_1, batch_2, batch_3, batch_4, batch_5], 0)
y_train = np.concatenate([labels_1, labels_2, labels_3, labels_4, labels_5], 0)

classes = ('airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def return_photo(batch_file):
    assert batch_file.shape[1] == 3072
    dim = np.sqrt(1024).astype(int)
    r = batch_file[:, 0:1024].reshape(batch_file.shape[0], dim, dim, 1)
    g = batch_file[:, 1024:2048].reshape(batch_file.shape[0], dim, dim, 1)
    b = batch_file[:, 2048:3072].reshape(batch_file.shape[0], dim, dim, 1)
    photo = np.concatenate([r, g, b], -1)
    return photo


x_train = return_photo(x_train)
x_test = return_photo(test)
y_test = np.array(label_test)


def plot_image(number, file, label, pred=None):
    fig = plt.figure(figsize=(3, 2))
    # img = return_photo (batch_file)
    plt.imshow(file[number])
    if pred is None:
        plt.title(classes[label[number]])
    else:
        plt.title('Lable_true: '+ classes[label[number]] + '\nLabel_pred: ' +classes[pred[number]])
    plt.show()

# for i in range(10):
#     random_image_number = np.random.randint(len(y_train))
#     plot_image(random_image_number, x_train, y_train)

# The cifar-10 is designed to balance distribution that the counts for each calssifiction are 5000
# sns.countplot(y_train)
# hist_t_train = pd.Series(y_train).groupby(y_train).count()
# print(hist_t_train)


# Final check for dimensions before pre-processing
print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print('x_test shape:', x_test.shape)
print('y_test shape:', y_test.shape)

# split the validation set out
x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(x_train, y_train, train_size=0.2, random_state=42)

# prepare for training & testing dataset. Define dataset class.
# define the random seed for reproducible results
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


class CIFAR10_from_array(Dataset):
    def __init__(self, data, label, transform=None):
        # Initialize path, transforms, and so on
        self.data = data
        self.label = label
        self.transform = transforms
        self.img_shape = data.shape

    def __getitem__(self, index):
        # 1. read from file (using numpy.fromfile, PIL.Image.open)
        # 2. preprocess the data (torchvistion.Transform)
        # 3. return the data (e.g. image and label)

        img = Image.formarray(self.data[index])
        label = self.label[index]
        if self.transform is not None:
            img = self.transform(img)
        else:
            img_to_tensor = transforms.ToTensor()
            img = img_to_tensor(img)
            # label = torch.from_numpy(label).long()
            return img, label

    def __len__(self):
        # indicate the total size of the dataset
        return len(self.data)

    def plot_image(self, number):
        file = self.data
        label = self.label
        fig = plt.figure(figsize=(3, 2))
        # img = return_photo(batch_file)
        plt.imshow(file[number])
        plt.title(classes[label[number]])


class CIFAR10_from_url(Dataset):
    pass

# normalize for R, G, B with img - img - mean / std
def normalize_dataset(data):
    mean = data.mean(axis=(0, 1, 2)) / 255.0
    std = data.std(axis=(0, 1, 2)) / 255.0
    normalize = transforms.Normalize(mean=mean, std=std)
    return normalize





print('end')





