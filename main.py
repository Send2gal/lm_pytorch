import torch
import torch.nn as nn
import torch.nn.functional as F

# for working with data
import torchvision
from torchvision import transforms, datasets

# visualization the data
import matplotlib.pyplot as plt

BATCH_SIZE = 10


def data():
    # download the date
    train = datasets.MNIST("", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    test = datasets.MNIST("", train=True, download=False, transform=transforms.Compose([transforms.ToTensor()]))

    # split the date to batches + shuffle
    train_set = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    test_set = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    return train_set, test_set


def visual_data(d, x=0, y=0):
    plt.imshow(d[x][y].view(d[x][y].shape[1], d[x][y].shape[2]))
    plt.show()


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 64)  # fc = fully connected, 28*28 number of pixels
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)  # 10 is the 10 output option

    def forward(self, x):
        x = F.relu(self.fc1(x))  # relu = rectified linear
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)


def main():
    train_set, test_set = data()
    net = Net()

    for d in train_set:
        x = d[0][0]
        y = d[1][0]
        # print(x)
        # print(y)
        visual_data(d)
        break



if __name__ == '__main__':
    main()

