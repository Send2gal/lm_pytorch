import torch

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


def visual_data(d,x=0,y=0):
    plt.imshow(d[x][y].view(d[x][y].shape[1], d[x][y].shape[2]))
    plt.show()


def main():
    train_set, test_set = data()

    for d in train_set:
        x = d[0][0]
        y = d[1][0]
        # print(x)
        # print(y)
        visual_data(d)
        break



if __name__ == '__main__':
    main()

