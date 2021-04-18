import torch

# for working with data
import torchvision
from torchvision import transforms, datasets

BATCH_SIZE = 10

def data():
    # download the date
    train = datasets.MNIST("", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    test = datasets.MNIST("", train=True, download=False, transform=transforms.Compose([transforms.ToTensor()]))

    # split the date to batches + shuffle
    train_set = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    test_set = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    return train_set, test_set

def main():
    train_set, test_set = data()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
