#kaggle competitions download -c dogs-vs-cats

import time
import os
from datetime import datetime

import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


REBUILD_DATA = False
IMG_SIZE = 50
VAL_PCT = 0.1
LR = 0.001
BATCH_SIZE = 100
EPOCHS = 3
CALC_ACCURACY = 50

class DogVSCats(object):
    CATS = "dog-vs-cats/train/cats"
    DOGS = "dog-vs-cats/train/dogs"
    LABELS = {CATS: 0, DOGS: 1}
    training_data = []
    cat_count = 0
    dog_count = 0

    def make_training_data(self):
        for label in self.LABELS:
            print(label)
            for f in tqdm(os.listdir(label)):
                try:
                    path = os.path.join(label, f)
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    img_flip = cv2.flip(img, 1)
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    img_flip = cv2.resize(img_flip, (IMG_SIZE, IMG_SIZE))
                    self.training_data.append([np.array(img), np.eye(2)[self.LABELS[label]]])
                    self.training_data.append([np.array(img_flip), np.eye(2)[self.LABELS[label]]])

                    if label == self.CATS:
                        self.cat_count += 1
                    if label == self.DOGS:
                        self.dog_count += 1
                except Exception as e:
                    print(str(e))

        np.random.shuffle(self.training_data)
        np.save('training_data.npy', self.training_data)
        print(f"Cats: {self.cat_count}")
        print(f"Dogs: {self.dog_count}")


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)

        x = torch.randn(IMG_SIZE, IMG_SIZE).view(-1, 1, IMG_SIZE, IMG_SIZE)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, 2)

    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))

        # print(x[0].shape)
        if self._to_linear is None:
            self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)


class Main():
    def __init__(self):
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None

        if REBUILD_DATA:
            self.build_data()
        self.training_data = np.load('training_data.npy', allow_pickle=True)

        if torch.cuda.is_available():
            self.device = torch.device("cude:0")
            print("Running on GPU")
            print(f"You have available GPU: {torch.cuda.device_count()}")
        else:
            self.device = torch.device("cpu")
            print("Running on CPU")

        self.net = Net().to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=LR)
        self.loss_function = nn.MSELoss()

    def build_data(self):
        dog_vs_cat = DogVSCats()
        dog_vs_cat.make_training_data()

    def prepare_data(self):
        X = torch.Tensor([i[0] for i in self.training_data]).view(-1, IMG_SIZE, IMG_SIZE)
        X = X / 255.0
        y = torch.Tensor([i[1] for i in self.training_data])
        val_size = int(len(X) * VAL_PCT)
        # print(val_size)

        self.train_x = X[:-val_size]
        self.train_y = y[:-val_size]

        self.test_x = X[-val_size:]
        self.test_y = y[-val_size:]

    def train(self, epochs, batch_size):
        writer = SummaryWriter(f"runs/{datetime.now().strftime('%d_%m_%Y')}_model_flip_epoch_{epochs}")

        for epoch in range(epochs):
            for i in tqdm(range(0, len(self.train_x), batch_size)):
                # print(i, i+BATCH_SIZE)
                batch_x = self.train_x[i:i+batch_size].view(-1, 1, IMG_SIZE, IMG_SIZE).to(self.device)
                batch_y = self.train_y[i:i+batch_size].to(self.device)

                acc, loss = self.forward_pass(batch_x, batch_y, train=True)
                if i % CALC_ACCURACY == 0:
                    val_acc, val_loss = self.test_(size=32)
                    _x = len(self.train_x)*epoch + i
                    # val == validation
                    writer.add_scalar('Loss/test', round(float(val_loss), 4), _x)
                    writer.add_scalar('Loss/train', round(float(loss), 4), _x)
                    writer.add_scalar('Accuracy/test', round(float(val_acc), 4), _x)
                    writer.add_scalar('Accuracy/train', round(float(acc), 4), _x)

    def test(self):
        correct = 0
        total = 0
        with torch.no_grad():
            for i in tqdm(range(len(self.test_x))):
                real_class = torch.argmax((self.test_y[i]))
                net_out = self.net(self.test_x[i].view(-1, 1, IMG_SIZE, IMG_SIZE).to(self.device))[0]
                predicted_class = torch.argmax(net_out)
                if predicted_class == real_class:
                    correct += 1
                total += 1
        print(f"Acuuracy: {round(correct / total, 3) * 100}\n")

    def forward_pass(self, x, y, train=False):
        if train:
            self.net.zero_grad()
        outputs = self.net(x)
        matches = [torch.argmax(i) == torch.argmax(j) for i, j in zip(outputs, y)]
        accuracy = matches.count(True)/len(matches)
        loss = self.loss_function(outputs, y)

        if train:
            loss.backward()
            self.optimizer.step()
        return accuracy, loss

    def test_(self, size=32):
        random_start = np.random.randint(len(self.test_x)-size)
        x = self.test_x[random_start:random_start+size].view(-1, 1, IMG_SIZE, IMG_SIZE).to(self.device)
        y = self.test_y[random_start:random_start+size].to(self.device)
        with torch.no_grad():
            val_acc, val_loss = self.forward_pass(x, y, train=False)
        return val_acc, val_loss


def main():

    for EPOCHS in range(3, 10):
        main_obj = Main()
        main_obj.prepare_data()
        main_obj.train(EPOCHS, BATCH_SIZE)
        print(f"\nEPOCHS: {EPOCHS}")
        main_obj.test()


if __name__ == '__main__':
    main()
