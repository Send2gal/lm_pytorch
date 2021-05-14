import numpy as np
import matplotlib.pyplot as plt
import cv2

def main():
    # training_data = np.load('training_data.npy', allow_pickle=True)
    img1 = cv2.imread('dog-vs-cats/test1/1.jpg')
    img1_flip = cv2.flip(img1, 1)
    plt.imshow(img1)
    plt.show()

    plt.imshow(img1_flip)
    plt.show()


if __name__ == '__main__':
    main()