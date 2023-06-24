import numpy as np
import matplotlib.pyplot as plt
from neural_network_mnist import train_network_with_iRprop_plus, Activation_LReLU, train_network_with_iRprop_minus, \
    train_network_with_Rprop_plus, Activation_ReLU, train_network_with_SGD
from mnist import get_dataset
import cv2

img0 = cv2.imread("0.bmp", 0)
img1 = cv2.imread("1.bmp", 0)
img2 = cv2.imread("2.bmp", 0)
img3 = cv2.imread("3.bmp", 0)
img4 = cv2.imread("4.bmp", 0)
img5 = cv2.imread("5.bmp", 0)
img6 = cv2.imread("6.bmp", 0)
img7 = cv2.imread("7.bmp", 0)
img8 = cv2.imread("8.bmp", 0)
img9 = cv2.imread("9.bmp", 0)

img_arr = np.array([img0, img1, img2, img3, img4, img5, img6, img7, img8, img9]).reshape(10, 784)

img_arr = img_arr / 255

(train_X, train_y), (test_X, test_y) = get_dataset(30000, 5000)
EPOCHS = 1000
N_NEURONS = 500

activ_func = Activation_LReLU()
network = train_network_with_SGD(train_X, train_y, test_X, test_y, EPOCHS, N_NEURONS, activ_func, "random")

prediction = network.forward(img_arr)

print("Testing the network on self made images digits from 0 to 9")
print(prediction)
