from keras.datasets import mnist
from matplotlib import pyplot
from sklearn.utils import shuffle
import numpy as np

 
def get_dataset(dimension_training_set, dimensione_test_set, scaled=False):
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    train_X = np.array(train_X)
    if scaled:
        train_X = train_X / 255
    train_X = train_X.reshape(train_X.shape[0], train_X.shape[1] * train_X.shape[2])
    train_X, train_y = shuffle(train_X, train_y)
    test_X = np.array(test_X)
    if scaled:
        test_X = test_X / 255
    test_X = test_X.reshape(test_X.shape[0], test_X.shape[1] * test_X.shape[2])
    test_X, test_y = shuffle(test_X, test_y)
    return (train_X[:dimension_training_set], train_y[:dimension_training_set]), (test_X[:dimensione_test_set], test_y[:dimensione_test_set])




#loading
#(train_X, train_y), (test_X, test_y) = mnist.load_data()
#print(train_y)

 
#plotting
'''
from matplotlib import pyplot
for i in range(9):  
    pyplot.subplot(330 + 1 + i)
    pyplot.imshow(train_X[i], cmap=pyplot.get_cmap('gray'))
pyplot.show()
'''
