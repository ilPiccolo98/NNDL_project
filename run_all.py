import numpy as np
import matplotlib.pyplot as plt
from neural_network_mnist import execute_SGD, execute_RProp_minus, execute_RProp_plus, execute_iRProp_plus, execute_iRProp_minus, Activation_LReLU, Activation_Sigmoid, Activation_ReLU
from mnist import get_dataset


(train_X, train_y), (test_X, test_y) = get_dataset(3000, 500)
EPOCHS = 500
N_NEURONS = 150

'''
#RELU
activation1_sgd = Activation_ReLU()
activation1_rprop_minus = Activation_ReLU()
activation1_rprop_plus = Activation_ReLU()
activation1_irprop_plus = Activation_ReLU()
activation1_irprop_minus = Activation_ReLU()


epochs_SGD, loss_values_SGD, accuracy_values_SGD, loss_test_SGD, accuracy_test_SGD = execute_SGD(train_X, train_y, test_X, test_y, EPOCHS, N_NEURONS, activation1_sgd)
epochs_Rpropmin, loss_values_Rpropmin, accuracy_values_Rpropmin, loss_test_Rpropmin, accuracy_test_Rpropmin = execute_RProp_minus(train_X, train_y, test_X, test_y, EPOCHS, N_NEURONS, activation1_rprop_minus)
epochs_Rpropplus, loss_values_Rpropplus, accuracy_values_Rpropplus, loss_test_Rpropplus, accuracy_test_Rpropplus = execute_RProp_plus(train_X, train_y, test_X, test_y, EPOCHS, N_NEURONS, activation1_rprop_plus)
epochs_iRProp_plus, loss_values_iRProp_plus, accuracy_values_iRProp_plus, loss_test_iRProp_plus, accuracy_test_iRProp_plus = execute_iRProp_plus(train_X, train_y, test_X, test_y, EPOCHS, N_NEURONS, activation1_irprop_plus)
epochs_iRProp_minus, loss_values_iRProp_minus, accuracy_values_iRProp_minus, loss_test_iRProp_minus, accuracy_test_iRProp_minus = execute_iRProp_minus(train_X, train_y, test_X, test_y, EPOCHS, N_NEURONS, activation1_irprop_minus)

print(f'SGD TEST SET --- acc: {accuracy_test_SGD:.3f}, ' f'loss: {loss_test_SGD:.3f}')
print(f'Rprop- TEST SET --- acc: {accuracy_test_Rpropmin:.3f}, ' f'loss: {loss_test_Rpropmin:.3f}')
print(f'Rprop+ TEST SET --- acc: {accuracy_test_Rpropplus:.3f}, ' f'loss: {loss_test_Rpropplus:.3f}')
print(f'iRprop+ TEST SET --- acc: {accuracy_test_iRProp_plus:.3f}, ' f'loss: {loss_test_iRProp_plus:.3f}')
print(f'iRprop- TEST SET --- acc: {accuracy_test_iRProp_minus:.3f}, ' f'loss: {loss_test_iRProp_minus:.3f}')

fig, axs = plt.subplots(2)
fig.suptitle("Grafici")
axs[0].plot(epochs_SGD, loss_values_SGD, label='SGD')
axs[0].plot(epochs_Rpropmin, loss_values_Rpropmin, label='Rprop-')
axs[0].plot(epochs_Rpropplus, loss_values_Rpropplus, label='Rprop+')
axs[0].plot(epochs_iRProp_plus, loss_values_iRProp_plus, label='iRprop-')
axs[0].plot(epochs_iRProp_minus, loss_values_iRProp_minus, label='iRprop+')

axs[0].legend()

axs[1].plot(epochs_SGD, accuracy_values_SGD, label='SGD')
axs[1].plot(epochs_Rpropmin, accuracy_values_Rpropmin, label='Rprop-')
axs[1].plot(epochs_Rpropplus, accuracy_values_Rpropplus, label='Rprop+')
axs[1].plot(epochs_iRProp_plus, accuracy_values_iRProp_plus, label='iRprop-')
axs[1].plot(epochs_iRProp_minus, accuracy_values_iRProp_minus, label='iRprop+')
axs.flat[0].set(xlabel='Epochs', ylabel='Loss')
axs.flat[1].set(xlabel='Epochs', ylabel='Accuracy')

axs[1].legend()
plt.show()
'''
'''
#LEAKY RELU
activation1_sgd = Activation_LReLU()
activation1_rprop_minus = Activation_LReLU()
activation1_rprop_plus = Activation_LReLU()
activation1_irprop_plus = Activation_LReLU()
activation1_irprop_minus = Activation_LReLU()


epochs_SGD, loss_values_SGD, accuracy_values_SGD, loss_test_SGD, accuracy_test_SGD = execute_SGD(train_X, train_y, test_X, test_y, EPOCHS, N_NEURONS, activation1_sgd)
epochs_Rpropmin, loss_values_Rpropmin, accuracy_values_Rpropmin, loss_test_Rpropmin, accuracy_test_Rpropmin = execute_RProp_minus(train_X, train_y, test_X, test_y, EPOCHS, N_NEURONS, activation1_rprop_minus)
epochs_Rpropplus, loss_values_Rpropplus, accuracy_values_Rpropplus, loss_test_Rpropplus, accuracy_test_Rpropplus = execute_RProp_plus(train_X, train_y, test_X, test_y, EPOCHS, N_NEURONS, activation1_rprop_plus)
epochs_iRProp_plus, loss_values_iRProp_plus, accuracy_values_iRProp_plus, loss_test_iRProp_plus, accuracy_test_iRProp_plus = execute_iRProp_plus(train_X, train_y, test_X, test_y, EPOCHS, N_NEURONS, activation1_irprop_plus)
epochs_iRProp_minus, loss_values_iRProp_minus, accuracy_values_iRProp_minus, loss_test_iRProp_minus, accuracy_test_iRProp_minus = execute_iRProp_minus(train_X, train_y, test_X, test_y, EPOCHS, N_NEURONS, activation1_irprop_minus)

print(f'SGD TEST SET --- acc: {accuracy_test_SGD:.3f}, ' f'loss: {loss_test_SGD:.3f}')
print(f'Rprop- TEST SET --- acc: {accuracy_test_Rpropmin:.3f}, ' f'loss: {loss_test_Rpropmin:.3f}')
print(f'Rprop+ TEST SET --- acc: {accuracy_test_Rpropplus:.3f}, ' f'loss: {loss_test_Rpropplus:.3f}')
print(f'iRprop+ TEST SET --- acc: {accuracy_test_iRProp_plus:.3f}, ' f'loss: {loss_test_iRProp_plus:.3f}')
print(f'iRprop- TEST SET --- acc: {accuracy_test_iRProp_minus:.3f}, ' f'loss: {loss_test_iRProp_minus:.3f}')

fig, axs = plt.subplots(2)
fig.suptitle("Grafici")
axs[0].plot(epochs_SGD, loss_values_SGD, label='SGD')
axs[0].plot(epochs_Rpropmin, loss_values_Rpropmin, label='Rprop-')
axs[0].plot(epochs_Rpropplus, loss_values_Rpropplus, label='Rprop+')
axs[0].plot(epochs_iRProp_plus, loss_values_iRProp_plus, label='iRprop-')
axs[0].plot(epochs_iRProp_minus, loss_values_iRProp_minus, label='iRprop+')

axs[0].legend()

axs[1].plot(epochs_SGD, accuracy_values_SGD, label='SGD')
axs[1].plot(epochs_Rpropmin, accuracy_values_Rpropmin, label='Rprop-')
axs[1].plot(epochs_Rpropplus, accuracy_values_Rpropplus, label='Rprop+')
axs[1].plot(epochs_iRProp_plus, accuracy_values_iRProp_plus, label='iRprop-')
axs[1].plot(epochs_iRProp_minus, accuracy_values_iRProp_minus, label='iRprop+')
axs.flat[0].set(xlabel='Epochs', ylabel='Loss')
axs.flat[1].set(xlabel='Epochs', ylabel='Accuracy')

axs[1].legend()
plt.show()
'''

#SIGMOID
activation1_sgd = Activation_Sigmoid()
activation1_rprop_minus = Activation_Sigmoid()
activation1_rprop_plus = Activation_Sigmoid()
activation1_irprop_plus = Activation_Sigmoid()
activation1_irprop_minus = Activation_Sigmoid()


epochs_SGD, loss_values_SGD, accuracy_values_SGD, loss_test_SGD, accuracy_test_SGD = execute_SGD(train_X, train_y, test_X, test_y, EPOCHS, N_NEURONS, activation1_sgd)
epochs_Rpropmin, loss_values_Rpropmin, accuracy_values_Rpropmin, loss_test_Rpropmin, accuracy_test_Rpropmin = execute_RProp_minus(train_X, train_y, test_X, test_y, EPOCHS, N_NEURONS, activation1_rprop_minus)
epochs_Rpropplus, loss_values_Rpropplus, accuracy_values_Rpropplus, loss_test_Rpropplus, accuracy_test_Rpropplus = execute_RProp_plus(train_X, train_y, test_X, test_y, EPOCHS, N_NEURONS, activation1_rprop_plus)
epochs_iRProp_plus, loss_values_iRProp_plus, accuracy_values_iRProp_plus, loss_test_iRProp_plus, accuracy_test_iRProp_plus = execute_iRProp_plus(train_X, train_y, test_X, test_y, EPOCHS, N_NEURONS, activation1_irprop_plus)
epochs_iRProp_minus, loss_values_iRProp_minus, accuracy_values_iRProp_minus, loss_test_iRProp_minus, accuracy_test_iRProp_minus = execute_iRProp_minus(train_X, train_y, test_X, test_y, EPOCHS, N_NEURONS, activation1_irprop_minus)

print(f'SGD TEST SET --- acc: {accuracy_test_SGD:.3f}, ' f'loss: {loss_test_SGD:.3f}')
print(f'Rprop- TEST SET --- acc: {accuracy_test_Rpropmin:.3f}, ' f'loss: {loss_test_Rpropmin:.3f}')
print(f'Rprop+ TEST SET --- acc: {accuracy_test_Rpropplus:.3f}, ' f'loss: {loss_test_Rpropplus:.3f}')
print(f'iRprop+ TEST SET --- acc: {accuracy_test_iRProp_plus:.3f}, ' f'loss: {loss_test_iRProp_plus:.3f}')
print(f'iRprop- TEST SET --- acc: {accuracy_test_iRProp_minus:.3f}, ' f'loss: {loss_test_iRProp_minus:.3f}')

fig, axs = plt.subplots(2)
fig.suptitle("Grafici")
axs[0].plot(epochs_SGD, loss_values_SGD, label='SGD')
axs[0].plot(epochs_Rpropmin, loss_values_Rpropmin, label='Rprop-')
axs[0].plot(epochs_Rpropplus, loss_values_Rpropplus, label='Rprop+')
axs[0].plot(epochs_iRProp_plus, loss_values_iRProp_plus, label='iRprop-')
axs[0].plot(epochs_iRProp_minus, loss_values_iRProp_minus, label='iRprop+')

axs[0].legend()

axs[1].plot(epochs_SGD, accuracy_values_SGD, label='SGD')
axs[1].plot(epochs_Rpropmin, accuracy_values_Rpropmin, label='Rprop-')
axs[1].plot(epochs_Rpropplus, accuracy_values_Rpropplus, label='Rprop+')
axs[1].plot(epochs_iRProp_plus, accuracy_values_iRProp_plus, label='iRprop-')
axs[1].plot(epochs_iRProp_minus, accuracy_values_iRProp_minus, label='iRprop+')
axs.flat[0].set(xlabel='Epochs', ylabel='Loss')
axs.flat[1].set(xlabel='Epochs', ylabel='Accuracy')

axs[1].legend()
plt.show()


# epochs, loss_values, accuracy_values, loss_test, accuracy_test = execute_SGD(train_X, train_y, test_X, test_y, EPOCHS, N_NEURONS)
# print(f'TEST SET --- acc: {accuracy_test:.3f}, ' f'loss: {loss_test:.3f}')
# plot_values(epochs, loss_values, accuracy_values, "SGD")



# epochs, loss_values, accuracy_values, loss_test, accuracy_test = execute_RProp_minus(train_X, train_y, test_X, test_y, EPOCHS, N_NEURONS)
# print(f'TEST SET --- acc: {accuracy_test:.3f}, ' f'loss: {loss_test:.3f}')
# plot_values(epochs, loss_values, accuracy_values, "RProp_minus")


# epochs, loss_values, accuracy_values, loss_test, accuracy_test = execute_RProp_plus(train_X, train_y, test_X, test_y, EPOCHS, N_NEURONS)
# print(f'TEST SET --- acc: {accuracy_test:.3f}, ' f'loss: {loss_test:.3f}')
# plot_values(epochs, loss_values, accuracy_values, "RProp_plus")


# epochs_iRProp_plus, loss_values_iRProp_plus, accuracy_values_iRProp_plus, loss_test_iRProp_plus, accuracy_test_iRProp_plus = execute_iRProp_plus(train_X, train_y, test_X, test_y, EPOCHS, N_NEURONS)
# print(f'TEST SET --- acc: {accuracy_test_iRProp_plus:.3f}, ' f'loss: {loss_test_iRProp_plus:.3f}')
# plot_values(epochs_iRProp_plus, loss_values_iRProp_plus, accuracy_values_iRProp_plus, "iRProp_plus")


# epochs_iRProp_minus, loss_values_iRProp_minus, accuracy_values_iRProp_minus, loss_test_iRProp_minus, accuracy_test_iRProp_minus = execute_iRProp_minus(train_X, train_y, test_X, test_y, EPOCHS, N_NEURONS)
# print(f'TEST SET --- acc: {accuracy_test_iRProp_minus:.3f}, ' f'loss: {loss_test_iRProp_minus:.3f}')
# plot_values(epochs_iRProp_minus, loss_values_iRProp_minus, accuracy_values_iRProp_minus, "iRProp_minus")
