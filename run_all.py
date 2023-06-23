import numpy as np
import matplotlib.pyplot as plt
from neural_network_mnist import execute_SGD, execute_RProp_minus, execute_RProp_plus, execute_iRProp_plus, execute_iRProp_minus, Activation_LReLU, Activation_Sigmoid, Activation_ReLU
from mnist import get_dataset
import threading

def thread_SGD(ret_container: dict, activ_func, weight_init):
    epochs, loss_values, accuracy_values, loss_test, accuracy_test = execute_SGD(train_X, train_y, test_X, test_y, EPOCHS, N_NEURONS, activ_func, weight_init)

    ret_container['epochs'] = epochs
    ret_container['loss'] = loss_values
    ret_container['loss_test'] = loss_test
    ret_container['accuracy'] = accuracy_values
    ret_container['accuracy_test'] = accuracy_test


def thread_Rprop_minus(ret_container: dict, activ_func, weight_init):
    epochs, loss_values, accuracy_values, loss_test, accuracy_test = execute_RProp_minus(train_X, train_y, test_X, test_y, EPOCHS, N_NEURONS, activ_func, weight_init)

    ret_container['epochs'] = epochs
    ret_container['loss'] = loss_values
    ret_container['loss_test'] = loss_test
    ret_container['accuracy'] = accuracy_values
    ret_container['accuracy_test'] = accuracy_test


def thread_Rprop_plus(ret_container: dict, activ_func, weight_init):
    epochs, loss_values, accuracy_values, loss_test, accuracy_test = execute_RProp_plus(train_X, train_y, test_X, test_y, EPOCHS, N_NEURONS, activ_func, weight_init)

    ret_container['epochs'] = epochs
    ret_container['loss'] = loss_values
    ret_container['loss_test'] = loss_test
    ret_container['accuracy'] = accuracy_values
    ret_container['accuracy_test'] = accuracy_test


def thread_iRprop_minus(ret_container: dict, activ_func, weight_init):
    epochs, loss_values, accuracy_values, loss_test, accuracy_test = execute_iRProp_minus(train_X, train_y, test_X, test_y, EPOCHS, N_NEURONS, activ_func, weight_init)

    ret_container['epochs'] = epochs
    ret_container['loss'] = loss_values
    ret_container['loss_test'] = loss_test
    ret_container['accuracy'] = accuracy_values
    ret_container['accuracy_test'] = accuracy_test


def thread_iRprop_plus(ret_container: dict, activ_func, weight_init):
    epochs, loss_values, accuracy_values, loss_test, accuracy_test = execute_iRProp_plus(train_X, train_y, test_X, test_y, EPOCHS, N_NEURONS, activ_func, weight_init)

    ret_container['epochs'] = epochs
    ret_container['loss'] = loss_values
    ret_container['loss_test'] = loss_test
    ret_container['accuracy'] = accuracy_values
    ret_container['accuracy_test'] = accuracy_test


# TEST PARAMETERS
EPOCHS = 1000
N_NEURONS = 150
network_weight_init_rule = "xavier"
normalize_dataset = False
(train_X, train_y), (test_X, test_y) = get_dataset(20000, 5000, normalize_dataset)


# RELU
# activation1_sgd = Activation_ReLU()
# activation1_rprop_minus = Activation_ReLU()
# activation1_rprop_plus = Activation_ReLU()
# activation1_irprop_plus = Activation_ReLU()
# activation1_irprop_minus = Activation_ReLU()

# LEAKY RELU
# activation1_sgd = Activation_LReLU()
# activation1_rprop_minus = Activation_LReLU()
# activation1_rprop_plus = Activation_LReLU()
# activation1_irprop_plus = Activation_LReLU()
# activation1_irprop_minus = Activation_LReLU()

# SIGMOID
activation1_sgd = Activation_Sigmoid()
activation1_rprop_minus = Activation_Sigmoid()
activation1_rprop_plus = Activation_Sigmoid()
activation1_irprop_plus = Activation_Sigmoid()
activation1_irprop_minus = Activation_Sigmoid()

print("Starting test")
print(f"Training set cardinality: {len(train_X)} \nTest set cardinality: {len(test_X)}")
print(f"Neurons: {N_NEURONS}\nWeight Init Rule: {network_weight_init_rule}\nDataset Normalized: {normalize_dataset}\n\n")

# MULTI THREAD EXECUTION
ret_SGD = {}
ret_Rprop_minus = {}
ret_Rprop_plus = {}
ret_iRprop_minus = {}
ret_iRprop_plus = {}

t1 = threading.Thread(target=thread_SGD, args=(ret_SGD, activation1_sgd, network_weight_init_rule))
t2 = threading.Thread(target=thread_Rprop_minus, args=(ret_Rprop_minus, activation1_rprop_minus, network_weight_init_rule))
t3 = threading.Thread(target=thread_Rprop_plus, args=(ret_Rprop_plus, activation1_rprop_plus, network_weight_init_rule))
t4 = threading.Thread(target=thread_iRprop_minus, args=(ret_iRprop_minus, activation1_irprop_minus, network_weight_init_rule))
t5 = threading.Thread(target=thread_iRprop_plus, args=(ret_iRprop_plus, activation1_irprop_plus, network_weight_init_rule))

t1.start()
t2.start()
t3.start()
t4.start()
t5.start()

t1.join()
t2.join()
t3.join()
t4.join()
t5.join()

print(f'SGD TEST SET --- acc: {ret_SGD.get("accuracy_test"):.3f}, ' f'loss: {ret_SGD.get("loss_test"):.3f}')
print(f'Rprop- TEST SET --- acc: {ret_Rprop_minus.get("accuracy_test"):.3f}, ' f'loss: {ret_Rprop_minus.get("loss_test"):.3f}')
print(f'Rprop+ TEST SET --- acc: {ret_Rprop_plus.get("accuracy_test"):.3f}, ' f'loss: {ret_Rprop_plus.get("loss_test"):.3f}')
print(f'iRprop+ TEST SET --- acc: {ret_iRprop_plus.get("accuracy_test"):.3f}, ' f'loss: {ret_iRprop_plus.get("loss_test"):.3f}')
print(f'iRprop- TEST SET --- acc: {ret_iRprop_minus.get("accuracy_test"):.3f}, ' f'loss: {ret_iRprop_minus.get("loss_test"):.3f}')

fig, axs = plt.subplots(2)
fig.suptitle(f"N{N_NEURONS}_{network_weight_init_rule}_data_{'norm' if normalize_dataset else 'not_norm'}")
axs[0].plot(ret_SGD.get("epochs"), ret_SGD.get("loss"), label='SGD')
axs[0].plot(ret_Rprop_minus.get("epochs"), ret_Rprop_minus.get("loss"), label='Rprop-')
axs[0].plot(ret_Rprop_plus.get("epochs"), ret_Rprop_plus.get("loss"), label='Rprop+')
axs[0].plot(ret_Rprop_minus.get("epochs"), ret_iRprop_minus.get("loss"), label='iRprop-')
axs[0].plot(ret_iRprop_plus.get("epochs"), ret_iRprop_plus.get("loss"), label='iRprop+')

axs[0].legend()

axs[1].plot(ret_SGD.get("epochs"), ret_SGD.get("accuracy"), label='SGD')
axs[1].plot(ret_Rprop_minus.get("epochs"), ret_Rprop_minus.get("accuracy"), label='Rprop-')
axs[1].plot(ret_Rprop_plus.get("epochs"), ret_Rprop_plus.get("accuracy"), label='Rprop+')
axs[1].plot(ret_Rprop_minus.get("epochs"), ret_iRprop_minus.get("accuracy"), label='iRprop-')
axs[1].plot(ret_iRprop_plus.get("epochs"), ret_iRprop_plus.get("accuracy"), label='iRprop+')
axs.flat[0].set(xlabel='Epochs', ylabel='Loss')
axs.flat[1].set(xlabel='Epochs', ylabel='Accuracy')

axs[1].legend()
plt.show()


# SINGLE THREAD EXECUTION
# epochs_SGD, loss_values_SGD, accuracy_values_SGD, loss_test_SGD, accuracy_test_SGD = execute_SGD(train_X, train_y, test_X, test_y, EPOCHS, N_NEURONS, activation1_sgd, "xavier")
# epochs_Rpropmin, loss_values_Rpropmin, accuracy_values_Rpropmin, loss_test_Rpropmin, accuracy_test_Rpropmin = execute_RProp_minus(train_X, train_y, test_X, test_y, EPOCHS, N_NEURONS, activation1_rprop_minus, "xavier")
# epochs_Rpropplus, loss_values_Rpropplus, accuracy_values_Rpropplus, loss_test_Rpropplus, accuracy_test_Rpropplus = execute_RProp_plus(train_X, train_y, test_X, test_y, EPOCHS, N_NEURONS, activation1_rprop_plus, "xavier")
# epochs_iRProp_plus, loss_values_iRProp_plus, accuracy_values_iRProp_plus, loss_test_iRProp_plus, accuracy_test_iRProp_plus = execute_iRProp_plus(train_X, train_y, test_X, test_y, EPOCHS, N_NEURONS, activation1_irprop_plus, "xavier")
# epochs_iRProp_minus, loss_values_iRProp_minus, accuracy_values_iRProp_minus, loss_test_iRProp_minus, accuracy_test_iRProp_minus = execute_iRProp_minus(train_X, train_y, test_X, test_y, EPOCHS, N_NEURONS, activation1_irprop_minus, "xavier")
#
# print(f'SGD TEST SET --- acc: {accuracy_test_SGD:.3f}, ' f'loss: {loss_test_SGD:.3f}')
# print(f'Rprop- TEST SET --- acc: {accuracy_test_Rpropmin:.3f}, ' f'loss: {loss_test_Rpropmin:.3f}')
# print(f'Rprop+ TEST SET --- acc: {accuracy_test_Rpropplus:.3f}, ' f'loss: {loss_test_Rpropplus:.3f}')
# print(f'iRprop+ TEST SET --- acc: {accuracy_test_iRProp_plus:.3f}, ' f'loss: {loss_test_iRProp_plus:.3f}')
# print(f'iRprop- TEST SET --- acc: {accuracy_test_iRProp_minus:.3f}, ' f'loss: {loss_test_iRProp_minus:.3f}')
#
# fig, axs = plt.subplots(2)
# fig.suptitle("Grafici")
# axs[0].plot(epochs_SGD, loss_values_SGD, label='SGD')
# axs[0].plot(epochs_Rpropmin, loss_values_Rpropmin, label='Rprop-')
# axs[0].plot(epochs_Rpropplus, loss_values_Rpropplus, label='Rprop+')
# axs[0].plot(epochs_iRProp_plus, loss_values_iRProp_plus, label='iRprop-')
# axs[0].plot(epochs_iRProp_minus, loss_values_iRProp_minus, label='iRprop+')
#
# axs[0].legend()
#
# axs[1].plot(epochs_SGD, accuracy_values_SGD, label='SGD')
# axs[1].plot(epochs_Rpropmin, accuracy_values_Rpropmin, label='Rprop-')
# axs[1].plot(epochs_Rpropplus, accuracy_values_Rpropplus, label='Rprop+')
# axs[1].plot(epochs_iRProp_plus, accuracy_values_iRProp_plus, label='iRprop-')
# axs[1].plot(epochs_iRProp_minus, accuracy_values_iRProp_minus, label='iRprop+')
# axs.flat[0].set(xlabel='Epochs', ylabel='Loss')
# axs.flat[1].set(xlabel='Epochs', ylabel='Accuracy')
#
# axs[1].legend()
# plt.show()


# epochs, loss_values, accuracy_values, loss_test, accuracy_test = execute_SGD(train_X, train_y, test_X, test_y, EPOCHS, N_NEURONS)
# print(f'TEST SET --- acc: {accuracy_test:.3f}, ' f'loss: {loss_test:.3f}')
# plot_values(epochs, loss_values, accuracy_values, "SGD")


# epochs, loss_values, accuracy_values, loss_test, accuracy_test = execute_RProp_minus(train_X, train_y, test_X, test_y, EPOCHS, N_NEURONS)
# print(f'TEST SET --- acc: {accuracy_test:.3f}, ' f'loss: {loss_test:.3f}')
# plot_values(epochs, loss_values, accuracy_values, "RProp_minus")


# epochs, loss_values, accuracy_values, loss_test, accuracy_test = execute_RProp_plus(train_X, train_y, test_X, test_y, EPOCHS, N_NEURONS, Activation_ReLU(), "xavier")
# print(f'TEST SET --- acc: {accuracy_test:.3f}, ' f'loss: {loss_test:.3f}')
# plot_values(epochs, loss_values, accuracy_values, "RProp_plus")


# epochs_iRProp_plus, loss_values_iRProp_plus, accuracy_values_iRProp_plus, loss_test_iRProp_plus, accuracy_test_iRProp_plus = execute_iRProp_plus(train_X, train_y, test_X, test_y, EPOCHS, N_NEURONS)
# print(f'TEST SET --- acc: {accuracy_test_iRProp_plus:.3f}, ' f'loss: {loss_test_iRProp_plus:.3f}')
# plot_values(epochs_iRProp_plus, loss_values_iRProp_plus, accuracy_values_iRProp_plus, "iRProp_plus")


# epochs_iRProp_minus, loss_values_iRProp_minus, accuracy_values_iRProp_minus, loss_test_iRProp_minus, accuracy_test_iRProp_minus = execute_iRProp_minus(train_X, train_y, test_X, test_y, EPOCHS, N_NEURONS)
# print(f'TEST SET --- acc: {accuracy_test_iRProp_minus:.3f}, ' f'loss: {loss_test_iRProp_minus:.3f}')
# plot_values(epochs_iRProp_minus, loss_values_iRProp_minus, accuracy_values_iRProp_minus, "iRProp_minus")
