import numpy as np
from keras.datasets import mnist
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.dweights_cache = np.zeros((n_inputs, n_neurons))
        self.dbiases_cache = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
    def backward(self, dvalues):
        if hasattr(self, "dweights"):
            self.dweights_cache = np.copy(self.dweights)
            self.dbiases_cache = np.copy(self.dbiases)
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)


class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


class Activation_Sigmoid:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))
    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.output) * self.output


class Activation_Linear:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = inputs
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        

class Activation_Softmax:
    def forward(self, inputs):
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss


class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples


class Loss_MeanSquaredError(Loss):
    def forward(self, y_pred, y_true):
        sample_losses = np.mean((y_true - y_pred)**2, axis=-1)
        return sample_losses
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])
        self.dinputs = -2 * (y_true - dvalues) / outputs
        self.dinputs = self.dinputs / samples


class Activation_Softmax_Loss_CategoricalCrossentropy():
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()
    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples


class Optimizer_SGD:
    def __init__(self, learning_rate=0.0001):
        self.learning_rate = learning_rate
    def update_params(self, layer):
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases


class Optimizer_RProp_Minus:
    def __init__(self, positive_eta=1.2, negative_eta=0.5, delta_max=50, delta_min=0):
        self.positive_eta = positive_eta
        self.negative_eta = negative_eta
        self.delta_max = delta_max
        self.delta_min = delta_min
    def update_params(self, layer):
        if not hasattr(layer, "delta_weights"):
            layer.delta_weights = np.ones(layer.dweights.shape) * 0.5
            layer.delta_biases = np.ones(layer.dbiases.shape) * 0.5
        self.__update_weights(layer)
        self.__update_biases(layer)
    def __update_weights(self, layer):
        same_sign_weight = layer.dweights_cache * layer.dweights > 0
        layer.delta_weights[same_sign_weight] = np.minimum(layer.delta_weights[same_sign_weight] * self.positive_eta, self.delta_max)
        different_sign_weight = layer.dweights_cache * layer.dweights < 0
        layer.delta_weights[different_sign_weight] = np.maximum(layer.delta_weights[different_sign_weight] * self.negative_eta, self.delta_min)
        layer.weights -= np.sign(layer.dweights) * layer.delta_weights
    def __update_biases(self, layer):
        same_sign_bias = layer.dbiases_cache * layer.dbiases > 0
        layer.delta_biases[same_sign_bias] = np.minimum(layer.delta_biases[same_sign_bias] * self.positive_eta, self.delta_max)
        different_sign_bias = layer.dbiases_cache * layer.dbiases < 0
        layer.delta_biases[different_sign_bias] = np.maximum(layer.delta_biases[different_sign_bias] * self.negative_eta, self.delta_min)
        layer.biases -= np.sign(layer.dbiases) * layer.delta_biases


class Optimizer_RProp_Plus:
    def __init__(self, positive_eta=1.2, negative_eta=0.5, delta_max=50, delta_min=0):
        self.positive_eta = positive_eta
        self.negative_eta = negative_eta
        self.delta_max = delta_max
        self.delta_min = delta_min
    def update_params(self, layer):
        if not hasattr(layer, "delta_weights"):
            layer.delta_weights = np.ones(layer.dweights.shape) * 0.5
            layer.delta_biases = np.ones(layer.dbiases.shape) * 0.5
            layer.delta_w_cache = np.zeros(layer.weights.shape)
            layer.delta_b_cache = np.zeros(layer.biases.shape)        
        self.__update_weights(layer)
        self.__update_biases(layer)
    def __update_weights(self, layer):
        delta_w = np.copy(layer.delta_w_cache)
        same_sign = layer.dweights_cache * layer.dweights > 0
        layer.delta_weights[same_sign] = np.minimum(layer.delta_weights[same_sign] * self.positive_eta, self.delta_max)
        delta_w[same_sign] = -np.sign(layer.dweights[same_sign]) * layer.delta_weights[same_sign]
        layer.weights[same_sign] += delta_w[same_sign]

        different_sign = layer.dweights_cache * layer.dweights < 0
        layer.delta_weights[different_sign] = np.maximum(layer.delta_weights[different_sign] * self.negative_eta, self.delta_min)
        layer.weights[different_sign] -= layer.delta_w_cache[different_sign]
        delta_w[different_sign] = -layer.delta_w_cache[different_sign]
        layer.dweights[different_sign] = 0

        zero_sign = layer.dweights_cache * layer.dweights == 0
        delta_w[zero_sign] = -np.sign(layer.dweights[zero_sign]) * layer.delta_weights[zero_sign]
        layer.weights[zero_sign] += delta_w[zero_sign]

        layer.delta_w_cache = delta_w
    def __update_biases(self, layer):
        delta_b = np.copy(layer.delta_b_cache)
        same_sign = layer.dbiases_cache * layer.dbiases > 0
        layer.delta_biases[same_sign] = np.minimum(layer.delta_biases[same_sign] * self.positive_eta, self.delta_max)
        delta_b[same_sign] = -np.sign(layer.dbiases[same_sign]) * layer.delta_biases[same_sign]
        layer.biases[same_sign] += delta_b[same_sign]

        different_sign = layer.dbiases_cache * layer.dbiases < 0
        layer.delta_biases[different_sign] = np.maximum(layer.delta_biases[different_sign] * self.negative_eta, self.delta_min)
        layer.biases[different_sign] -= layer.delta_b_cache[different_sign]
        delta_b[different_sign] = -layer.delta_b_cache[different_sign]
        layer.dbiases[different_sign] = 0

        zero_sign = layer.dbiases_cache * layer.dbiases == 0
        delta_b[zero_sign] = -np.sign(layer.dbiases[zero_sign]) * layer.delta_biases[zero_sign]
        layer.biases[zero_sign] += delta_b[zero_sign]

        layer.delta_b_cache = delta_b


class Optimizer_iRProp_Plus:
    def __init__(self, positive_eta=1.2, negative_eta=0.5, delta_max=50, delta_min=0):
        self.positive_eta = positive_eta
        self.negative_eta = negative_eta
        self.delta_max = delta_max
        self.delta_min = delta_min
    def update_params(self, layer, loss):
        if not hasattr(layer, "delta_weights"):
            layer.delta_weights = np.ones(layer.dweights.shape) * 0.5
            layer.delta_biases = np.ones(layer.dbiases.shape) * 0.5
            layer.delta_w_cache = np.zeros(layer.weights.shape)
            layer.delta_b_cache = np.zeros(layer.biases.shape)
            layer.loss_cache = 0        
        self.__update_weights(layer, loss)
        self.__update_biases(layer, loss)
    def __update_weights(self, layer, loss):
        delta_w = np.copy(layer.delta_w_cache)
        same_sign = layer.dweights_cache * layer.dweights > 0
        layer.delta_weights[same_sign] = np.minimum(layer.delta_weights[same_sign] * self.positive_eta, self.delta_max)
        delta_w[same_sign] = -np.sign(layer.dweights[same_sign]) * layer.delta_weights[same_sign]
        layer.weights[same_sign] += delta_w[same_sign]

        different_sign = layer.dweights_cache * layer.dweights < 0
        layer.delta_weights[different_sign] = np.maximum(layer.delta_weights[different_sign] * self.negative_eta, self.delta_min)
        if loss > layer.loss_cache:
            layer.weights[different_sign] -= layer.delta_w_cache[different_sign]
            delta_w[different_sign] = -layer.delta_w_cache[different_sign]
        else:
            delta_w[different_sign] = 0
        layer.dweights[different_sign] = 0

        zero_sign = layer.dweights_cache * layer.dweights == 0
        delta_w[zero_sign] = -np.sign(layer.dweights[zero_sign]) * layer.delta_weights[zero_sign]
        layer.weights[zero_sign] += delta_w[zero_sign]

        layer.delta_w_cache = delta_w
    def __update_biases(self, layer, loss):
        delta_b = np.copy(layer.delta_b_cache)
        same_sign = layer.dbiases_cache * layer.dbiases > 0
        layer.delta_biases[same_sign] = np.minimum(layer.delta_biases[same_sign] * self.positive_eta, self.delta_max)
        delta_b[same_sign] = -np.sign(layer.dbiases[same_sign]) * layer.delta_biases[same_sign]
        layer.biases[same_sign] += delta_b[same_sign]

        different_sign = layer.dbiases_cache * layer.dbiases < 0
        layer.delta_biases[different_sign] = np.maximum(layer.delta_biases[different_sign] * self.negative_eta, self.delta_min)
        if loss > layer.loss_cache:
            layer.biases[different_sign] -= layer.delta_b_cache[different_sign]
            delta_b[different_sign] = -layer.delta_b_cache[different_sign]
        else:
            delta_b[different_sign] = 0
        layer.dbiases[different_sign] = 0

        zero_sign = layer.dbiases_cache * layer.dbiases == 0
        delta_b[zero_sign] = -np.sign(layer.dbiases[zero_sign]) * layer.delta_biases[zero_sign]
        layer.biases[zero_sign] += delta_b[zero_sign]

        layer.delta_b_cache = delta_b


class Optimizer_iRProp_Minus:
    def __init__(self, positive_eta=1.2, negative_eta=0.5, delta_max=50, delta_min=0):
        self.positive_eta = positive_eta
        self.negative_eta = negative_eta
        self.delta_max = delta_max
        self.delta_min = delta_min
    def update_params(self, layer):
        if not hasattr(layer, "delta_weights"):
            layer.delta_weights = np.ones(layer.dweights.shape) * 0.5
            layer.delta_biases = np.ones(layer.dbiases.shape) * 0.5
        self.__update_weights(layer)
        self.__update_biases(layer)
    def __update_weights(self, layer):
        same_sign_weight = layer.dweights_cache * layer.dweights > 0
        layer.delta_weights[same_sign_weight] = np.minimum(layer.delta_weights[same_sign_weight] * self.positive_eta, self.delta_max)
        different_sign_weight = layer.dweights_cache * layer.dweights < 0
        layer.delta_weights[different_sign_weight] = np.maximum(layer.delta_weights[different_sign_weight] * self.negative_eta, self.delta_min)
        layer.dweights[different_sign_weight] = 0
        layer.weights -= np.sign(layer.dweights) * layer.delta_weights
    def __update_biases(self, layer):
        same_sign_bias = layer.dbiases_cache * layer.dbiases > 0
        layer.delta_biases[same_sign_bias] = np.minimum(layer.delta_biases[same_sign_bias] * self.positive_eta, self.delta_max)
        different_sign_bias = layer.dbiases_cache * layer.dbiases < 0
        layer.delta_biases[different_sign_bias] = np.maximum(layer.delta_biases[different_sign_bias] * self.negative_eta, self.delta_min)
        layer.dbiases[different_sign_bias] = 0
        layer.biases -= np.sign(layer.dbiases) * layer.delta_biases

def plot_values(epochs, loss_values, accuracy_values, suptitle):
    fig, axs = plt.subplots(2)
    fig.suptitle(suptitle)
    axs[0].plot(epochs, loss_values)
    axs[1].plot(epochs, accuracy_values)
    axs.flat[0].set(xlabel='Epochs', ylabel='Loss')
    axs.flat[1].set(xlabel='Epochs', ylabel='Accuracy')
    plt.show()


def execute_SGD(train_X, train_y, test_X, test_Y, epochs, n_neurons):
    dense1 = Layer_Dense(train_X.shape[1], n_neurons)
    activation1 = Activation_ReLU()
    dense2 = Layer_Dense(n_neurons, 10)
    loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
    optimizer = Optimizer_SGD()
    loss_values = []
    accuracy_values = []
    for epoch in range(epochs):
        dense1.forward(train_X)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        loss = loss_activation.forward(dense2.output, train_y)
        loss_values.append(loss)
        predictions = np.argmax(loss_activation.output, axis=1)
        if len(train_y.shape) == 2:
            train_y = np.argmax(train_y, axis=1)
        accuracy = np.mean(predictions==train_y)
        accuracy_values.append(accuracy)
        if not epoch % 100:
            print(f'epoch: {epoch}, ' f'acc: {accuracy:.3f}, ' f'loss: {loss:.3f}')
        loss_activation.backward(loss_activation.output, train_y)
        dense2.backward(loss_activation.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
    dense1.forward(test_X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss_test = loss_activation.forward(dense2.output, test_Y)
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(test_Y.shape) == 2:
        test_Y = np.argmax(test_Y, axis=1)
    accuracy_test = np.mean(predictions==test_Y)
    return range(epochs), loss_values, accuracy_values, loss_test, accuracy_test


def execute_RProp_minus(train_X, train_y, test_X, test_Y, epochs, n_neurons):
    dense1 = Layer_Dense(train_X.shape[1], n_neurons)
    activation1 = Activation_ReLU()
    dense2 = Layer_Dense(n_neurons, 10)
    loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
    optimizer = Optimizer_RProp_Minus()
    loss_values = []
    accuracy_values = []
    for epoch in range(epochs):
        dense1.forward(train_X)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        loss = loss_activation.forward(dense2.output, train_y)
        loss_values.append(loss)
        predictions = np.argmax(loss_activation.output, axis=1)
        if len(train_y.shape) == 2:
            train_y = np.argmax(train_y, axis=1)
        accuracy = np.mean(predictions==train_y)
        accuracy_values.append(accuracy)
        if not epoch % 100:
            print(f'epoch: {epoch}, ' f'acc: {accuracy:.3f}, ' f'loss: {loss:.3f}')
        loss_activation.backward(loss_activation.output, train_y)
        dense2.backward(loss_activation.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
    dense1.forward(test_X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss_test = loss_activation.forward(dense2.output, test_Y)
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(test_Y.shape) == 2:
        test_Y = np.argmax(test_Y, axis=1)
    accuracy_test = np.mean(predictions==test_Y)
    return range(epochs), loss_values, accuracy_values, loss_test, accuracy_test


def execute_RProp_plus(train_X, train_y, test_X, test_Y, epochs, n_neurons):
    dense1 = Layer_Dense(train_X.shape[1], n_neurons)
    activation1 = Activation_ReLU()
    dense2 = Layer_Dense(n_neurons, 10)
    loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
    optimizer = Optimizer_RProp_Plus()
    loss_values = []
    accuracy_values = []
    for epoch in range(epochs):
        dense1.forward(train_X)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        loss = loss_activation.forward(dense2.output, train_y)
        loss_values.append(loss)
        predictions = np.argmax(loss_activation.output, axis=1)
        if len(train_y.shape) == 2:
            train_y = np.argmax(train_y, axis=1)
        accuracy = np.mean(predictions==train_y)
        accuracy_values.append(accuracy)
        if not epoch % 100:
            print(f'epoch: {epoch}, ' f'acc: {accuracy:.3f}, ' f'loss: {loss:.3f}')
        loss_activation.backward(loss_activation.output, train_y)
        dense2.backward(loss_activation.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
    dense1.forward(test_X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss_test = loss_activation.forward(dense2.output, test_Y)
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(test_Y.shape) == 2:
        test_Y = np.argmax(test_Y, axis=1)
    accuracy_test = np.mean(predictions==test_Y)
    return range(epochs), loss_values, accuracy_values, loss_test, accuracy_test


def execute_iRProp_plus(train_X, train_y, test_X, test_Y, epochs, n_neurons):
    dense1 = Layer_Dense(train_X.shape[1], n_neurons)
    activation1 = Activation_ReLU()
    dense2 = Layer_Dense(n_neurons, 10)
    loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
    optimizer = Optimizer_iRProp_Plus()
    loss_values = []
    accuracy_values = []
    for epoch in range(epochs):
        dense1.forward(train_X)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        loss = loss_activation.forward(dense2.output, train_y)
        loss_values.append(loss)
        predictions = np.argmax(loss_activation.output, axis=1)
        if len(train_y.shape) == 2:
            train_y = np.argmax(train_y, axis=1)
        accuracy = np.mean(predictions==train_y)
        accuracy_values.append(accuracy)
        if not epoch % 100:
            print(f'epoch: {epoch}, ' f'acc: {accuracy:.3f}, ' f'loss: {loss:.3f}')
        loss_activation.backward(loss_activation.output, train_y)
        dense2.backward(loss_activation.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)
        optimizer.update_params(dense1, loss)
        optimizer.update_params(dense2, loss)
    dense1.forward(test_X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss_test = loss_activation.forward(dense2.output, test_Y)
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(test_Y.shape) == 2:
        test_Y = np.argmax(test_Y, axis=1)
    accuracy_test = np.mean(predictions==test_Y)
    return range(epochs), loss_values, accuracy_values, loss_test, accuracy_test


def execute_iRProp_minus(train_X, train_y, test_X, test_Y, epochs, n_neurons):
    dense1 = Layer_Dense(train_X.shape[1], n_neurons)
    activation1 = Activation_ReLU()
    dense2 = Layer_Dense(n_neurons, 10)
    loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
    optimizer = Optimizer_iRProp_Minus()
    loss_values = []
    accuracy_values = []
    for epoch in range(epochs):
        dense1.forward(train_X)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        loss = loss_activation.forward(dense2.output, train_y)
        loss_values.append(loss)
        predictions = np.argmax(loss_activation.output, axis=1)
        if len(train_y.shape) == 2:
            train_y = np.argmax(train_y, axis=1)
        accuracy = np.mean(predictions==train_y)
        accuracy_values.append(accuracy)
        if not epoch % 100:
            print(f'epoch: {epoch}, ' f'acc: {accuracy:.3f}, ' f'loss: {loss:.3f}')
        loss_activation.backward(loss_activation.output, train_y)
        dense2.backward(loss_activation.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
    dense1.forward(test_X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss_test = loss_activation.forward(dense2.output, test_Y)
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(test_Y.shape) == 2:
        test_Y = np.argmax(test_Y, axis=1)
    accuracy_test = np.mean(predictions==test_Y)
    return range(epochs), loss_values, accuracy_values, loss_test, accuracy_test


def overlap_plots(values, epochs, xlabel, ylabel):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for value in values:
        plt.plot(epochs, value)
    plt.show()


(train_X, train_y), (test_X, test_y) = get_dataset(100, 100)
EPOCHS = 1000
N_NEURONS = 64

'''
epochs, loss_values, accuracy_values, loss_test, accuracy_test = execute_SGD(train_X, train_y, test_X, test_y, EPOCHS, N_NEURONS)
print(f'TEST SET --- acc: {accuracy_test:.3f}, ' f'loss: {loss_test:.3f}')
plot_values(epochs, loss_values, accuracy_values, "SGD")
'''

'''
epochs, loss_values, accuracy_values, loss_test, accuracy_test = execute_RProp_minus(train_X, train_y, test_X, test_y, EPOCHS, N_NEURONS)
print(f'TEST SET --- acc: {accuracy_test:.3f}, ' f'loss: {loss_test:.3f}')
plot_values(epochs, loss_values, accuracy_values, "RProp_minus")
'''
'''
epochs, loss_values, accuracy_values, loss_test, accuracy_test = execute_RProp_plus(train_X, train_y, test_X, test_y, EPOCHS, N_NEURONS)
print(f'TEST SET --- acc: {accuracy_test:.3f}, ' f'loss: {loss_test:.3f}')
plot_values(epochs, loss_values, accuracy_values, "RProp_plus")
'''

epochs_iRProp_plus, loss_values_iRProp_plus, accuracy_values_iRProp_plus, loss_test_iRProp_plus, accuracy_test_iRProp_plus = execute_iRProp_plus(train_X, train_y, test_X, test_y, EPOCHS, N_NEURONS)
print(f'TEST SET --- acc: {accuracy_test_iRProp_plus:.3f}, ' f'loss: {loss_test_iRProp_plus:.3f}')
plot_values(epochs_iRProp_plus, loss_values_iRProp_plus, accuracy_values_iRProp_plus, "iRProp_plus")

'''
epochs_iRProp_minus, loss_values_iRProp_minus, accuracy_values_iRProp_minus, loss_test_iRProp_minus, accuracy_test_iRProp_minus = execute_iRProp_minus(train_X, train_y, test_X, test_y, EPOCHS, N_NEURONS)
print(f'TEST SET --- acc: {accuracy_test_iRProp_minus:.3f}, ' f'loss: {loss_test_iRProp_minus:.3f}')
plot_values(epochs_iRProp_minus, loss_values_iRProp_minus, accuracy_values_iRProp_minus, "iRProp_minus")
'''


#overlap_plots([accuracy_values_iRProp_plus, accuracy_values_iRProp_minus], epochs_iRProp_minus, "Epochs", "Loss")
#overlap_plots([accuracy_values_iRProp_plus, accuracy_values_iRProp_minus], epochs_iRProp_minus, "Epochs", "Accuracy")

'''
dense1 = Layer_Dense(train_X.shape[1], 64)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(64, 10)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
optimizer = Optimizer_SGD()
rprop_minus = Optimizer_RProp_Minus()
rprop_plus = Optimizer_RProp_Plus()
irprop_plus = Optimizer_iRProp_Plus()
irprop_minus = Optimizer_iRProp_Minus()
loss_values = []
EPOCHS = 100
accuracy_values = []
for epoch in range(EPOCHS):
    dense1.forward(train_X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output, train_y)
    loss_values.append(loss)
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(train_y.shape) == 2:
        train_y = np.argmax(train_y, axis=1)
    accuracy = np.mean(predictions==train_y)
    accuracy_values.append(accuracy)
    #if not epoch % 100:
    print(f'epoch: {epoch}, ' f'acc: {accuracy:.3f}, ' f'loss: {loss:.3f}')
    loss_activation.backward(loss_activation.output, train_y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)
    
    #optimizer.update_params(dense1)
    #optimizer.update_params(dense2)
    #rprop_minus.update_params(dense1)
    #rprop_minus.update_params(dense2)
    #rprop_plus.update_params(dense1)
    #rprop_plus.update_params(dense2)
    #irprop_plus.update_params(dense1, loss)
    #irprop_plus.update_params(dense2, loss)
    irprop_minus.update_params(dense1)
    irprop_minus.update_params(dense2)

epochs = range(EPOCHS)
plot_values(epochs, loss_values, accuracy_values)
'''