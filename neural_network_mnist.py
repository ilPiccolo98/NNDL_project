import numpy as np
import math

def get_weights_xavier(n_inputs, n_neurons):
    np.random.seed(0)
    scale = 1/max(1., (2+2)/2.)
    limit = math.sqrt(3.0 * scale)
    weights = np.random.uniform(-limit, limit, size=(n_inputs,n_neurons))
    return weights


def get_weights(n_inputs, n_neurons):
    weights = 0.01 * np.random.randn(n_inputs, n_neurons)
    return weights


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, initialization="xavier"):
        if initialization == "random":
            self.weights = get_weights(n_inputs, n_neurons)
        elif initialization == "xavier":
            self.weights = get_weights_xavier(n_inputs, n_neurons)
        else:
            raise Exception("Invalid weight initialization rule")
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


class Activation_LReLU:
    def __init__(self, alpha=0.01):
        self.alpha = alpha
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0,inputs) + self.alpha * np.minimum(0, inputs)
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs == 0] = 0
        self.dinputs[self.inputs < 0] *= self.alpha


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
    def forward_without_loss(self, inputs):
        self.activation.forward(inputs)
        self.output = self.activation.output
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples


class Optimizer_SGD:
    def __init__(self, learning_rate=1):
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
        if not hasattr(layer, "weights_step_size"):
            layer.weights_step_size = np.ones(layer.dweights.shape) * 0.5
            layer.biases_step_size = np.ones(layer.dbiases.shape) * 0.5
        self.__update_weights(layer)
        self.__update_biases(layer)
    def __update_weights(self, layer):
        same_sign_weight = layer.dweights_cache * layer.dweights > 0
        layer.weights_step_size[same_sign_weight] = np.minimum(layer.weights_step_size[same_sign_weight] * self.positive_eta, self.delta_max)
        different_sign_weight = layer.dweights_cache * layer.dweights < 0
        layer.weights_step_size[different_sign_weight] = np.maximum(layer.weights_step_size[different_sign_weight] * self.negative_eta, self.delta_min)
        layer.weights -= np.sign(layer.dweights) * layer.weights_step_size
    def __update_biases(self, layer):
        same_sign_bias = layer.dbiases_cache * layer.dbiases > 0
        layer.biases_step_size[same_sign_bias] = np.minimum(layer.biases_step_size[same_sign_bias] * self.positive_eta, self.delta_max)
        different_sign_bias = layer.dbiases_cache * layer.dbiases < 0
        layer.biases_step_size[different_sign_bias] = np.maximum(layer.biases_step_size[different_sign_bias] * self.negative_eta, self.delta_min)
        layer.biases -= np.sign(layer.dbiases) * layer.biases_step_size


class Optimizer_RProp_Plus:
    def __init__(self, positive_eta=1.2, negative_eta=0.5, delta_max=50, delta_min=0):
        self.positive_eta = positive_eta
        self.negative_eta = negative_eta
        self.delta_max = delta_max
        self.delta_min = delta_min
    def update_params(self, layer):
        if not hasattr(layer, "weights_step_size"):
            layer.weights_step_size = np.ones(layer.dweights.shape) * 0.5
            layer.biases_step_size = np.ones(layer.dbiases.shape) * 0.5
            layer.delta_w_cache = np.zeros(layer.weights.shape)
            layer.delta_b_cache = np.zeros(layer.biases.shape)        
        self.__update_weights(layer)
        self.__update_biases(layer)
    def __update_weights(self, layer):
        delta_w = np.copy(layer.delta_w_cache)
        same_sign = layer.dweights_cache * layer.dweights > 0
        layer.weights_step_size[same_sign] = np.minimum(layer.weights_step_size[same_sign] * self.positive_eta, self.delta_max)
        delta_w[same_sign] = -np.sign(layer.dweights[same_sign]) * layer.weights_step_size[same_sign]
        layer.weights[same_sign] += delta_w[same_sign]

        different_sign = layer.dweights_cache * layer.dweights < 0
        layer.weights_step_size[different_sign] = np.maximum(layer.weights_step_size[different_sign] * self.negative_eta, self.delta_min)
        layer.weights[different_sign] -= layer.delta_w_cache[different_sign]
        layer.dweights[different_sign] = 0

        zero_sign = layer.dweights_cache * layer.dweights == 0
        delta_w[zero_sign] = -np.sign(layer.dweights[zero_sign]) * layer.weights_step_size[zero_sign]
        layer.weights[zero_sign] += delta_w[zero_sign]

        layer.delta_w_cache = delta_w
    def __update_biases(self, layer):
        delta_b = np.copy(layer.delta_b_cache)
        same_sign = layer.dbiases_cache * layer.dbiases > 0
        layer.biases_step_size[same_sign] = np.minimum(layer.biases_step_size[same_sign] * self.positive_eta, self.delta_max)
        delta_b[same_sign] = -np.sign(layer.dbiases[same_sign]) * layer.biases_step_size[same_sign]
        layer.biases[same_sign] += delta_b[same_sign]

        different_sign = layer.dbiases_cache * layer.dbiases < 0
        layer.biases_step_size[different_sign] = np.maximum(layer.biases_step_size[different_sign] * self.negative_eta, self.delta_min)
        layer.biases[different_sign] -= layer.delta_b_cache[different_sign]
        layer.dbiases[different_sign] = 0

        zero_sign = layer.dbiases_cache * layer.dbiases == 0
        delta_b[zero_sign] = -np.sign(layer.dbiases[zero_sign]) * layer.biases_step_size[zero_sign]
        layer.biases[zero_sign] += delta_b[zero_sign]

        layer.delta_b_cache = delta_b


class Optimizer_iRProp_Plus:
    def __init__(self, positive_eta=1.2, negative_eta=0.5, delta_max=50, delta_min=0):
        self.positive_eta = positive_eta
        self.negative_eta = negative_eta
        self.delta_max = delta_max
        self.delta_min = delta_min
    def update_params(self, layer, loss):
        if not hasattr(layer, "weights_step_size"):
            layer.weights_step_size = np.ones(layer.dweights.shape) * 0.5
            layer.biases_step_size = np.ones(layer.dbiases.shape) * 0.5
            layer.delta_w_cache = np.zeros(layer.weights.shape)
            layer.delta_b_cache = np.zeros(layer.biases.shape)
            layer.loss_cache = 0        
        self.__update_weights(layer, loss)
        self.__update_biases(layer, loss)
        layer.loss_cache = loss
    def __update_weights(self, layer, loss):
        delta_w = np.copy(layer.delta_w_cache)
        same_sign = layer.dweights_cache * layer.dweights > 0
        layer.weights_step_size[same_sign] = np.minimum(layer.weights_step_size[same_sign] * self.positive_eta, self.delta_max)
        delta_w[same_sign] = -np.sign(layer.dweights[same_sign]) * layer.weights_step_size[same_sign]
        layer.weights[same_sign] += delta_w[same_sign]

        different_sign = layer.dweights_cache * layer.dweights < 0
        layer.weights_step_size[different_sign] = np.maximum(layer.weights_step_size[different_sign] * self.negative_eta, self.delta_min)
        if loss > layer.loss_cache:
            layer.weights[different_sign] -= layer.delta_w_cache[different_sign]
        layer.dweights[different_sign] = 0

        zero_sign = layer.dweights_cache * layer.dweights == 0
        delta_w[zero_sign] = -np.sign(layer.dweights[zero_sign]) * layer.weights_step_size[zero_sign]
        layer.weights[zero_sign] += delta_w[zero_sign]

        layer.delta_w_cache = delta_w
    def __update_biases(self, layer, loss):
        delta_b = np.copy(layer.delta_b_cache)
        same_sign = layer.dbiases_cache * layer.dbiases > 0
        layer.biases_step_size[same_sign] = np.minimum(layer.biases_step_size[same_sign] * self.positive_eta, self.delta_max)
        delta_b[same_sign] = -np.sign(layer.dbiases[same_sign]) * layer.biases_step_size[same_sign]
        layer.biases[same_sign] += delta_b[same_sign]

        different_sign = layer.dbiases_cache * layer.dbiases < 0
        layer.biases_step_size[different_sign] = np.maximum(layer.biases_step_size[different_sign] * self.negative_eta, self.delta_min)
        if loss > layer.loss_cache:
            layer.biases[different_sign] -= layer.delta_b_cache[different_sign]
            delta_b[different_sign] = -layer.delta_b_cache[different_sign]
        else:
            delta_b[different_sign] = 0
        layer.dbiases[different_sign] = 0

        zero_sign = layer.dbiases_cache * layer.dbiases == 0
        delta_b[zero_sign] = -np.sign(layer.dbiases[zero_sign]) * layer.biases_step_size[zero_sign]
        layer.biases[zero_sign] += delta_b[zero_sign]

        layer.delta_b_cache = delta_b


class Optimizer_iRProp_Minus:
    def __init__(self, positive_eta=1.2, negative_eta=0.5, delta_max=50, delta_min=0):
        self.positive_eta = positive_eta
        self.negative_eta = negative_eta
        self.delta_max = delta_max
        self.delta_min = delta_min
    def update_params(self, layer):
        if not hasattr(layer, "weights_step_size"):
            layer.weights_step_size = np.ones(layer.dweights.shape) * 0.5
            layer.biases_step_size = np.ones(layer.dbiases.shape) * 0.5
        self.__update_weights(layer)
        self.__update_biases(layer)
    def __update_weights(self, layer):
        same_sign_weight = layer.dweights_cache * layer.dweights > 0
        layer.weights_step_size[same_sign_weight] = np.minimum(layer.weights_step_size[same_sign_weight] * self.positive_eta, self.delta_max)
        different_sign_weight = layer.dweights_cache * layer.dweights < 0
        layer.weights_step_size[different_sign_weight] = np.maximum(layer.weights_step_size[different_sign_weight] * self.negative_eta, self.delta_min)
        layer.dweights[different_sign_weight] = 0
        layer.weights -= np.sign(layer.dweights) * layer.weights_step_size
    def __update_biases(self, layer):
        same_sign_bias = layer.dbiases_cache * layer.dbiases > 0
        layer.biases_step_size[same_sign_bias] = np.minimum(layer.biases_step_size[same_sign_bias] * self.positive_eta, self.delta_max)
        different_sign_bias = layer.dbiases_cache * layer.dbiases < 0
        layer.biases_step_size[different_sign_bias] = np.maximum(layer.biases_step_size[different_sign_bias] * self.negative_eta, self.delta_min)
        layer.dbiases[different_sign_bias] = 0
        layer.biases -= np.sign(layer.dbiases) * layer.biases_step_size


def execute_SGD(train_X, train_y, test_X, test_Y, epochs, n_neurons, activation1):
    dense1 = Layer_Dense(train_X.shape[1], n_neurons)
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


def execute_RProp_minus(train_X, train_y, test_X, test_Y, epochs, n_neurons, activation1):
    dense1 = Layer_Dense(train_X.shape[1], n_neurons)
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


def execute_RProp_plus(train_X, train_y, test_X, test_Y, epochs, n_neurons, activation1):
    dense1 = Layer_Dense(train_X.shape[1], n_neurons)
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


def execute_iRProp_plus(train_X, train_y, test_X, test_Y, epochs, n_neurons, activation1):
    dense1 = Layer_Dense(train_X.shape[1], n_neurons)
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


def execute_iRProp_minus(train_X, train_y, test_X, test_Y, epochs, n_neurons, activation1):
    dense1 = Layer_Dense(train_X.shape[1], n_neurons)
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

class NeuralNetwork:
    def __init__(self, layer1, activation_f, layer2, loss_activation_f, optimizer):
        self.layer1 = layer1
        self.activation_f = activation_f
        self.layer2 = layer2
        self.softmax = loss_activation_f
        self.optimizer = optimizer

    def forward(self, elem):
        self.layer1.forward(elem)
        self.activation_f.forward(self.layer1.output)
        self.layer2.forward(self.activation_f.output)
        self.softmax.forward_without_loss(self.layer2.output)
        prediction = np.argmax(self.softmax.output, axis=1)
        return prediction

def train_network_with_iRprop_plus(train_X, train_y, test_X, test_Y, epochs, n_neurons, activation1):
    dense1 = Layer_Dense(train_X.shape[1], n_neurons)
    dense2 = Layer_Dense(n_neurons, 10)
    loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
    loss_values = []
    accuracy_values = []
    optimizer = Optimizer_iRProp_Plus()
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
    accuracy_test = np.mean(predictions == test_Y)

    print("Training ended.")
    print(f"Results on TEST SET are: accuracy = {accuracy_test}, loss = {loss_test}")

    return NeuralNetwork(dense1, activation1, dense2, loss_activation, optimizer)


def train_network_with_iRprop_minus(train_X, train_y, test_X, test_Y, epochs, n_neurons, activation1):
    dense1 = Layer_Dense(train_X.shape[1], n_neurons)
    dense2 = Layer_Dense(n_neurons, 10)
    loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
    loss_values = []
    accuracy_values = []
    optimizer = Optimizer_iRProp_Minus()
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
    accuracy_test = np.mean(predictions == test_Y)

    print("Training ended.")
    print(f"Results on TEST SET are: accuracy = {accuracy_test}, loss = {loss_test}")

    return NeuralNetwork(dense1, activation1, dense2, loss_activation, optimizer)


def train_network_with_Rprop_plus(train_X, train_y, test_X, test_Y, epochs, n_neurons, activation1):
    dense1 = Layer_Dense(train_X.shape[1], n_neurons)
    dense2 = Layer_Dense(n_neurons, 10)
    loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
    loss_values = []
    accuracy_values = []
    optimizer = Optimizer_RProp_Plus()
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
    accuracy_test = np.mean(predictions == test_Y)

    print("Training ended.")
    print(f"Results on TEST SET are: accuracy = {accuracy_test}, loss = {loss_test}")

    return NeuralNetwork(dense1, activation1, dense2, loss_activation, optimizer)

