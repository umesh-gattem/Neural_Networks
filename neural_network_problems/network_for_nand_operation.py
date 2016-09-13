import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


class Network(object):
    def __init__(self, layer_sizes):
        np.random.seed(2)
        self.layer_sizes = layer_sizes
        self.biases = [np.random.randn(y, 1) for y in layer_sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]

    def train_data(self, training_data, iterations, mini_batch_size, learn_rate, test_data=None):
        size_of_input = len(training_data)
        for each_iter in range(iterations):
            mini_batch = [training_data[k: k + mini_batch_size] for k in range(0, size_of_input, mini_batch_size)]
            for each_batch in mini_batch:
                self.update_mini_batch(each_batch, learn_rate)
            if test_data:
                print(self.evaluate(test_data))
            else:
                print("Iteration {0} is completed :", format(each_iter))

    def update_mini_batch(self, each_batch, learn_rate):
        initial_bias = [np.zeros(b.shape) for b in self.biases]
        initial_weights = [np.zeros(w.shape) for w in self.weights]
        for input, output in each_batch:
            delta_bias, delta_weights = self.back_propagation(input, output)
            initial_bias = [nb + dnb for nb, dnb in zip(initial_bias, delta_bias)]
            initial_weights = [nw + dnw for nw, dnw in zip(initial_weights, delta_weights)]
        self.weights = [w - (learn_rate / len(each_batch)) * nw for w, nw in zip(self.weights, initial_weights)]
        self.biases = [b - (learn_rate / len(each_batch)) * nb for b, nb in zip(self.biases, initial_bias)]

    def feedforward(self, input):
        activation = input.T
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            activation = sigmoid(z)
        return activation

    def back_propagation(self, input, output):
        change_in_bias = [np.zeros(b.shape) for b in self.biases]
        change_in_weights = [np.zeros(w.shape) for w in self.weights]
        activation_list = []
        sigmoid_weights = []
        activation = input.T
        activation_list.append(activation)
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            sigmoid_weights.append(z)
            activation = sigmoid(z)
            activation_list.append(activation)
        delta = (activation - output) * (sigmoid_prime(sigmoid_weights[-1]))
        change_in_bias[-1] = delta
        change_in_weights[-1] = np.dot(delta, activation_list[-2].T)
        for layers in range(2, len(self.layer_sizes)):
            delta = (np.dot(self.weights[-layers + 1].T, delta)) * (sigmoid_prime(sigmoid_weights[-layers]))
            change_in_bias[-layers] = delta
            change_in_weights[-layers] = np.dot(delta, activation_list[-layers - 1].T)
        return change_in_bias, change_in_weights

    def evaluate(self, test_data):
        output_list = []
        test_results = [((self.feedforward(x)), y) for (x, y) in test_data]
        for x, y in test_results:
            output_list.append(x)
        return output_list
