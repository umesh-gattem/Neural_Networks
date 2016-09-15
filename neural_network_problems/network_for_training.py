import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

np.random.seed(2)


class Network(object):

    def __init__(self, layers):
        self.layers = layers
        self.layer_size = len(layers)
        self.bias = [np.random.randn(1, y) for y in layers[1:]]
        self.weights = [np.random.randn(x, y) for y, x in zip(layers[1:], layers[:-1])]


    def train_data(self, training_data, iterations, learning_rate):
        for each_iteration in range(iterations):
            change_in_bias = [np.zeros(bias.shape) for bias in self.bias]
            change_in_weights = [np.zeros(weights.shape) for weights in self.weights]
            for input, output in training_data:
                delta_bias, delta_weights = self.back_propagation(input, output)
                # print(delta_bias)
                # print(delta_weights)
                change_in_bias = [(bias + d_bias) for bias, d_bias in zip(change_in_bias, delta_bias)]
                change_in_weights = [(weights + d_weights) for weights, d_weights in zip(change_in_weights, delta_weights)]
            self.weights = [w - (learning_rate / len(training_data)) * nw for w, nw in zip(self.weights, change_in_weights)]
            self.bias = [b - (learning_rate / len(training_data)) * nb for b, nb in zip(self.bias, change_in_bias)]
        # print(self.weights)
        # print(self.bias)
        # exit()

    def back_propagation(self, input, output):
        activation = input
        activation_list = [activation]
        sigmoid_list = []
        activation = self.forward_pass(activation, activation_list, sigmoid_list)
        bias, weights = self.backward_pass(activation, activation_list, sigmoid_list, output)
        return bias, weights

    def forward_pass(self, activation, activation_list, sigmoid_list):
        for bias, weight in zip(self.bias, self.weights):
            z = np.add(np.dot(activation, weight), bias)
            sigmoid_list.append(z)
            activation = sigmoid(z)
            activation_list.append(activation)
        return activation

    def backward_pass(self, activation, activation_list, sigmoid_list, output):
        change_in_bias = [np.zeros(b.shape) for b in self.bias]
        change_in_weights = [np.zeros(w.shape) for w in self.weights]
        z = sigmoid_list[-1]
        delta = (self.cost_derivative(activation, output)) * sigmoid_prime(z)
        change_in_bias[-1] = delta
        change_in_weights[-1] = np.dot(activation_list[-2].T, delta)
        for layers in range(2, self.layer_size):
            z = sigmoid_list[-layers]
            delta = (np.dot(delta, self.weights[-layers + 1].T)) * (sigmoid_prime(z))
            change_in_bias[-layers] = delta
            change_in_weights[-layers] = np.dot(activation_list[-layers - 1].T, delta)
        return change_in_bias, change_in_weights

    def cost_derivative(self, activation, output):
        cost = (activation - output)
        return cost

    def feed_forward_network(self, input):
        activation = input
        # print(self.weights)
        # print(self.bias)
        # print("got it")
        for b, w in zip(self.bias, self.weights):
            z = np.dot(activation, w) + b
            activation = sigmoid(z)
        print("Output is :", activation)



