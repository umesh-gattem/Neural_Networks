import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


class Network(object):
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        np.random.seed(10)
        self.biases = [np.random.randn(y, 1) for y in layer_sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]
        lr = 10
        input_value = np.array([[0, 0]])
        # [0, 1],
        # [1, 0],
        # [1, 1]])
        output_value = np.array([[1, 1, 1, 0]])
        for iter in range(100):
            activations = []
            sigmoid_weights = []
            activation = input_value.T
            print(activation)
            exit()
            activations.append(activation)
            for b, w in zip(self.biases, self.weights):
                z = np.dot(w, activation) + b
                sigmoid_weights.append(z)
                activation = sigmoid(z)
                activations.append(activation)
            delta = 0.5 * ((activation - output_value)**2) * sigmoid_prime(sigmoid_weights[-1]) * lr
            self.biases[-1] = delta
            self.weights[-1] = np.dot(delta, activations[-2].T)
            for layers in range(2, len(layer_sizes)):
                # print("layers :", -layers)
                delta = (np.dot(self.weights[-layers + 1].T, delta)) * lr * (sigmoid_prime(sigmoid_weights[-layers]))
                self.biases[-layers] = delta
                self.weights[-layers] = np.dot(delta, activations[-layers - 1].T)
            print("output for ", iter, " iteration is  :", activation)

