import numpy as np


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def algo_to_train_data(layers, activation, output, weights, bias):
    activation_list = [activation]
    sigmoid_list = []
    for b, w in zip(bias, weights):
        z = np.add(np.dot(activation, w), b)
        sigmoid_list.append(z)
        activation = sigmoid(z)
        activation_list.append(activation)
    change_in_bias = [np.zeros(b.shape) for b in bias]
    change_in_weights = [np.zeros(w.shape) for w in weights]
    z = sigmoid_list[-1]
    delta = (cost_derivative(activation_list[-1], output)) * sigmoid_prime(z)
    change_in_bias[-1] = delta
    change_in_weights[-1] = np.dot(activation_list[-2].T, delta)
    for layers in range(2, len(layers)):
        z = sigmoid_list[-layers]
        delta = (np.dot(delta, weights[-layers + 1].T)) * (sigmoid_prime(z))
        change_in_bias[-layers] = delta
        change_in_weights[-layers] = np.dot(activation_list[-layers - 1].T, delta)
    return change_in_weights, change_in_bias


def cost_derivative(activation, output):
    cost = (activation - output)
    return cost
