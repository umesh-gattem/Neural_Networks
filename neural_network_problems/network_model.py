"""
This python file gives the dynamic network model for any problem.
 @since 15-09-2016
"""
import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


def get_input_output(csv_file):
    train_data_input = []
    train_data_output = []
    for index, row in csv_file.iterrows():
        input_data = np.array([row[:-1]])
        output_data = np.array([[row[len(row) - 1]]])
        train_data_input.append(input_data)
        train_data_output.append(output_data)
    return train_data_input, train_data_output


class NetworkModel(object):
    def __init__(self, layers, input, output, epochs, learning_rate, weight_list, bias_list):
        self.train_data_input = input
        self.train_data_output = output
        self.layers = layers
        self.layers_size = len(layers)
        self.train_network(layers, epochs, learning_rate, self.train_data_input,
                           self.train_data_output, weight_list, bias_list)

    def train_network(self, layers, epochs, learning_rate, train_data_input, train_data_output, weight_list, bias_list):
        for each_epoch in range(epochs):
            change_in_bias = [np.zeros(bias.shape) for bias in weight_list]
            change_in_weights = [np.zeros(weights.shape) for weights in bias_list]
            for input, output in zip(train_data_input, train_data_output):
                from neural_network_problems.App_main import define_model
                activation_list, sigmoid_list = define_model(input, weight_list, bias_list)
                delta_weights, delta_bias = self.back_propagation(layers, activation_list,
                                                                  output, weight_list, bias_list,
                                                                  sigmoid_list)
                change_in_bias = [(bias + d_bias) for bias, d_bias in zip(change_in_bias, delta_bias)]
                change_in_weights = [(weights + d_weights) for weights, d_weights in zip(change_in_weights,
                                                                                         delta_weights)]
            weight_list[0] = np.array([(w - (learning_rate / len(train_data_input)) * nw) for w, nw in
                                       zip(weight_list[0], change_in_weights[0])])
            weight_list[1] = np.array([(w - (learning_rate / len(train_data_input)) * nw) for w, nw in
                                       zip(weight_list[1], change_in_weights[1])])
            weight_list[2] = np.array([(w - (learning_rate / len(train_data_input)) * nw) for w, nw in
                                       zip(weight_list[2], change_in_weights[2])])
            weight_list[3] = np.array([(w - (learning_rate / len(train_data_input)) * nw) for w, nw in
                                       zip(weight_list[3], change_in_weights[3])])
            bias_list[0] = np.array([(b - (learning_rate / len(train_data_input)) * nb) for b, nb in
                                     zip(bias_list[0], change_in_bias[0])])
            bias_list[1] = np.array([(b - (learning_rate / len(train_data_input)) * nb) for b, nb in
                                     zip(bias_list[1], change_in_bias[1])])
            bias_list[2] = np.array([(b - (learning_rate / len(train_data_input)) * nb) for b, nb in
                                     zip(bias_list[2], change_in_bias[2])])
            bias_list[3] = np.array([(b - (learning_rate / len(train_data_input)) * nb) for b, nb in
                                     zip(bias_list[3], change_in_bias[3])])
        for input in train_data_input:
            self.test_data(input, weight_list, bias_list)

    def back_propagation(self, layers, activation_list, output, weights, bias, sigmoid_list):
        change_in_bias = [np.zeros(b.shape) for b in bias]
        change_in_weights = [np.zeros(w.shape) for w in weights]
        z = sigmoid_list[-1]
        delta = (self.cost_derivative(activation_list[-1], output)) * sigmoid_prime(z)
        change_in_bias[-1] = delta
        change_in_weights[-1] = np.dot(activation_list[-2].T, delta)
        for layers in range(2, len(layers)):
            z = sigmoid_list[-layers]
            delta = (np.dot(delta, weights[-layers + 1].T)) * (sigmoid_prime(z))
            change_in_bias[-layers] = delta
            change_in_weights[-layers] = np.dot(activation_list[-layers - 1].T, delta)
        return change_in_weights, change_in_bias

    def cost_derivative(self, activation, output):
        cost = (activation - output)
        return cost

    def test_data(self, activation, weight_list, bias_list):
        for w, b in zip(weight_list, bias_list):
            z = np.add(np.dot(activation, w), b)
            activation = sigmoid(z)
        print(activation)
