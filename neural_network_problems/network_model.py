"""
This python file gives the dynamic network model for any problem.
 @since 15-09-2016
"""
import numpy as np
from neural_network_problems import train_data


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))




class NetworkModel(object):
    def __init__(self, layers, input, output, epochs, learning_rate):
        self.train_data_input = input
        self.train_data_output = output
        self.layers = layers
        self.layers_size = len(layers)
        self.bias = self.define_bias(layers)
        print(self.bias[0].shape)
        # exit()
        self.weights = self.define_weights(layers)
        # print(self.weights)
        # print(self.bias)
        # exit()
        self.train_network(epochs, learning_rate, self.train_data_input, self.train_data_output)


        # print(self.train_data_input)
        # print(self.train_data_output)

    def define_weights(self, layers):
        weight_list = []
        self.weight1 = np.random.randn(layers[0], layers[1])
        weight_list.append(self.weight1)
        self.weight2 = np.random.randn(layers[1], layers[2])
        weight_list.append(self.weight2)
        # print(self.weight2)
        self.weight3 = np.random.randn(layers[2], layers[3])
        weight_list.append(self.weight3)
        self.weight4 = np.random.randn(layers[3], layers[4])
        weight_list.append(self.weight4)
        return weight_list

    def define_bias(self, layers):
        bias_list = []
        self.bias1 = np.random.randn(1, layers[1])
        bias_list.append(self.bias1)
        self.bias2 = np.random.randn(1, layers[2])
        bias_list.append(self.bias2)
        self.bias3 = np.random.randn(1, layers[3])
        bias_list.append(self.bias3)
        self.bias4 = np.random.randn(1, layers[4])
        bias_list.append(self.bias4)
        return bias_list

    def train_network(self, epochs, learning_rate, train_data_input, train_data_output):
        for each_epoch in range(epochs):
            # print(self.bias)
            change_in_bias = [np.zeros(bias.shape) for bias in self.bias]
            change_in_weights = [np.zeros(weights.shape) for weights in self.weights]
            for input, output in zip(train_data_input, train_data_output):
                delta_weights, delta_bias = train_data.algo_to_train_data(self.layers, input, output, self.weights, self.bias)
                # print(delta_bias)
                # print(delta_weights)
                change_in_bias = [(bias + d_bias) for bias, d_bias in zip(change_in_bias, delta_bias)]
                change_in_weights = [(weights + d_weights) for weights, d_weights in zip(change_in_weights,
                                                                                         delta_weights)]
            self.weights = [w - (learning_rate / len(train_data_input)) * nw for w, nw in
                                zip(self.weights, change_in_weights)]
            self.bias = [b - (learning_rate / len(train_data_input)) * nb for b, nb in zip(self.bias, change_in_bias)]
            #     # print(change_in_weights)
            #     # print(change_in_weights[-1])
            #     # exit()
            # for w, nw in zip(self.weight1, change_in_weights[0]):
            #      self.weight1 = w - (learning_rate / len(train_data_input)) * nw
            # self.weight2 = [(w - (learning_rate / len(train_data_input)) * nw) for w, nw in
            #                 zip(self.weight2, change_in_weights[1])]
            # print(self.weight2)
            # exit()
            # self.weight3 = [w - (learning_rate / len(train_data_input)) * nw for w, nw in
            #                 zip(self.weight3, change_in_weights[2])]
            # self.weight4 = [w - (learning_rate / len(train_data_input)) * nw for w, nw in
            #                 zip(self.weight4, change_in_weights[3])]
            # self.weights[0] = self.weight1
            # self.weights[1] = self.weight2
            # self.weights[2] = self.weight3
            # self.weights[3] = self.weight4
            # self.bias1 = [b - (learning_rate / len(train_data_input)) * nb for b, nb in
            #               zip(self.bias1, change_in_bias[0])]
            # self.bias2 = [b - (learning_rate / len(train_data_input)) * nb for b, nb in
            #              zip(self.bias2, change_in_bias[1])]
            # self.bias3 = [b - (learning_rate / len(train_data_input)) * nb for b, nb in
            #              zip(self.bias3, change_in_bias[2])]
            # self.bias4 = [b - (learning_rate / len(train_data_input)) * nb for b, nb in
            #              zip(self.bias4, change_in_bias[3])]
            # self.bias[0] = self.bias1
            # self.bias[1] = self.bias2
            # self.bias[2] = self.bias3
            # self.bias[3] = self.bias4
            # print(self.bias)
            # # for bias in self.bias:
            # #     print(bias.shape)
            # # exit()

        for input in train_data_input:
            self.test_data(input)

    def test_data(self, activation):
        for w, b in zip(self.weights, self.bias):
            z = np.add(np.dot(activation, w), b)
            activation = sigmoid(z)
        print(activation)
