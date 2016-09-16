"""
This python file is used to read csv file using pandas and
defines the input and output for oyr network model.
@since 15-09-2016
"""

import pandas as pd
import numpy as np
from neural_network_problems import network_model
from collections import OrderedDict

# from neural_network_problems.network_model import train_network

read_file = pd.read_csv("/home/umesh/mycsv_file.csv")
train_data_input, train_data_output = network_model.get_input_output(read_file)

input_nodes = train_data_input[1].size
output_nodes = train_data_output[1].size
first_hidden_layer_nodes = 6
second_hidden_layer_nodes = 8
third_hidden_layer_nodes = 9
layers_list = [input_nodes, first_hidden_layer_nodes, second_hidden_layer_nodes,
               third_hidden_layer_nodes, output_nodes]
epochs = 10000
learning_rate = 0.9


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


np.random.seed(2)

weight_dictionary = OrderedDict()

weight_dictionary['weight1'] = np.random.randn(layers_list[0], layers_list[1])
weight_dictionary['weight2'] = np.random.randn(layers_list[1], layers_list[2])
weight_dictionary['weight3'] = np.random.randn(layers_list[2], layers_list[3])
weight_dictionary['weight4'] = np.random.randn(layers_list[3], layers_list[4])

bias_dictionary = OrderedDict()
bias_dictionary['bias1'] = np.random.randn(1, layers_list[1])
bias_dictionary['bias2'] = np.random.randn(1, layers_list[2])
bias_dictionary['bias3'] = np.random.randn(1, layers_list[3])
bias_dictionary['bias4'] = np.random.randn(1, layers_list[4])


def define_model(activation, weights_list, bias_list):
    activation_list = [activation]
    sigmoid_list = []
    z1 = np.add(np.dot(activation, weights_list[0]), bias_list[0])
    sigmoid_list.append(z1)
    activation1 = sigmoid(z1)
    activation_list.append(activation1)
    z2 = np.add(np.dot(activation1, weights_list[1]), bias_list[1])
    sigmoid_list.append(z2)
    activation2 = sigmoid(z2)
    activation_list.append(activation2)
    z3 = np.add(np.dot(activation2, weights_list[2]), bias_list[2])
    sigmoid_list.append(z3)
    activation3 = sigmoid(z3)
    activation_list.append(activation3)
    z4 = np.add(np.dot(activation3, weights_list[3]), bias_list[3])
    sigmoid_list.append(z4)
    activation4 = sigmoid(z4)
    activation_list.append(activation4)
    return activation_list, sigmoid_list


weight_list = [weight for weight in weight_dictionary.values()]
bias_list = [bias for bias in bias_dictionary.values()]
network_model.NetworkModel(layers_list, train_data_input,
                           train_data_output, epochs, learning_rate, weight_list, bias_list)
