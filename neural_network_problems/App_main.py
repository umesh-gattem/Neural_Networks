"""
This python file is used to read csv file using pandas and
defines the input and output for oyr network model.
@since 15-09-2016
"""

import pandas as pd
import numpy as np
from neural_network_problems import network_model

read_file = pd.read_csv("/home/umesh/mycsv_file.csv")
train_data_input = []
train_data_output = []
for index, row in read_file.iterrows():
    input = np.array([row[:-1]])
    output = np.array([[row[len(row)-1]]])
    train_data_input.append(input)
    train_data_output.append(output)

input_nodes = len(row[:-1])
output_nodes = 1
layers = [input_nodes, 6, 8, 9, output_nodes]
epochs = 8000
learning_rate = 0.9

network = network_model.NetworkModel(layers, train_data_input, train_data_output, 10000, learning_rate)


