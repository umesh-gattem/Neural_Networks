import csv
import numpy as np
from neural_network_problems import network_for_training

with open("/home/umesh/csvfile.csv", "r+") as file:
    reader = csv.reader(file)
    training_data= []
    for row in reader:
        input = row
        input = [int(i) for i in row]
        input_data = np.array([input[:-1]])
        output_data = np.array([input[len(input)-1]])
        each_tuple = (input_data, output_data)
        training_data.append(each_tuple)

input_layers = len(input[:-1])
output_layers = 1
hidden_layers = 6
layers = [input_layers, hidden_layers, output_layers]

network = network_for_training.Network(layers)
network.train_data(training_data, 10000, 0.9)
for input, output in training_data:
    network.feed_forward_network(input)




# import pandas as pd
#
# data_frame = pd.read_csv("/home/umesh/mycsv_file.csv")
# print(data_frame)

