from neural_network_problems import network_for_training
import numpy as np


layers = [2, 6, 1]
network = network_for_training.Network(layers)
training_data = [(np.array([[0, 0]]), np.array([[0]])),
                 (np.array([[1, 0]]), np.array([[1]])),
                 (np.array([[0, 1]]), np.array([[1]])),
                 (np.array([[1, 1]]), np.array([[1]]))]
network.train_data(training_data, 10000, 0.9)

print(training_data)
# network.feed_forward_network(print(input) for input, output in training_data)
for input, output in training_data:
    network.feed_forward_network(input)