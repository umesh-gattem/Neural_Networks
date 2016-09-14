from neural_network_problems import network_for_nand_operation
import numpy as np

layer_size = [2, 6, 1]
network = network_for_nand_operation.Network(layer_size)
training_data = [(np.array([[0, 0]]), np.array([[1]])),
                 (np.array([[1, 0]]), np.array([[1]])),
                 (np.array([[0, 1]]), np.array([[1]])),
                 (np.array([[1, 1]]), np.array([[0]]))]
test_data = training_data

for input, output in training_data:
    print(input, output)

# network.train_data(training_data, 10000, 4, 0.9, test_data=test_data)
