# from neural_network_problems import nand_gate_network
#
#
# layer_size = [2, 1]
# network = nand_gate_network.Network(layer_size)


from neural_network_problems import sample
import numpy as np

layer_size = [2, 1]
network = sample.Network(layer_size)
training_data = [(np.array([[0, 0]]), np.array([[1]])),
                 (np.array([[1, 0]]), np.array([[1]])),
                 (np.array([[0, 1]]), np.array([[1]])),
                 (np.array([[1, 1]]), np.array([[0]]))]
test_data = training_data
network.SGD(training_data, 1000, 4, 8.0, test_data=test_data)
# network.train_data(training_data, 1000, 4, 1.0, test_data=test_data)
