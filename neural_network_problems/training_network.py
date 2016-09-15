import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


np.random.seed(2)


class Network(object):
    def __init__(self, layers):
        self.layers = layers
        self.layer_size = len(layers)
        self.weights = self.define_weights(layers)
        self.bias = self.define_bias(layers)
        # print(self.weights)
        # print(self.bias)
        # exit()

    def define_weights(self, layers):
        weight_list = []
        self.weight1 = np.random.randn(layers[0], layers[1])
        weight_list.append(self.weight1)
        # self.weight2 = np.random.randn(layers[1], layers[2])
        # weight_list.append(self.weight2)
        return weight_list

    def define_bias(self, layers):
        bias_list = []
        self.bias1 = np.random.randn(1, layers[1])
        bias_list.append(self.bias1)
        # self.bias2 = np.random.randn(1, layers[2])
        # bias_list.append(self.bias2)
        return bias_list

    def train_data(self, training_data, iterations, learning_rate):
        for each_iteration in range(iterations):
            print("Iter", each_iteration)
            change_in_bias = [np.zeros(bias.shape) for bias in self.bias]
            change_in_weights = [np.zeros(weights.shape) for weights in self.weights]
            for input, output in training_data:
                delta_bias, delta_weights = self.back_propagation(input, output)
                change_in_bias = [(bias + d_bias) for bias, d_bias in zip(change_in_bias, delta_bias)]
                change_in_weights = [(weights + d_weights) for weights, d_weights in
                                     zip(change_in_weights, delta_weights)]
            self.weights = [w - (learning_rate / len(training_data)) * nw for w, nw in
                            zip(self.weights, change_in_weights)]
            self.bias = [b - (learning_rate / len(training_data)) * nb for b, nb in zip(self.bias, change_in_bias)]
            print(self.weights)
            print(self.bias)
            exit()
        for input, output in training_data:
            self.feed_forward_network(input)

    def back_propagation(self, input, output):
        activation = input
        activation_list = [activation]
        sigmoid_list = []
        activation = self.forward_pass(activation, activation_list, sigmoid_list)
        bias, weights = self.backward_pass(activation, activation_list, sigmoid_list, output)
        return bias, weights

    def forward_pass(self, activation, activation_list, sigmoid_list):
        z1 = np.add(np.dot(activation, self.weight1), self.bias1)
        sigmoid_list.append(z1)
        activation_layer1 = sigmoid(z1)
        activation_list.append(activation_layer1)
        # z2 = np.add(np.dot(activation_layer1, self.weight2), self.bias2)
        # sigmoid_list.append(z2)
        # activation_layer2 = sigmoid(z2)
        # activation_list.append(activation_layer2)
        return activation_list[-1]

    def backward_pass(self, activation, activation_list, sigmoid_list, output):
        change_in_bias = [np.zeros(b.shape) for b in self.bias]
        change_in_weights = [np.zeros(w.shape) for w in self.weights]
        z = sigmoid_list[-1]
        delta = (self.cost_derivative(activation, output)) * sigmoid_prime(z)
        change_in_bias[-1] = delta
        change_in_weights[-1] = np.dot(activation_list[-2].T, delta)
        for layers in range(2, self.layer_size):
            z = sigmoid_list[-layers]
            delta = (np.dot(delta, self.weights[-layers + 1].T)) * (sigmoid_prime(z))
            change_in_bias[-layers] = delta
            change_in_weights[-layers] = np.dot(activation_list[-layers - 1].T, delta)
        return change_in_bias, change_in_weights

    def cost_derivative(self, activation, output):
        cost = (activation - output)
        return cost

    def feed_forward_network(self, input):
        activation = input
        for b, w in zip(self.bias, self.weights):
            z = np.dot(activation, w) + b
            activation = sigmoid(z)
        print("Output is :", activation)
