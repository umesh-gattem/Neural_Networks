import numpy as np


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


t
