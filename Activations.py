import numpy as np

def relu(z):
    return max(z, 0)


def sigmoid(z):
    return (1.0 / (1 + np.exp(-z)))


def sigmoid_derivative(z):
    return z * (1.0 - z)


def softmax(z, nb_classes):
    predictions = np.exp(z)
    return predictions / np.sum(nb_classes)