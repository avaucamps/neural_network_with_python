import numpy as np
from Layer import Layer
from Activations import *

class FullyConnectedLayer(Layer):

    def __init__(self, input_size, size, is_output_layer, activation):
        assert activation == 'relu' or activation == 'softmax' or activation == 'sigmoid', "Activation function not supported."
        self.activation = activation
        self.size = size
        self.is_output_layer = is_output_layer
        self.weights = self.initialized_weights((input_size, size))
        self.bias = self.initialized_bias(size)
        self.input = np.zeros((size))
        self.output = np.zeros((size))


    def initialized_weights(self, size):
        return np.random.rand(size[0], size[1])


    def initialized_bias(self, size):
        return np.zeros((size, 1))


    def execute_forward_pass(self, input_array):
        self.input = input_array
        z = np.dot(self.input, self.weights)

        if self.activation == 'relu':
            self.output = relu(z)
        elif self.activation == 'sigmoid':
            self.output = sigmoid(z)
        elif self.activation == 'softmax':
            self.ouptut = softmax(z, self.size)

        return self.output


    def execute_backward_pass(self, desired_output, output_next_layer=None, weights_next_layer=None):
        if self.is_output_layer:
            self.backward_pass_output_layer(desired_output)
        else:
            self.backward_pass_hidden_layer(desired_output, output_next_layer, weights_next_layer)


    def backward_pass_output_layer(self, desired_output):
        if self.activation == 'sigmoid':
            d_weights = np.dot(
                self.input.T, 
                (desired_output - self.output) * sigmoid_derivative(self.output)
            )
            self.weights += d_weights


    def backward_pass_hidden_layer(self, desired_output, output_next_layer, weights_next_layer):
        if self.activation == 'sigmoid':
            d_weights = np.dot(
                self.input.T, 
                np.dot(
                    (desired_output - output_next_layer) * sigmoid_derivative(output_next_layer), 
                    weights_next_layer.T
                ) * sigmoid_derivative(self.output)
            )
            self.weights += d_weights


    def get_output(self):
        return self.output

    
    def get_weights(self):
        return self.weights