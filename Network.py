import numpy as np 
from Loss import quadratic_loss

class Network:

    def __init__(self, epochs, layers):
        self.epochs = epochs
        self.layers = layers

    
    def train(self, input_array, desired_output):
        for i in range(self.epochs):
            output = self.feedforward(input_array, desired_output)
            self.backpropagation(input_array, desired_output)
            print("Epoch " + str(i) + ", loss = " + str(quadratic_loss(output, desired_output)))

        print(output)
        print("Loss: " + str(quadratic_loss(output, desired_output)))

    
    def feedforward(self, input_array, desired_output):
        previous_layer_output = input_array
        for layer in self.layers:
            previous_layer_output = layer.execute_forward_pass(previous_layer_output)

        return previous_layer_output


    def backpropagation(self, input_array, desired_output):
        output_next_layer = None
        weights_next_layer = None

        for layer in reversed(self.layers): 
            layer.execute_backward_pass(
                desired_output=desired_output,
                output_next_layer=output_next_layer,
                weights_next_layer=weights_next_layer
            )

            output_next_layer = layer.get_output()
            weights_next_layer = layer.get_weights()

            