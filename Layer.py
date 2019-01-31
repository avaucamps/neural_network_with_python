from abc import ABC, abstractmethod

class Layer(ABC):
    @abstractmethod
    def execute_forward_pass(self, input_array):
        pass


    @abstractmethod
    def execute_backward_pass(self, desired_output, output_next_layer=None, weights_next_layer=None):
        pass

    
    @abstractmethod
    def get_output(self):
        pass

    
    @abstractmethod
    def get_weights(self):
        pass