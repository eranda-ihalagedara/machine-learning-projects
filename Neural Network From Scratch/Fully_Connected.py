import Activations as act
import numpy as np

class Fully_Connected:
    # Set initial paramters: size of the layer(size_out), activation function (linear by defaut)
    def __init__(self, size = 1, activation='linear'):
        self.size_out = size
        self.size_in = size
        if activation == 'ReLU':
            self.activation = act.reLu
        else if activation == 'Sigmoid':
            self.activation = act.sigmoid
        else :
            self.activation = act.linear
    
    # Set input size to the layer and initilize the weight matrix and bias vector when building the a model
    def build(self, size_in)
        self.size_in = size_in
        self.w = np.random.rand(self.size_out,self.size_in)
        self.b = np.random.rand(self.size_out,1)
    
    def forward_pass(self):
        pass

    def backward_pass(self):
        pass

