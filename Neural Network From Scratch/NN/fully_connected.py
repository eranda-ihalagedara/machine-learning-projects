import NN.activations as act
import numpy as np

class Fully_Connected:
    # Set initial paramters: size of the layer(size_out), activation function (linear by defaut)
    def __init__(self, size = 1, activation='linear'):
        self.size_out = size
 
        if activation == 'ReLU':
            self.g = act.reLu
        elif activation == 'Sigmoid':
            self.g = act.sigmoid
        else :
            self.g = act.linear
    
    # Set input size to the layer and initilize the weight matrix and bias vector when building the a model
    def build(self, size_in):
        self.size_in = size_in
        self.w = np.random.rand(self.size_out,self.size_in)
        self.b = np.random.rand(self.size_out,1)
    
    def forward_pass(self, a):
        self.al_1 = a
        self.z = np.matmul(self.w, a) + self.b
        return self.g(self.z)

    def backward_pass(self):
        pass

