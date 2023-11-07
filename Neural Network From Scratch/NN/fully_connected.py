import NN.activations as act
import numpy as np

class Fully_Connected:
    # Set initial paramters: size of the layer(size_out), activation function (linear by defaut)
    def __init__(self, size = 1, activation='linear'):
        self.size_out = size
 
        if activation == 'ReLU':
            self.g = act.relu
            self.g_prime = act.relu_prime
        elif activation == 'Sigmoid':
            self.g = act.sigmoid
            self.g_prime = act.sigmoid_prime
        else :
            self.g = act.linear
            self.g_prime = act.linear_prime
    
    # Set input size to the layer and initilize the weight matrix and bias vector when building the a model
    def build(self, size_in):
        self.size_in = size_in
        self.w = np.random.rand(self.size_out,self.size_in)
        self.b = np.random.rand(self.size_out,1)
    
    def forward_pass(self, a):
        self.a_l_munus_1 = a
        self.z = np.matmul(self.w, a) + self.b
        return self.g(self.z)

    def backward_pass(self, da_l, m=1):
        self.dz = da_l*self.g_prime(self.z)
        self.dw = np.matmul(self.dz,self.a_l_munus_1.T)/m
        self.db = np.sum(self.dz, axis=1, keepdims=True)/m
        return np.matmul(self.w.T, self.dz) # Return da_l_munus_1

    def update_weights(self, learning_rate):
        self.w -= learning_rate*self.dw
        self.b -= learning_rate*self.db

