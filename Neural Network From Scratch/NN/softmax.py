import NN.activations as act
import numpy as np

class Softmax:
    
    # Set initial parameters: size of the layer(size_out), input_size
    def __init__(self, size = 1, input_size = None):
        self.size_out = size
        self.size_in = input_size
        self.g = act.softmax

        
    # Set input size to the layer and initialize the weight matrix and bias vector when building the model
    def build(self, size_in, layer):
        self.layer_id=layer
        self.size_in = size_in
        self.w = np.random.rand(self.size_out,self.size_in)-0.5
        self.b = np.zeros([self.size_out,1])

    
    def forward_pass(self, a):
        self.a_l_munus_1 = a
        self.z = np.matmul(self.w, a) + self.b
        return self.g(self.z)

    
    def backward_pass(self, dz):
        m = dz.shape[1]
        self.dz = dz 
        self.dw = self.dz @ self.a_l_munus_1.T /m
        self.db = np.sum(self.dz, axis=1, keepdims=True)/m

        return self.w.T @ self.dz # Return da_l_munus_1
    
            
    def update_weights(self, learning_rate, grad_clip = 1e1):
        # Weight update with gradient clipping
        self.w -= learning_rate * np.maximum(-grad_clip,np.minimum(grad_clip, self.dw))
        self.b -= learning_rate * np.maximum(-grad_clip,np.minimum(grad_clip, self.db))

        # Check if nan in weights
        if np.isnan(self.w).sum() == 1:
            print('Layer:', self.layer_id,'nan in W')
            print('dw:', self.dw)
        if np.isnan(self.b).sum() == 1:
            print('Layer:', self.layer_id,'nan in b')
            print('db:', self.db)
