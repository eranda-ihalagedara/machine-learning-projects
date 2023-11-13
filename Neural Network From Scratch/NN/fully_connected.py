import NN.activations as act
import numpy as np

class Fully_Connected:
    
    # Set initial paramters: size of the layer(size_out), activation function (linear by defaut)
    def __init__(self, size = 1, activation='linear', input_size = None):
        self.size_out = size
        self.size_in = input_size
 
        if activation == 'relu':
            self.g = act.relu
            self.g_prime = act.relu_prime
        elif activation == 'sigmoid':
            self.g = act.sigmoid
            self.g_prime = act.sigmoid_prime
        elif activation == 'linear':
            self.g = act.linear
            self.g_prime = act.linear_prime
        else:
            raise Exception('\'' + str(activation) + '\' activation not found!')
            
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

    
    def backward_pass(self, da_l):
        m = da_l.shape[1]
           
        try:
            self.dz = da_l*self.g_prime(self.z)            
            self.dw = self.dz @ self.a_l_munus_1.T /m
            self.db = np.sum(self.dz, axis=1, keepdims=True)/m

            return self.w.T @ self.dz # Return da_l_munus_1
        
        except Exception as e:
            print('#Layer:', self.layer_id)
            print(e)
            print('w:', self.w.shape, 'wT:', self.w.T.shape)
            print('z:', self.z.shape)
            print('g_prime:', self.g_prime)
            print('dz:', self.dz.shape)
            print('da_l:', da_l.shape)
            
            

    def update_weights(self, learning_rate):
        # Weight update with gradient clipping
        self.w -= learning_rate * np.maximum(-1e0,np.minimum(1e0, self.dw))
        self.b -= learning_rate * np.maximum(-1e0,np.minimum(1e0, self.db))

        # Check if nan in weights
        if np.isnan(self.w).sum() == 1:
            print('Layer:', self.layer_id,'nan in W')
            print('dw:', self.dw)
        if np.isnan(self.b).sum() == 1:
            print('Layer:', self.layer_id,'nan in b')
            print('db:', self.db)

