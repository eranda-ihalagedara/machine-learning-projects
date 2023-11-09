import numpy as np
import matplotlib.pyplot as plt

class Model:
    # Set layers
    def __init__(self, layers, learning_rate=0.0001, opt = 'gradient_desccent'):
        self.layers = layers
        self.learning_rate = learning_rate
        self.opt = opt
        self.build()

    # Build each layer
    def build(self):
        size_l = self.layers[0].size_out
        self.layers[0].build(size_l,0)
        for i in range(1, len(self.layers)):
            self.layers[i].build(size_l, i)
            size_l = self.layers[i].size_out

    # Train model
    def train(self, x_train, y_train, batch_size = 32, epochs = 1):
        m = x_train.shape[1]
        self.losses = []
        
        for epoch in range(epochs):
            for i in range(0, m, batch_size):
                x = x_train[:, i:min(i+batch_size,m)]
                a_l = self.predict(x)

                # For MSE as loss function
                da_l = a_l -  y_train[:, i:min(i+batch_size,m)]

                # MSE
                loss = np.sum(np.square(da_l))
                self.losses.append(loss)
                
                print('epoch:', '\tstep:', i%batch_size,'\tloss:',loss)
                
                for layer in list(reversed(self.layers)):
                    da_l = layer.backward_pass(da_l)
                    layer.update_weights(self.learning_rate)

        steps = np.arange(len(self.losses))
        plt.plot(steps, np.array(self.losses))
        plt.show()

    # Predict - forward pass through each layer
    def predict(self, x):
        for layer in self.layers:
            x = layer.forward_pass(x)
        return x
        
        